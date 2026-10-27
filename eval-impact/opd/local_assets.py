"""Download the supplied manifest and prepare train/eval data without Hub reads."""
import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse
from .data import read_jsonl, write_jsonl, save_json


def prepare_reference_dolly(root, calib_n=256):
    """Prepare the downloaded MiniLLM-format Dolly splits without Hub reads."""
    from .data import normalize, digest
    import random
    root = Path(root).resolve()
    source = root / "raw/dolly_reference/original"
    for split in ("train", "dev", "valid"):
        if not (source / f"{split}.jsonl").is_file():
            raise FileNotFoundError(f"Download Dolly first with project_commands.sh download: {source}")
    splits = {s: [dict(normalize(r, "dolly-dskd", f"bundled-dolly/{s}", i),
                       id=f"dolly-dskd:{s}:{i}", prompt_format="raw")
                  for i, r in enumerate(read_jsonl(source / f"{s}.jsonl"))]
              for s in ("train", "dev", "valid")}
    test_hashes = {r["prompt_sha256"] for r in splits["valid"]}
    train = [r for r in splits["train"] if r["prompt_sha256"] not in test_hashes]
    forbidden = test_hashes | {r["prompt_sha256"] for r in train}
    dev = {r["prompt_sha256"]: r for r in splits["dev"] if r["prompt_sha256"] not in forbidden}
    if len(dev) < calib_n:
        raise ValueError("Not enough disjoint Dolly dev prompts")
    out = root / "data/opd/dolly-dskd"
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "test.jsonl", splits["valid"])
    for seed in (42, 43, 44):
        rows = sorted(dev.values(), key=lambda r: r["id"])
        random.Random(seed).shuffle(rows)
        write_jsonl(out / f"calibration-seed{seed}.jsonl", rows[:calib_n])
        if seed == 42:
            write_jsonl(out / "calibration.jsonl", rows[:calib_n])
    save_json(out / "manifest.json", {"source": "bundled processed Dolly (MiniLLM format)",
        "source_sha256": {s: digest((source / f"{s}.jsonl").read_text()) for s in splits},
        "original_counts": {s: len(rows) for s, rows in splits.items()},
        "train_count": len(train), "removed_train_rows_overlapping_test": len(splits["train"]) - len(train),
        "unique_disjoint_dev_pool": len(dev), "calibration_count": calib_n,
        "test_count": len(splits["valid"]), "prompt_format": "raw; no BOS/chat template/extra newline",
        "note": "Training duplicates retained except test overlap; calibration excludes both train and test."})


def parse_manifest(path, project_root):
    root = Path(project_root).resolve()
    entries = []
    for line in Path(path).read_text().splitlines():
        fields = shlex.split(line, comments=True)
        if not fields:
            continue
        if len(fields) != 3 or fields[0] not in ("--hf", "--hf-dataset", "--url"):
            raise ValueError(f"Invalid download manifest line: {line}")
        kind, source, target = fields
        marker = "/impact_opd/"
        if marker not in target:
            raise ValueError("Manifest destination must be inside impact_opd")
        destination = (root / target.split(marker, 1)[1]).resolve()
        if not destination.is_relative_to(root):
            raise ValueError("Download destination escapes project root")
        entries.append({"kind": kind, "source": source, "destination": str(destination)})
    return entries


def verify_dolly_snapshot(target):
    """Check only the selected original JSONL files; no remote manifest paths are followed."""
    target = Path(target)
    manifest = json.loads((target / "manifest.json").read_text())
    expected = {r["path"]: r for r in manifest["files"]}
    for split in ("train", "dev", "valid"):
        name = f"original/{split}.jsonl"
        path = target / name
        if name not in expected or not path.is_file():
            raise ValueError(f"Incomplete Dolly snapshot: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]["sha256"]:
            raise ValueError(f"Dolly checksum mismatch: {name}")


def download(manifest, root):
    existing = Path(root) / "download_manifest.json"
    records = json.loads(existing.read_text())["entries"] if existing.is_file() else []
    for entry in parse_manifest(manifest, root):
        target = Path(entry["destination"])
        target.mkdir(parents=True, exist_ok=True)
        if entry["kind"] == "--hf-dataset":
            from huggingface_hub import snapshot_download
            reference = json.loads((Path(__file__).resolve().parents[1] / "HF_DATASET.json").read_text())
            if entry["source"] != reference["repo_id"]:
                raise ValueError("Dataset must match the pinned HF_DATASET.json reference")
            revision = reference["revision"]
            snapshot_download(repo_id=entry["source"], repo_type="dataset", revision=revision,
                local_dir=target, allow_patterns=["original/*.jsonl", "manifest.json", "provenance.json", "README.md", "LICENSE"])
            verify_dolly_snapshot(target)
            records.append({**entry, "repo_type": "dataset", "resolved_commit": revision})
        elif entry["kind"] == "--hf":
            from huggingface_hub import snapshot_download, HfApi
            revision = HfApi().model_info(entry["source"]).sha
            snapshot_download(repo_id=entry["source"], revision=revision, local_dir=target,
                ignore_patterns=["*.bin", "*.pt", "*.onnx", "*.gguf", "original/*"])
            records.append({**entry, "resolved_commit": revision})
        else:
            name = Path(urlparse(entry["source"]).path).name
            dest = target / name
            if not dest.exists() or dest.stat().st_size == 0:
                tmp = dest.with_suffix(dest.suffix + ".partial")
                import requests
                with requests.get(entry["source"], timeout=120, stream=True) as response, tmp.open("wb") as stream:
                    response.raise_for_status()
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                tmp.replace(dest)
            records.append({**entry, "file": str(dest), "bytes": dest.stat().st_size,
                            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()})
        print(entry["destination"], flush=True)
    dolly_snapshot = Path(root) / "raw/dolly_reference"
    if (dolly_snapshot / "manifest.json").is_file():
        verify_dolly_snapshot(dolly_snapshot)
    unique = {(r["kind"], r["source"]): r for r in records}
    save_json(Path(root) / "download_manifest.json", {"entries": list(unique.values())})


def prepare(root, train_n=4000, calib_n=256):
    import pyarrow.parquet as pq
    root = Path(root).resolve()
    gsm_train = root / "raw/gsm8k/train-00000-of-00001.parquet"
    gsm_test = root / "raw/gsm8k/test-00000-of-00001.parquet"
    math_test = root / "raw/math500/test-00000-of-00001.parquet"
    math_json = root / "raw/math500/test.jsonl"
    dolly = root / "raw/dolly/databricks-dolly-15k.jsonl"
    for path in (gsm_train, gsm_test, math_json, dolly):
        if not path.is_file():
            raise FileNotFoundError(f"Download raw asset first: {path}")
    import pyarrow as pa
    pq.write_table(pa.Table.from_pylist(read_jsonl(math_json)), math_test)
    rows = pq.read_table(gsm_train).to_pylist()
    converted = root / "raw/gsm8k/train.jsonl"
    write_jsonl(converted, rows)
    subprocess.run([sys.executable, "-m", "opd.data", "--local", f"dolly={dolly}",
        "--local", f"gsm8k={converted}", "--out", str(root / "data/opd"),
        "--train-per-domain", str(train_n), "--calib-per-domain", str(calib_n),
        "--calibration-seeds", "42", "43", "44", "--seed", "42"], check=True)
    # Reject train/test contamination using prompt text, independent of local path names.
    from .data import digest
    tests = pq.read_table(gsm_test).to_pylist() + pq.read_table(math_test).to_pylist()
    test_hashes = {digest(r.get("question", r.get("problem", ""))) for r in tests}
    for domain in ("dolly", "gsm8k", "mixed"):
        for path in (root / "data/opd" / domain).glob("*.jsonl"):
            if any(r["prompt_sha256"] in test_hashes for r in read_jsonl(path)):
                raise ValueError(f"Train/calibration prompt overlaps evaluation test set: {path}")
    counts = {"gsm8k_test": pq.read_metadata(gsm_test).num_rows,
              "math500_test": pq.read_metadata(math_test).num_rows}
    if counts != {"gsm8k_test": 1319, "math500_test": 500}:
        raise ValueError(f"Unexpected benchmark sizes: {counts}")
    save_json(root / "data/local_eval_manifest.json", {"counts": counts,
              "gsm8k_test": str(gsm_test), "math500_test": str(math_test),
              "train_test_prompt_overlap": False})


def write_eval_tasks(raw_root, out):
    """Keep native graders/few-shot examples; override only dataset storage."""
    import importlib.util
    import yaml
    spec = importlib.util.find_spec("lm_eval")
    if spec is None:
        raise RuntimeError("lm_eval is required to locate the native task YAMLs")
    tasks = Path(next(iter(spec.submodule_search_locations))) / "tasks"
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    mapping = {"minerva_math500": "opd_math500_local", "gsm8k_cot": "opd_gsm8k_local"}
    for task, local in mapping.items():
        math = task == "minerva_math500"
        native = tasks / ("minerva_math/minerva_math500.yaml" if math else "gsm8k/gsm8k-cot.yaml")
        parquet = Path(raw_root).resolve() / ("math500" if math else "gsm8k") / "test-00000-of-00001.parquet"
        if not native.is_file() or not parquet.is_file():
            raise FileNotFoundError(native if not native.is_file() else parquet)
        config = {"include": str(native), "task": local, "dataset_path": "parquet",
                  "dataset_name": None, "dataset_kwargs": {"data_files": {"test": str(parquet)}},
                  "training_split": None, "validation_split": None, "test_split": "test"}
        (out / f"{local}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return mapping


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["download", "prepare"])
    p.add_argument("--project-root", required=True)
    p.add_argument("--manifest", default="download.txt")
    p.add_argument("--train-per-domain", type=int, default=4000)
    p.add_argument("--calib-per-domain", type=int, default=256)
    p.add_argument("--profile", choices=["llama2-tinyllama", "math-small", "tiny"], default="llama2-tinyllama")
    args = p.parse_args()
    if args.action == "download":
        download(args.manifest, args.project_root)
    else:
        if args.profile == "llama2-tinyllama":
            prepare_reference_dolly(args.project_root, args.calib_per_domain)
        else:
            prepare(args.project_root, args.train_per_domain, args.calib_per_domain)


if __name__ == "__main__":
    main()
