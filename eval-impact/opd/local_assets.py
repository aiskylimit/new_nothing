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


def parse_manifest(path, project_root):
    root = Path(project_root).resolve()
    entries = []
    for line in Path(path).read_text().splitlines():
        fields = shlex.split(line, comments=True)
        if not fields:
            continue
        if len(fields) != 3 or fields[0] not in ("--hf", "--url"):
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


def download(manifest, root):
    existing = Path(root) / "download_manifest.json"
    records = json.loads(existing.read_text())["entries"] if existing.is_file() else []
    for entry in parse_manifest(manifest, root):
        target = Path(entry["destination"])
        target.mkdir(parents=True, exist_ok=True)
        if entry["kind"] == "--hf":
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=entry["source"], local_dir=target,
                ignore_patterns=["*.bin", "*.pt", "*.onnx", "*.gguf", "original/*"])
            records.append(entry)
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
    args = p.parse_args()
    if args.action == "download":
        download(args.manifest, args.project_root)
    else:
        prepare(args.project_root, args.train_per_domain, args.calib_per_domain)


if __name__ == "__main__":
    main()
