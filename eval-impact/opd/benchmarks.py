"""Pinned held-out instruction benchmarks; preserve raw prompts and references."""
import argparse
import hashlib
from pathlib import Path
from .data import digest, read_jsonl, save_json, write_jsonl


BENCHMARKS = {
    "self-inst": dict(label="SelfInst", repo="MiniLLM/self-inst",
        revision="ad9edcc55ec5623f6d9d2472963bda3701681af8", file="valid.jsonl",
        n_samples=242, sha256="18922e8ac8a6034ecf0ec8026e24473a30297aaa2e7182d6beb56682586af525"),
    "vicuna": dict(label="VicunaEval", repo="MiniLLM/Vicuna",
        revision="05a18702266a6c51547034268d6d58bce214bd77", file="valid.jsonl",
        n_samples=80, sha256="04f5fabc7e2660a778ef86f7d2779f69a8f108b892074517566b7d714a269c3a"),
    "sinst": dict(label="S-NI", repo="MiniLLM/sinst",
        revision="84b8356a5b631002f92ad73adebd5ac9db5612cd", file="11_/valid.jsonl",
        n_samples=1694, sha256="8c62f31543f02f3dc44ceb62436ec633ab7add9a94307a7dd1977f90378576a6"),
}
ALL_BENCHMARKS = ("dolly", *BENCHMARKS)


def normalized_rows(path, name):
    spec = BENCHMARKS[name]
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError(f"{name}: source checksum mismatch")
    source = read_jsonl(path)
    if len(source) != spec["n_samples"]:
        raise ValueError(f"{name}: unexpected test size {len(source)}")
    rows = []
    for i, row in enumerate(source):
        prompt, response = row.get("prompt"), row.get("output")
        references = response if isinstance(response, list) else [response]
        if not isinstance(prompt, str) or not prompt.strip() or not references or any(
                not isinstance(r, str) or not r.strip() for r in references):
            raise ValueError(f"{name}: invalid prompt/reference at row {i}")
        prompt = prompt.replace("<n>", "\n")  # Same convention as the legacy loader.
        rows.append({"id": f"{name}:valid:{i}", "prompt": prompt,
                     "response": response, "prompt_format": "raw", "domain": name,
                     "prompt_sha256": digest(prompt),
                     "source": f"{spec['repo']}@{spec['revision']}/{spec['file']}"})
    return rows


def prepare(root, names=tuple(BENCHMARKS), download=False):
    root = Path(root)
    for name in names:
        spec = BENCHMARKS[name]
        raw = root / "raw/instruction_benchmarks" / name / spec["file"]
        if not raw.is_file():
            if not download:
                raise FileNotFoundError(f"Download benchmark data first: {raw}")
            import requests
            url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{spec['file']}"
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest() != spec["sha256"]:
                raise ValueError(f"{name}: download checksum mismatch")
            raw.parent.mkdir(parents=True, exist_ok=True)
            temporary = raw.with_suffix(".partial")
            temporary.write_bytes(response.content)
            temporary.replace(raw)
        rows = normalized_rows(raw, name)
        target = root / "data/opd/benchmarks" / name
        write_jsonl(target / "test.jsonl", rows)
        save_json(target / "manifest.json", {**spec, "benchmark": name,
            "prepared_sha256": digest((target / "test.jsonl").read_text()),
            "references": "all original references; best per metric",
            "prompt_format": "raw; <n> replaced with newline; no chat template"})
        print(f"{name}: {len(rows)} test samples -> {target}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["download", "prepare"])
    p.add_argument("--project-root", required=True)
    p.add_argument("--benchmarks", nargs="+", choices=list(BENCHMARKS), default=list(BENCHMARKS))
    args = p.parse_args()
    prepare(args.project_root, args.benchmarks, download=args.action == "download")


if __name__ == "__main__":
    main()
