"""Dataset preparation with deterministic, disjoint training/calibration samples."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def normalize(row, domain, source, index):
    prompt = row.get("prompt") or row.get("question") or row.get("problem") or row.get("instruction")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Missing text prompt: {source}:{index}")
    context = row.get("context", "")
    if context:
        prompt += "\n\n" + context
    response = row.get("response", row.get("output", row.get("answer", row.get("solution", ""))))
    return {"id": f"{domain}:{digest(prompt)[:20]}", "prompt": prompt,
            "response": str(response), "domain": domain, "source": source,
            "source_index": index, "prompt_sha256": digest(prompt)}


def split_samples(rows, train_n, calib_n, seed):
    unique = {}
    for row in rows:
        unique.setdefault(row["prompt_sha256"], row)
    values = sorted(unique.values(), key=lambda r: r["id"])
    random.Random(seed).shuffle(values)
    if train_n < 1 or calib_n < 2 or len(values) < train_n + calib_n:
        raise ValueError(f"Need {train_n + calib_n} unique prompts, got {len(values)}")
    return values[calib_n:calib_n + train_n], values[:calib_n]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--train-per-domain", type=int, default=4000)
    p.add_argument("--calib-per-domain", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--calibration-seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--local", action="append", default=[], metavar="DOMAIN=JSONL")
    p.add_argument("--hf", action="store_true", help="Download Dolly and GSM8K TRAIN splits")
    args = p.parse_args()
    groups, revisions = {}, {}
    if args.hf:
        from datasets import load_dataset
        for domain, name, config in [("dolly", "databricks/databricks-dolly-15k", None),
                                      ("gsm8k", "openai/gsm8k", "main")]:
            ds = load_dataset(name, config, split="train")
            revisions[domain] = {"dataset": name, "config": config, "split": "train",
                                 "fingerprint": ds._fingerprint}
            groups[domain] = [normalize(r, domain, f"{name}/train", i) for i, r in enumerate(ds)]
    for spec in args.local:
        domain, path = spec.split("=", 1)
        if domain in groups:
            raise ValueError(f"Repeated domain {domain}")
        groups[domain] = [normalize(r, domain, str(Path(path).resolve()), i)
                          for i, r in enumerate(read_jsonl(path))]
        revisions[domain] = {"file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    if not groups:
        p.error("Specify --hf or --local DOMAIN=JSONL")
    out = Path(args.out)
    train, calib = {}, {}
    for domain, rows in groups.items():
        train[domain], calib[domain] = split_samples(rows, args.train_per_domain,
                                                    args.calib_per_domain, args.seed)
    # Prevent overlap even when the same question appears in multiple input domains.
    held_out = {r["prompt_sha256"] for rows in calib.values() for r in rows}
    for domain in train:
        train[domain] = [r for r in train[domain] if r["prompt_sha256"] not in held_out]
        if not train[domain]:
            raise ValueError(f"No training samples remain for {domain}")
        write_jsonl(out / domain / "train.jsonl", train[domain])
        write_jsonl(out / domain / "calibration.jsonl", calib[domain])
    # Same total N for mixture and each single-domain run; balanced quotas.
    domains = sorted(train)
    mixed_train, mixed_calib = [], []
    for i, domain in enumerate(domains):
        nt = args.train_per_domain // len(domains) + (i < args.train_per_domain % len(domains))
        nc = args.calib_per_domain // len(domains) + (i < args.calib_per_domain % len(domains))
        if len(train[domain]) < nt or len(calib[domain]) < nc:
            raise ValueError(f"Insufficient samples for a balanced mixed budget: {domain}")
        mixed_train.extend(train[domain][:nt])
        mixed_calib.extend(calib[domain][:nc])
    random.Random(args.seed).shuffle(mixed_train)
    random.Random(args.seed).shuffle(mixed_calib)
    write_jsonl(out / "mixed" / "train.jsonl", mixed_train)
    write_jsonl(out / "mixed" / "calibration.jsonl", mixed_calib)
    train_hashes = {r["prompt_sha256"] for values in train.values() for r in values}
    for seed in args.calibration_seeds:
        mixed_repeat = []
        for i, domain in enumerate(domains):
            if seed == args.seed:
                repeat = calib[domain]
            else:
                pool = {r["prompt_sha256"]: r for r in groups[domain]
                        if r["prompt_sha256"] not in train_hashes}
                candidates = sorted(pool.values(), key=lambda r: r["id"])
                random.Random(seed).shuffle(candidates)
                if len(candidates) < args.calib_per_domain:
                    raise ValueError(f"Insufficient held-out pool for calibration resampling: {domain}")
                repeat = candidates[:args.calib_per_domain]
            write_jsonl(out / domain / f"calibration-seed{seed}.jsonl", repeat)
            nc = args.calib_per_domain // len(domains) + (i < args.calib_per_domain % len(domains))
            mixed_repeat.extend(repeat[:nc])
        random.Random(seed).shuffle(mixed_repeat)
        write_jsonl(out / "mixed" / f"calibration-seed{seed}.jsonl", mixed_repeat)
    save_json(out / "manifest.json", {"seed": args.seed, "calibration_seeds": args.calibration_seeds, "sources": revisions,
              "counts": {d: {"train": len(train[d]), "calibration": len(calib[d])} for d in domains},
              "mixed_counts": {"train": len(mixed_train), "calibration": len(mixed_calib)},
              "note": "Only records from train splits are sampled; test records are never used for training or calibration."})
    print(out.resolve())


if __name__ == "__main__":
    main()
