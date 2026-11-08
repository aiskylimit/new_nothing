"""Create generation prompts using only the newly downloaded MiniLLM raw file."""

import argparse
import json
from pathlib import Path


def prepare(raw_path, output_dir, dev_num=1000):
    with Path(raw_path).open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not 0 < dev_num < len(records):
        raise ValueError(f"Need more than {dev_num} Dolly records; got {len(records)}")
    # Same split as process_data_dolly.py; never mix valid.jsonl into training.
    splits = {"dev": records[:dev_num], "train": records[dev_num:]}
    for record in records:
        if not all(key in record for key in ("instruction", "input", "output", "prompt")):
            raise ValueError("MiniLLM/dolly raw.jsonl is missing required fields")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        target = output_dir / f"{split}.jsonl"
        temporary = target.with_suffix(".jsonl.partial")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                for record in rows:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Prepared {len(rows)} {split} prompts from {raw_path}: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dev-num", type=int, default=1000)
    args = parser.parse_args()
    prepare(args.raw_path, args.output_dir, args.dev_num)
