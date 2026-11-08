import os
import re
from pathlib import Path

import pyarrow.parquet as pq


def main():
    root = Path(os.environ.get("OPENWEBTEXT_DATASET_PATH", "data/raw/openwebtext-100k"))
    # Read only train shards from the provisioned HF snapshot, never a Hub ID.
    files = sorted(root.glob("data/train-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No train Parquet shards in {root / 'data'}. Provision download.txt first."
        )
    output = Path(os.environ.get("OPENWEBTEXT_TEXT_PATH", "data/prepared/openwebtext/data.txt"))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".txt.partial")
    num = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for shard in files:
                for batch in pq.ParquetFile(shard).iter_batches(batch_size=1024, columns=["text"]):
                    for text in batch.column(0).to_pylist():
                        handle.write(re.sub(r"\n+", "<@x(x!>", text) + "\n")
                        num += 1
                        if num >= 250000:
                            break
                    if num >= 250000:
                        break
                if num >= 250000:
                    break
        if num == 0:
            raise ValueError(f"Empty OpenWebText dataset: {root}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print("Number of lines:", num)


if __name__ == "__main__":
    main()
