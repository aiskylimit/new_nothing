# Evaluation data

`src/fetch_eval.py` materializes these four files from the local Hugging Face
snapshots listed in `download.txt`:

- `aime24.jsonl`
- `aime25.jsonl`
- `amc12.jsonl`
- `math500.jsonl`

Each line has the normalized shape `{"question": ..., "answer": ...}`. The
files are shared by all four baseline runs.

