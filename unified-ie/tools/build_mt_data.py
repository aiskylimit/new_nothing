"""Builds data/general_domain/mt_<v>: every row of the single-task folders in one folder, for one student
trained on EE, NER and RE at once.
    python tools/build_mt_data.py v2   ->  mt_v2 = ee_v2 + ner_v2 + re_tacred + re_fewrel
Rows are copied unchanged (same prompts, "dataset", "task", "gold_eval"), folder after folder, so each
split is a plain concatenation; the train sampler shuffles. RE has one label format, so both versions
share it. Tokenize with --max-prompt-length 480 (NER needs it)."""
import os
import sys

v = sys.argv[1]
root = "data/general_domain"
parts = [f"ee_{v}", f"ner_{v}", "re_tacred", "re_fewrel"]
out = os.path.join(root, f"mt_{v}")
os.makedirs(out, exist_ok=True)
for split in ("train", "dev", "dev_full", "test"):
    n = 0
    with open(os.path.join(out, f"{split}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for p in parts:
            with open(os.path.join(root, p, f"{split}.jsonl"), encoding="utf-8") as g:
                for line in g:
                    if line.strip():
                        f.write(line.rstrip("\n") + "\n")
                        n += 1
    print(split, n)
