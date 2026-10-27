"""Rescore finished RE runs with the current ed_eval.re_evaluate_by_dataset (adds triple_gold_pair) from the
answers they saved, and append the RESULT lines to each run's log.txt.

    python tools/rescore_re.py results/qwen3/general_domain/sft_4B_re_tacred/e3-bs8-lr0.0001-G4-N1-NN1-lora-32-64-0.1 ...

A run dir holds args.json and eval/<epoch>/{dev,test}_answers.jsonl. Answer i belongs to row i of
<data_dir>/{valid,test}.jsonl after the run's --datasets filter, as in LMTrainDataset."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ed_eval import format_by_dataset, re_evaluate_by_dataset


def rescore(run_dir):
    with open(os.path.join(run_dir, "args.json")) as f:
        args = json.load(f)
    wanted = set(args["datasets"].split(",")) if args.get("datasets") else None
    lines = []
    eval_dir = os.path.join(run_dir, "eval")
    for epoch in sorted(os.listdir(eval_dir), key=int):
        for split, data_split in (("dev", "valid"), ("test", "test")):
            path = os.path.join(eval_dir, epoch, f"{split}_answers.jsonl")
            if not os.path.exists(path):
                continue      # --eval-gen-mode best: test only at epochs where dev improved
            with open(path) as f:
                responses = [json.loads(line)["text"] for line in f]
            with open(os.path.join(args["data_dir"], f"{data_split}.jsonl")) as f:
                rows = [json.loads(line) for line in f]
            rows = [r for r in rows if wanted is None or r["dataset"] in wanted][:len(responses)]
            lines.append(format_by_dataset(re_evaluate_by_dataset(responses, rows), split, epoch))
    text = "RESCORED with triple_gold_pair (tools/rescore_re.py)\n" + "\n".join(lines)
    print(f"== {run_dir}\n{text}")
    with open(os.path.join(run_dir, "log.txt"), "a") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    for run_dir in sys.argv[1:]:
        rescore(run_dir)
