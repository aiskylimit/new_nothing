"""#0 — Adaptive Prefix Truncation (P-ALIGN).

Cắt CoT teacher xuống prefix NGẮN NHẤT mà student (base Qwen) vẫn ra đúng đáp án.
Student không tiêu hoá nổi CoT 10k-token; đuôi là nước đi riêng của teacher -> nhiễu.

Grid frac thay vì binary search (1 pass vLLM, dễ song song):
  với mỗi mẫu, thử k = ceil(frac * n_steps), frac in FRACS.
  prompt = <chat question> + "<think>\n" + steps[:k] + "\n</think>\n\n"
  student sinh N completion -> grade vs gold. pass nếu >= ceil(N/2) đúng.
  k* = frac nhỏ nhất pass; không cái nào -> giữ full (sufficient=False).

    python -m proposal_v2.prefix_truncate --gpu 0 --gpu_mem 0.33 \
        --in data/train_s1k.jsonl --out data/train_s1k_prefixed.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os

FRACS = [0.15, 0.30, 0.45, 0.60, 0.80, 1.00]


def build_prefix_prompt(tokenizer, question: str, steps, k: int) -> str:
    from core.segmentation import render_prompt
    pre = render_prompt(tokenizer, question)
    think_open = "" if pre.rstrip().endswith("<think>") else "<think>\n"
    body = "\n\n".join(steps[:k])
    return f"{pre}{think_open}{body}\n</think>\n\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/train_s1k.jsonl")
    ap.add_argument("--out", default="data/train_s1k_prefixed.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--gpu_mem", type=float, default=0.33)
    ap.add_argument("--n", type=int, default=4, help="sample/(mẫu,frac)")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from eval.grade import grade_one

    rows = [json.loads(l) for l in open(args.inp)]
    if args.limit:
        rows = rows[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.model)

    # (row_idx, frac_idx, k)
    prompt_cap = args.max_model_len - args.max_tokens - 64
    jobs = []
    prompts = []
    n_skip_long = 0
    for ri, r in enumerate(rows):
        steps = r["steps"]
        ns = len(steps)
        seen = set()
        for fi, frac in enumerate(FRACS):
            k = max(1, min(ns, math.ceil(frac * ns)))
            if k in seen:
                continue
            seen.add(k)
            p = build_prefix_prompt(tok, r["question"], steps, k)
            if len(tok(p, add_special_tokens=False)["input_ids"]) > prompt_cap:
                n_skip_long += 1
                continue
            jobs.append((ri, fi, k))
            prompts.append(p)

    print(f"[prefix] {len(rows)} mẫu -> {len(prompts)} prompt (grid {FRACS}), "
          f"bỏ {n_skip_long} prompt quá dài (>{prompt_cap} tok)", flush=True)

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, enable_prefix_caching=True, seed=0)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_tokens, stop=["<|im_end|>", "<|endoftext|>"])
    outs = llm.generate(prompts, sp)

    # gom kết quả theo row
    per_row = {ri: {} for ri in range(len(rows))}
    need = math.ceil(args.n / 2)
    for (ri, fi, k), o in zip(jobs, outs):
        gold = str(rows[ri].get("answer", ""))
        n_ok = sum(1 for c in o.outputs if grade_one(c.text, gold))
        per_row[ri][k] = (n_ok, fi)

    n_written = n_suff = 0
    tot_frac = 0.0
    with open(args.out, "w") as w:
        for ri, r in enumerate(rows):
            ns = len(r["steps"])
            passing = sorted([k for k, (n_ok, _) in per_row[ri].items() if n_ok >= need])
            if passing:
                kstar = passing[0]
                suff = True
                n_suff += 1
            else:
                kstar = ns
                suff = False
            frac = kstar / max(ns, 1)
            tot_frac += frac
            new_steps = r["steps"][:kstar]
            out = dict(r)
            out["steps"] = new_steps
            out["thinking"] = "\n\n".join(new_steps)
            out["orig_n_steps"] = ns
            out["prefix_n_steps"] = kstar
            out["prefix_frac"] = round(frac, 3)
            out["prefix_sufficient"] = suff
            w.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[prefix] xong -> {args.out}", flush=True)
    print(f"[prefix] {n_written} mẫu | sufficient {n_suff}/{n_written} "
          f"({100*n_suff/n_written:.0f}%) | frac TB {tot_frac/n_written:.2f}", flush=True)


if __name__ == "__main__":
    main()
