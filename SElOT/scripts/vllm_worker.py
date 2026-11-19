#!/usr/bin/env python3
"""Greedy generation with vLLM for text-only steps (no activation hook is needed there): the label stage
(fit_data.engine=vllm) and the Llama Guard / HarmBench judges (judge.engine=vllm). Called through
selot.vllm_engine.vllm_generate.

    python scripts/vllm_worker.py --model DIR --inp in.jsonl --out out.jsonl --max-tokens N [--gpu-mem 0.85]

in.jsonl rows: {"id", "text"} where text already has the chat template applied (as selot.activations.format_prompts
or LlamaGuardJudge.chat build it; tokenized here with add_special_tokens=False, so no second BOS), or {"id", "ids"}:
token ids built by the caller with the judge's own HF tokenizer (HarmBench: slow tokenizer, BOS added). Either way the
model sees exactly the tokens the HF path would. out.jsonl rows: {"id", "output"}.
Run as its own process: when it exits, the GPU memory is fully released.
Greedy vLLM and greedy HF can differ in a few tokens (other kernels); that is fine for labels and judge verdicts (each
label records its engine), but steered and unsteered eval responses are both generated with HF (scripts/generate.py)
so they share one engine.
"""

import argparse
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    rows = [json.loads(l) for l in open(a.inp) if l.strip()]
    tok = AutoTokenizer.from_pretrained(a.model) if any("ids" not in r for r in rows) else None
    ids = [r["ids"] if "ids" in r else tok(r["text"], add_special_tokens=False)["input_ids"] for r in rows]
    longest = max(len(x) for x in ids) + a.max_tokens
    llm = LLM(model=a.model, dtype=a.dtype, gpu_memory_utilization=a.gpu_mem, max_model_len=longest,
              enable_prefix_caching=True, seed=0)
    params = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, skip_special_tokens=True)
    outs = llm.generate([TokensPrompt(prompt_token_ids=x) for x in ids], params)
    with open(a.out, "w") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({"id": r["id"], "output": o.outputs[0].text}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
