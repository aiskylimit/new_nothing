"""Prefix alignment (P-ALIGN step 2): the student continues the solution from the truncated prefix.

Port of P-ALIGN/src/prefix-alignment.py: same prompt, chat template with enable_thinking=False,
vLLM sampling n=1, T=0.6, top_p=0.9, repetition_penalty=1.05, up to 32768 new tokens
(max_model_len = max_tokens unless given). Resumable: questions already in --output are skipped.
"""
import argparse
import json
import os
from pathlib import Path

from sgl.palign.truncate import apply_chat

CONTINUE_TEMPLATE = (
    "Please continue from the draft and solve the problem step by step, "
    "and put your final answer within \\boxed{{}}. "
    "I will provide you with some prior knowledge as a draft to assist you in solving the question."
    "*Question*:{question}\n"
    "*Prefix*:{prefix}"
)


def continue_prompt(question: str, prefix: str) -> str:
    return CONTINUE_TEMPLATE.format(question=question, prefix=prefix)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="the student that writes the continuation")
    parser.add_argument("--input", required=True, help="sgl.palign.truncate output")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=None, help="vLLM sampling seed (upstream: none)")
    return parser


def main(argv: list[str] | None = None) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    args = build_parser().parse_args(argv)
    rows = []
    for line in open(args.input, encoding="utf-8"):
        if line.strip():
            item = json.loads(line)
            rows.append({
                "question": item["question"],
                "sufficient_reasoning": item["sufficient_reasoning"],
                "prompt": continue_prompt(item["question"], item["sufficient_reasoning"]),
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        for line in output.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["question"])
            except (json.JSONDecodeError, KeyError):
                continue
    todo = [row for row in rows if row["question"] not in done]
    print(f"{len(rows)} prefixes, {len(done)} already continued, {len(todo)} to go")
    if not todo:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len or args.max_tokens, trust_remote_code=True,
              tensor_parallel_size=1)
    sampling = SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p,
                              repetition_penalty=args.repetition_penalty, max_tokens=args.max_tokens,
                              seed=args.seed)
    with output.open("a", encoding="utf-8") as handle:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            texts = [apply_chat(tokenizer, row["prompt"]) for row in batch]
            for row, result in zip(batch, llm.generate(texts, sampling)):
                handle.write(json.dumps({**row, "output": result.outputs[0].text}, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(f"continued {min(start + args.batch_size, len(todo))}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
