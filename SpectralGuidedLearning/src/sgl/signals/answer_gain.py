"""Per-step answer-information gain: how much each reasoning step raises the frozen student's
log-probability of the gold final answer.

    gain_k = log p(answer | prompt, steps_1..k) - log p(answer | prompt, steps_1..k-1)

p(answer | context) is read by appending a fixed probe ("... the final answer is \\boxed{") and
scoring the gold answer tokens + "}" with vLLM prompt_logprobs. All probes of a sample share the
response prefix, so vLLM's prefix cache makes this ~one forward pass per sample. The gold answer
is the last \\boxed{...} of the training response (P-ALIGN keeps only answer-verified samples).
Output: {id: [gain per step]} JSON, aligned with train-segmented.jsonl steps.
"""

import argparse
import json
from pathlib import Path

from sgl.eval.graders.answer_scoring import extract_boxed

PROBE = "\n\nTherefore, the final answer is \\boxed{"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="train-segmented.jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records = [json.loads(line) for line in open(args.data_path)][: args.limit]
    probe_ids = tokenizer(PROBE, add_special_tokens=False)["input_ids"]

    prompts, index = [], []  # index[i] = (record id, step k (0 = before any step), n answer tokens)
    skipped = 0
    for record in records:
        answer = extract_boxed(record["response"])
        if not answer:
            skipped += 1
            continue
        answer_ids = tokenizer(answer + "}", add_special_tokens=False)["input_ids"]
        start = record["response_token_span"][0]
        cuts = [start] + [step["token_end"] for step in record["steps"]]
        for k, cut in enumerate(cuts):
            ids = record["input_ids"][:cut] + probe_ids + answer_ids
            if len(ids) >= args.max_model_len:
                continue
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            index.append((record["id"], k, len(answer_ids)))
    print(f"{len(prompts):,} probes over {len(records) - skipped} samples (skipped {skipped} without \\boxed)")

    llm = LLM(
        model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, enable_prefix_caching=True, enforce_eager=True,
    )
    outputs = llm.generate(prompts, SamplingParams(max_tokens=1, prompt_logprobs=0))

    logp = {}  # (id, k) -> log p(answer | prefix up to step k)
    for (rid, k, n_answer), out, prompt in zip(index, outputs, prompts):
        ids = prompt["prompt_token_ids"]
        total = 0.0
        for position in range(len(ids) - n_answer, len(ids)):
            total += out.prompt_logprobs[position][ids[position]].logprob
        logp[(rid, k)] = total

    gains = {}
    for record in records:
        rid, n_steps = record["id"], len(record["steps"])
        if (rid, 0) not in logp:
            continue
        values = [logp.get((rid, k)) for k in range(n_steps + 1)]
        # a step whose probe was skipped (too long) inherits the last known value: zero gain
        for k in range(1, len(values)):
            if values[k] is None:
                values[k] = values[k - 1]
        gains[rid] = [values[k] - values[k - 1] for k in range(1, n_steps + 1)]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(gains))
    print(f"wrote gains for {len(gains)} samples -> {args.output}")


if __name__ == "__main__":
    main()
