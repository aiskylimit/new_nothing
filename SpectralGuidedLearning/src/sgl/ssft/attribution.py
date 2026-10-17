"""Integrated Gradients from every reasoning token to the final answer (Segment-Selective SFT).

Port of SegmentSelectiveSFT/Attribution/grad_analyze.py (the fork's batched, memory-lean version):

    input  = chat(user: "<question>\\nPlease reason step by step, ...") + "".join(segments)
             + "</think> So, the final answer is \\boxed{<answer>}"
    target = sum of log p(answer tokens), the tokens strictly inside the \\boxed{...} braces
    IG     = (x - x') * mean_j grad_x target(x' + a_j (x - x')),  a_j = linspace(0, 1, steps)
             on input embeddings, x' = embeddings of an all-pad sequence, summed over the hidden
             dimension and L2-normalised over the sequence.

Weights are frozen (only d/d inputs_embeds is needed), lm_head runs only on the answer positions,
and interpolation points are processed `--ig-batch-size` at a time. Samples that OOM or exceed
--max-input-tokens get all-zero scores (Segment-Selective SFT then keeps only its default
segments for them). The interpolation runs in the model's dtype (bf16 for the released runs, as
upstream hard-codes).

Outputs, row-aligned with the input:
    --output          {"segments": [[n_tokens, sum|IG|, sum IG], ...]} per sample (what `select` reads)
    --output-full     optional: per-token scores per segment (upstream's IG.jsonl format)
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

USER_TEMPLATE = "{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
ANSWER_TEMPLATE = "</think> So, the final answer is \\boxed{{{answer}}}"

OOM_ERROR = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError


def segment_token_spans(tokenizer, segments: list[str], offset: int) -> tuple[list[int], list[tuple[int, int]]]:
    """Token ids of "".join(segments) and each segment's absolute [start, end) span.

    A token belongs to the segment holding its first character; a segment with no token gets
    (0, 0). Spans come from one offset mapping of the joined text, never from re-tokenizing
    prefixes (BPE merges differ at boundaries).
    """
    joined = "".join(segments)
    encoding = tokenizer(joined, add_special_tokens=False, return_offsets_mapping=True)
    char_starts, cursor = [], 0
    for segment in segments:
        char_starts.append(cursor)
        cursor += len(segment)

    first: dict[int, int] = {}
    last: dict[int, int] = {}
    index = 0
    for position, (start, end) in enumerate(encoding["offset_mapping"]):
        if end <= start:
            continue
        while index + 1 < len(char_starts) and start >= char_starts[index + 1]:
            index += 1
        first.setdefault(index, position)
        last[index] = position
    spans = [
        (first[k] + offset, last[k] + 1 + offset) if k in first else (0, 0)
        for k in range(len(segments))
    ]
    return encoding["input_ids"], spans


def answer_token_range(tokenizer, answer: str) -> tuple[list[int], int, int]:
    """Ids of the answer string and the [start, end) of the tokens inside \\boxed{...}.

    Upstream's token scan, kept as is: start after the last token containing "boxed" (plus the
    token holding "{"), end at the token that closes the brace depth.
    """
    ids = tokenizer(ANSWER_TEMPLATE.format(answer=answer), add_special_tokens=False)["input_ids"]
    pieces = tokenizer.convert_ids_to_tokens(ids)
    boxed_at = None
    for position, piece in enumerate(pieces):
        if "boxed" in piece:
            boxed_at = position + 1
    if boxed_at is None or "{" not in pieces[boxed_at]:
        raise ValueError(f"cannot locate \\boxed{{ in the answer tokens: {pieces}")
    end, depth, done = len(pieces), 0, False
    for position in range(boxed_at, len(pieces)):
        piece = pieces[position]
        if "{" in piece or "}" in piece:
            for char in piece:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end, done = position, True
                        break
            if done:
                break
    return ids, boxed_at + 1, end


def build_example(tokenizer, row: dict) -> tuple[list[int], list[tuple[int, int]], tuple[int, int]]:
    """(input ids, absolute segment spans, absolute answer [start, end)) for one row."""
    user_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": USER_TEMPLATE.format(input=row["question"])}],
        tokenize=True, add_generation_prompt=True,
    )
    if hasattr(user_ids, "keys"):  # transformers 5 returns a BatchEncoding
        user_ids = user_ids["input_ids"]
    user_ids = list(user_ids)
    response_ids, spans = segment_token_spans(tokenizer, row["segments"], len(user_ids))
    answer_ids, answer_start, answer_end = answer_token_range(tokenizer, row["answer"])
    prefix = len(user_ids) + len(response_ids)
    return user_ids + response_ids + answer_ids, spans, (answer_start + prefix, answer_end + prefix)


class IntegratedGradients:
    def __init__(self, model, steps: int = 20, batch_size: int = 1):
        self.model = model
        self.steps = steps
        self.batch_size = batch_size
        self.device = next(model.parameters()).device
        self.dtype = model.get_input_embeddings().weight.dtype

    def __call__(self, input_ids: list[int], spans: list[tuple[int, int]], answer: tuple[int, int],
                 baseline_token_id: int) -> list[list[float]]:
        model, steps = self.model, self.steps
        ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device).unsqueeze(0)
        embed = model.get_input_embeddings()
        with torch.no_grad():
            inputs = embed(ids)
            baseline = embed(torch.full_like(ids, baseline_token_id))
        alphas = torch.linspace(0, 1, steps).view(steps, 1, 1).to(self.dtype).to(self.device)
        total = torch.zeros_like(inputs)
        start, end = answer
        targets = ids[0, start:end]

        position = 0
        while position < steps:
            stop = min(position + self.batch_size, steps)
            chunk = stop - position
            alpha = alphas[position:stop]
            start_point = baseline.expand(chunk, -1, -1)
            path = start_point + alpha * (inputs.expand(chunk, -1, -1) - start_point).detach()
            path = path.to(dtype=self.dtype)
            path.requires_grad_(True)
            model.zero_grad(set_to_none=True)
            hidden = model.model(inputs_embeds=path).last_hidden_state
            logits = model.lm_head(hidden[:, start - 1:end - 1, :])
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            # lm_head may sit on another GPU than the embeddings when the model is sharded
            index = targets.to(log_probs.device).unsqueeze(0).expand(chunk, -1).unsqueeze(-1)
            picked = log_probs.gather(-1, index).squeeze(-1)
            gradient = torch.autograd.grad(picked.sum(), path, retain_graph=False, create_graph=False)[0]
            total += gradient.sum(dim=0, keepdim=True)
            position = stop

        attributions = ((inputs - baseline) * (total / steps)).sum(dim=-1).squeeze(0)
        attributions = attributions / attributions.norm()
        scores = [attributions[a:b].detach().cpu().float().numpy().tolist() for a, b in spans]
        del inputs, baseline, total
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return scores


def compact(scores: list[list[float]]) -> list[list[float]]:
    """Per segment [n_tokens, sum|IG|, sum IG], rounded as upstream's *_compact.jsonl."""
    rows = []
    for segment in scores:
        n = len(segment)
        sum_abs = float(np.sum(np.abs(segment))) if n else 0.0
        sum_signed = float(np.sum(segment)) if n else 0.0
        rows.append([n, round(sum_abs, 8), round(sum_signed, 8)])
    return rows


def load_model(name: str, dtype: str, device_map: str | None, gradient_checkpointing: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if not tokenizer.is_fast:
        raise SystemExit(f"{name}: a fast tokenizer (offset mapping) is required to locate segments")
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=getattr(torch, dtype), device_map=device_map, trust_remote_code=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if gradient_checkpointing:
        dropout = getattr(model.config, "attention_dropout", 0.0) or 0.0
        if dropout > 0 or any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in model.modules()):
            raise SystemExit("model has dropout > 0: gradient checkpointing needs train() mode, which "
                             "would make IG noisy; rerun with --no-gradient-checkpointing")
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()  # HF only checkpoints in training mode; no dropout, so results are unchanged
    return model, tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="attribution model (paper: DeepSeek-R1-Distill-Qwen-7B)")
    parser.add_argument("--data-path", required=True, help="rows with question / answer / segments (sgl.data.s1k)")
    parser.add_argument("--output", required=True, help="compact IG jsonl, row-aligned with --data-path")
    parser.add_argument("--output-full", help="also write per-token IG (large)")
    parser.add_argument("--ig-steps", type=int, default=20, help="interpolation points J (paper 50; 20 is ~5%% off)")
    parser.add_argument("--ig-batch-size", type=int, default=4, help="interpolation points per forward pass")
    parser.add_argument("--max-input-tokens", type=int, default=0, help="longer samples get zero scores; 0 = no cap")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto", help="'none' to load on CPU")
    parser.add_argument("--resume", action="store_true", help="continue a partial --output instead of overwriting it")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = [json.loads(line) for line in open(args.data_path) if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]
    model, tokenizer = load_model(args.model, args.dtype, None if args.device_map == "none" else args.device_map,
                                  args.gradient_checkpointing)
    ig = IntegratedGradients(model, args.ig_steps, args.ig_batch_size)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    if args.resume and output.exists():
        done = sum(1 for line in output.open() if line.strip())
        print(f"resuming after {done}/{len(rows)} rows")
    mode = "a" if done else "w"
    full = open(args.output_full, mode) if args.output_full else None
    skipped = {"oom": 0, "too_long": 0}
    with output.open(mode) as handle:
        for index in range(done, len(rows)):
            input_ids, spans, answer = build_example(tokenizer, rows[index])
            zeros = [[0.0] * max(0, b - a) for a, b in spans]
            if args.max_input_tokens and len(input_ids) > args.max_input_tokens:
                scores, skipped["too_long"] = zeros, skipped["too_long"] + 1
            else:
                try:
                    scores = ig(input_ids, spans, answer, tokenizer.pad_token_id)
                except OOM_ERROR:
                    model.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    scores, skipped["oom"] = zeros, skipped["oom"] + 1
            handle.write(json.dumps({"segments": compact(scores)}) + "\n")
            handle.flush()
            if full:
                full.write(json.dumps(scores) + "\n")
            print(f"[{index + 1}/{len(rows)}] {len(input_ids)} tokens, {len(spans)} segments", flush=True)
    if full:
        full.close()
    print(f"wrote {output}; zero-scored samples: {skipped}")


if __name__ == "__main__":
    main()
