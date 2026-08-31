"""Phase 2: tokenize long-CoT trajectories and segment them into steps.

Output: JSONL per sample with input_ids, the response span, and absolute token spans per
reasoning step -- step text isn't stored, it's recovered by decoding
input_ids[token_start:token_end] (round-trip asserted below).
"""

import argparse
import json
import statistics
import unicodedata
from pathlib import Path

import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from segmentation import encode_with_offsets, step_token_spans

PROMPT_TEMPLATE = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n"
SHUFFLE_BUFFER = 5_000


def build_prompt(tokenizer, problem: str, chat_template: bool, enable_thinking: bool) -> str:
    """Plain-text continuation prompt (base models) or chat-templated prompt (instruct models).

    enable_thinking is only meaningful for models whose template defines it (e.g. Qwen3); a
    tokenizer without that concept just ignores the extra kwarg passed to its jinja template.
    """
    if not chat_template:
        return PROMPT_TEMPLATE.format(problem=problem)
    instruction = PROMPT_TEMPLATE.format(problem=problem)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def build_record(
    tokenizer, problem: str, response: str, max_tokens: int,
    chat_template: bool = False, enable_thinking: bool = True,
) -> dict | None:
    """Tokenize one problem/response pair and map its steps to absolute token spans.

    Returns None when the sample exceeds `max_tokens`, so over-length samples — the most
    expensive ones — are rejected right after tokenizing, before any span mapping.
    """
    # NFC-normalize before storing: some sources (e.g. AceReason) use decomposed forms
    # (U+2261+U+0338 for "≢") that the tokenizer collapses to one codepoint, which would
    # break token-span recovery otherwise.
    problem = unicodedata.normalize("NFC", problem)
    response = unicodedata.normalize("NFC", response)

    prompt = build_prompt(tokenizer, problem, chat_template, enable_thinking)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids, token_starts = encode_with_offsets(tokenizer, response)
    if len(prompt_ids) + len(response_ids) > max_tokens:
        return None

    step_spans = step_token_spans(response, token_starts, offset=len(prompt_ids))
    input_ids = prompt_ids + response_ids
    return {
        "prompt": prompt,
        "response": response,
        "input_ids": input_ids,
        "response_token_span": [len(prompt_ids), len(input_ids)],
        "steps": [{"token_start": start, "token_end": end} for start, end in step_spans],
        "n_tokens": len(input_ids),
    }


def verify_span_alignment(tokenizer, record: dict) -> bool:
    """Decoded step spans must reconstruct the original response text, step by step."""
    decoded = "".join(
        tokenizer.decode(record["input_ids"][step["token_start"] : step["token_end"]])
        for step in record["steps"]
    )
    return decoded.strip() == record["response"].strip()


def iter_samples(config: dict):
    """Stream the source dataset in shuffled order, yielding (problem, response) pairs.

    Shuffling with the configured seed makes any requested subset a random draw from the
    corpus rather than its first rows, while staying deterministic and streaming.
    """
    dataset = load_dataset(config["dataset_name"], split=config["dataset_split"], streaming=True)
    dataset = dataset.shuffle(seed=config["seed"], buffer_size=SHUFFLE_BUFFER)
    wanted = config.get("category_filter")
    for row in dataset:
        category = str(row.get("category") or row.get("source") or row.get("cot_type") or "").lower()
        if wanted and wanted not in category:
            continue
        problem = row.get("problem") or row.get("question") or row.get("input")
        # s1K-1.1 splits the long CoT into a thinking trace + a short final write-up (its
        # own "solution" field is just the bare final answer, e.g. "128" — no steps to select).
        if row.get("deepseek_thinking_trajectory") and row.get("deepseek_attempt"):
            response = (
                f"<think>\n{row['deepseek_thinking_trajectory'].strip()}\n</think>\n\n"
                f"{row['deepseek_attempt'].strip()}"
            )
        elif row.get("generated_response"):
            # VoCuc/s1K-1.1-DeepSeek-R1-Distill-Qwen-32B: already "{reasoning}</think>\n\n{final
            # write-up}", just missing the opening "<think>\n".
            response = f"<think>\n{row['generated_response'].strip()}"
        else:
            response = row.get("solution") or row.get("response") or row.get("output")
        if problem and response:
            yield problem, response


def percentiles(values: list[int]) -> str:
    """Compact distribution summary: p10 / median / p90 / max."""
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]  # noqa: E731
    return f"p10={pick(0.1)} median={pick(0.5)} p90={pick(0.9)} max={ordered[-1]}"


def log_stats(records: list[dict]) -> None:
    """Token-length, steps/sample and tokens/step distributions (phase 2 step 3)."""
    if not records:
        print("no accepted samples; skip token/step statistics")
        return
    tokens = [record["n_tokens"] for record in records]
    steps = [len(record["steps"]) for record in records]
    step_lengths = [
        step["token_end"] - step["token_start"] for record in records for step in record["steps"]
    ]
    print(f"tokens/sample : mean={statistics.mean(tokens):.0f} {percentiles(tokens)}")
    print(f"steps/sample  : mean={statistics.mean(steps):.1f} {percentiles(steps)}")
    print(f"tokens/step   : mean={statistics.mean(step_lengths):.1f} {percentiles(step_lengths)}")


def collect_records(tokenizer, config: dict, limit_scan: int) -> tuple[list[dict], dict]:
    """Scan the shuffled stream until the requested records pass both filters.

    When n_samples is unset, the full source split is used after filtering. The bar tracks
    accepted records; its postfix carries the rejection counts, since a stalled bar with a
    climbing `scanned` means the filters are eating the corpus.
    """
    # Warn before the bar sits at 0: streaming yields nothing until the shuffle buffer fills.
    print(f"buffering {SHUFFLE_BUFFER:,} rows for the shuffled stream (source-bound)...", flush=True)

    records, counts = [], {"scanned": 0, "skipped_long": 0}
    target = config.get("n_samples")
    scan_cap = None if limit_scan <= 0 else limit_scan
    progress = tqdm(total=target, unit="sample", desc="segmenting")
    for problem, response in iter_samples(config):
        counts["scanned"] += 1
        if scan_cap is not None and counts["scanned"] > scan_cap:
            break
        if target is not None and len(records) >= target:
            break

        record = build_record(
            tokenizer, problem, response, config["max_tokens"],
            chat_template=config.get("chat_template", False),
            enable_thinking=config.get("enable_thinking", True),
        )
        if record is None:
            counts["skipped_long"] += 1
        else:
            record["id"] = len(records)
            records.append(record)
            # Every record, not a sample: the whole pass costs ~3s against a ~60s run.
            if not verify_span_alignment(tokenizer, record):
                raise RuntimeError(f"step span round-trip failed on record {record['id']}")
            progress.update(1)
        # Redraw on rejected rows too (throttled), else a long reject streak looks frozen.
        progress.set_postfix(counts, refresh=counts["scanned"] % 50 == 0)

    progress.close()
    return records, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="optional yaml base; CLI flags below override it")
    parser.add_argument("--dataset-name")
    parser.add_argument("--dataset-split")
    parser.add_argument("--category-filter", help="substring match against category/source/cot_type")
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--tokenizer", help="HF tokenizer id, e.g. Qwen/Qwen3-8B")
    parser.add_argument("--output-path")
    parser.add_argument("--chat-template", action=argparse.BooleanOptionalAction)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction)
    parser.add_argument("--limit-scan", type=int, default=0, help="max source rows to scan; 0 means no scan cap")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    overrides = {
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "category_filter": args.category_filter,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "tokenizer": args.tokenizer,
        "output_path": args.output_path,
        "chat_template": args.chat_template,
        "enable_thinking": args.enable_thinking,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    config.setdefault("dataset_split", "train")
    config.setdefault("seed", 42)
    if config.get("n_samples") is not None and config["n_samples"] <= 0:
        raise ValueError("--n-samples must be positive when set; omit it to use the full dataset")

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    records, counts = collect_records(tokenizer, config, args.limit_scan)

    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    print(f"wrote {len(records)} samples -> {output_path}")
    print(" ".join(f"{name}={value}" for name, value in counts.items()))
    log_stats(records)


if __name__ == "__main__":
    main()
