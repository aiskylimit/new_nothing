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

from sgl.segment import SEGMENTERS
from sgl.segment.sentence import encode_with_offsets, step_token_spans

# Must stay byte-identical to benchmarks.MATH_PROMPT so the student sees the same instruction
# at train and eval time; the shape is P-ALIGN's (instruction first, no separator).
PROMPT_TEMPLATE = "Please reason step by step, and put your final answer within \\boxed{{}}.{problem}"
SHUFFLE_BUFFER = 5_000
# What the student is supervised on. "long_cot" is the long thinking trajectory plus the final
# write-up (the P-ALIGN / spectral setting); "answer" drops every model-generated token and trains
# on the source's ground-truth solution only, the answer-only SFT baseline.
RESPONSE_MODES = ("long_cot", "answer")


def build_prompt(
    tokenizer, problem: str, chat_template: bool, enable_thinking: bool, palign_prompt: bool = False,
    instruction_separator: str = "",
) -> str:
    """Plain-text continuation prompt (base models) or chat-templated prompt (instruct models).

    Templates that define enable_thinking (e.g. Qwen3) honour the kwarg; ones that ignore it
    but hard-code an open `<think>` (R1-Distill) are closed by close_open_thinking(), so
    --no-enable-thinking means the same thing for every student.

    `instruction_separator` goes between the instruction and the problem. It is empty for the
    SGL arms (train prompt == eval prompt); P-ALIGN's LLaMA-Factory training joins its alpaca
    `instruction` and `input` with "\n" (while its evaluation uses no separator).
    """
    instruction = PROMPT_TEMPLATE.format(problem=instruction_separator + problem)
    if not chat_template:
        return instruction
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if enable_thinking:
        return prompt
    return palign_close_thinking(prompt) if palign_prompt else close_open_thinking(prompt)


def close_open_thinking(prompt: str) -> str:
    """Close a thinking block the template opened but the run asked not to use.

    R1-Distill's template hard-codes a trailing `<think>` and ignores `enable_thinking` (only
    Qwen3-style templates define it), so --no-enable-thinking would otherwise still hand the
    model an open thinking block. Closing it at once renders the prompt the way the non-thinking
    templates already render it (`... </think>\n\n`), which is also what answer_only_response()
    is wrapped for. A prompt with no open block is returned unchanged.
    """
    return f"{prompt}</think>\n\n" if prompt.rstrip().endswith("<think>") else prompt


PALIGN_EMPTY_THINK = "<think>\n\n</think>\n\n"


def palign_close_thinking(prompt: str) -> str:
    """Close an open `<think>` the way P-ALIGN/src/test.py apply_chat() does.

    R1-Distill's template ends in `<think>\\n`; P-ALIGN strips it and appends
    `<think>\\n\\n</think>\\n\\n` (two newlines inside the block, unlike close_open_thinking's
    one), which is also what Qwen3 renders for enable_thinking=False.
    """
    stripped = prompt.rstrip()
    if stripped.endswith("<think>"):
        return stripped[: -len("<think>")] + PALIGN_EMPTY_THINK
    return prompt


def reconcile_thinking_markers(prompt: str, response: str) -> str:
    """Never let the response open a thinking block the prompt did not open.

    s1K-1.1 responses carry their own `<think>` wrapper, but templates differ in what they
    leave open: R1-Distill opens `<think>` (so the response must only close it), while Qwen3
    with enable_thinking=False emits a closed empty block and Qwen2.5-Instruct emits nothing
    (so the response reasons in plain prose, matching P-ALIGN Table 8). Keyed on the rendered
    prompt rather than the model name, and applied to every track, so the supervision format
    cannot silently differ between two students the comparison treats as equivalent.
    """
    body = response.lstrip()
    if prompt.rstrip().endswith("<think>"):
        return body[len("<think>") :].lstrip("\n") if body.startswith("<think>") else response
    return body.replace("<think>", "", 1).replace("</think>", "", 1).lstrip("\n")


def build_record(
    tokenizer, problem: str, response: str, max_tokens: int,
    chat_template: bool = False, enable_thinking: bool = True, palign_prompt: bool = False,
    segmenter: str = "sentence", instruction_separator: str = "", stop_suffix: str = "",
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

    prompt = build_prompt(tokenizer, problem, chat_template, enable_thinking, palign_prompt, instruction_separator)
    response = reconcile_thinking_markers(prompt, response)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids, token_starts = encode_with_offsets(tokenizer, response)
    if len(prompt_ids) + len(response_ids) > max_tokens:
        return None

    step_spans = step_token_spans(response, token_starts, offset=len(prompt_ids), segmenter=segmenter)
    # The stop token has to be part of the target, or the model is only ever supervised to
    # continue past its final answer and never to end the turn. It sits outside
    # response_token_span (which stays text-only so step spans still round-trip) and outside
    # every step span, so sgl.selection.build_masks supervises it separately in both arms.
    stop_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    # --stop-suffix: text the trainer being reproduced also supervises after the stop token
    # (LLaMA-Factory's qwen/qwen3 templates end every assistant turn with "<|im_end|>\n").
    if stop_suffix:
        stop_ids += tokenizer(stop_suffix, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + response_ids + stop_ids
    return {
        "prompt": prompt,
        "response": response,
        "input_ids": input_ids,
        "response_token_span": [len(prompt_ids), len(prompt_ids) + len(response_ids)],
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


# A ground-truth `solution` this short with no line break is a bare answer ("128", "\\frac{1}{2}"),
# not a worked solution, and gets boxed so the target still ends in the \\boxed{} the grader reads.
_BARE_ANSWER_MAX_CHARS = 64


def answer_only_response(row: dict) -> str | None:
    """The row's ground-truth solution as the target, with no model-generated reasoning.

    s1K-1.1 carries the source dataset's own `solution` (a full reference solution for some
    sources, just the final answer for others); LIMO-style rows carry a bare `answer`. Neither
    the DeepSeek/Gemini trajectory nor their attempt is used. A target with no `\\boxed{}` gets
    one appended (from `answer` when present, else from the solution itself when it is bare), so
    the model is still trained to end with the boxed answer the grader extracts.

    Wrapped in an empty thinking block so reconcile_thinking_markers renders it the same way for
    every template: an open `<think>` prompt gets the block closed at once, a non-thinking prompt
    gets the bare solution. Returns None when the row has no ground truth.
    """
    solution = str(row.get("solution") or "").strip()
    answer = str(row.get("answer") if row.get("answer") not in (None, "") else "").strip()
    if not solution and not answer:
        return None
    if "\\boxed" in solution:
        target = solution
    elif not solution or ("\n" not in solution and len(solution) <= _BARE_ANSWER_MAX_CHARS):
        target = f"The final answer is \\boxed{{{answer or solution}}}."
    elif answer:
        target = f"{solution}\n\nThe final answer is \\boxed{{{answer}}}."
    else:
        target = solution
    return f"<think>\n\n</think>\n\n{target}"


def iter_samples(config: dict):
    """Stream the source dataset in shuffled order, yielding (problem, response) pairs.

    Shuffling with the configured seed makes any requested subset a random draw from the
    corpus rather than its first rows, while staying deterministic and streaming.
    """
    name = config["dataset_name"]
    if Path(name).is_file():  # local json/jsonl(.gz) file, e.g. P-ALIGN's palign_sft_*.json.gz
        dataset = load_dataset("json", data_files=name, split="train", streaming=True)
    else:
        dataset = load_dataset(name, split=config["dataset_split"], streaming=True)
    dataset = dataset.shuffle(seed=config["seed"], buffer_size=SHUFFLE_BUFFER)
    wanted = config.get("category_filter")
    answer_only = config.get("response_mode", "long_cot") == "answer"
    for row in dataset:
        category = str(row.get("category") or row.get("source") or row.get("cot_type") or "").lower()
        if wanted and wanted not in category:
            continue
        problem = row.get("problem") or row.get("question") or row.get("input")
        if answer_only:
            response = answer_only_response(row)
            if problem and response:
                yield problem, response
            continue
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
        elif row.get("instruction") is not None and row.get("output"):
            # P-ALIGN alpaca rows (instruction = the "Please reason..." line, input = problem):
            # `output` is the prefix-aligned trace, trained verbatim as P-ALIGN does -- plain prose
            # with its <Begin_of_Prefix>/<End_of_Prefix> markers and final \boxed{}, no <think>.
            response = row["output"].strip()
        else:
            # LIMO-style sources: bare reasoning `solution` + short final `answer`, no <think>
            # markup. For a thinking student the response closes </think> and states the \boxed
            # answer outside it; reconcile_thinking_markers handles the opener.
            solution = row.get("solution") or row.get("response") or row.get("output")
            answer = row.get("answer")
            if solution:
                ans = str(answer).strip() if answer not in (None, "") else ""
                tail = f"\n\nThe final answer is \\boxed{{{ans}}}." if ans else ""
                response = f"{solution.strip()}\n</think>{tail}"
            else:
                response = solution
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
            palign_prompt=config.get("palign_prompt", False),
            segmenter=config.get("segmenter", "sentence"),
            instruction_separator=config.get("instruction_separator", ""),
            stop_suffix=config.get("stop_suffix", ""),
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


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--palign-prompt", action=argparse.BooleanOptionalAction,
        help="thinking off: close R1's forced <think> as P-ALIGN / sgl.eval.evaluate --palign-prompt do "
        "(<think>\\n\\n</think>\\n\\n) so train and eval prompts tokenize identically",
    )
    parser.add_argument(
        "--response-mode", choices=RESPONSE_MODES,
        help="long_cot = thinking trajectory + final write-up (default); answer = ground-truth solution only",
    )
    parser.add_argument(
        "--segmenter", choices=sorted(SEGMENTERS),
        help="step segmentation rule (default sentence; paragraph/cue are Segment-Selective SFT's)",
    )
    parser.add_argument(
        "--instruction-separator",
        help="text between the instruction and the problem (default none; P-ALIGN's LLaMA-Factory "
        "training uses a newline)",
    )
    parser.add_argument(
        "--stop-suffix",
        help="extra supervised text after the stop token (LLaMA-Factory's qwen/qwen3 templates: a newline)",
    )
    parser.add_argument("--limit-scan", type=int, default=0, help="max source rows to scan; 0 means no scan cap")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

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
        "palign_prompt": args.palign_prompt,
        "response_mode": args.response_mode,
        "segmenter": args.segmenter,
        "instruction_separator": args.instruction_separator,
        "stop_suffix": args.stop_suffix,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    config.setdefault("dataset_split", "train")
    config.setdefault("seed", 42)
    config.setdefault("response_mode", "long_cot")
    if config["response_mode"] not in RESPONSE_MODES:
        raise ValueError(f"response_mode must be one of {RESPONSE_MODES}, got {config['response_mode']!r}")
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
