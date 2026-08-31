"""Phase 4 (Pru-CoT baseline): LLM-guided pruning with fidelity constraints (paper Section 3.4,
adapted from Pru-CoT/LLM_prune_threshold.py).

Steps below the importance threshold from prucot_weight.py (Eq. 3) become pruning candidates;
a scale-matched LLM pruning agent then decides, per candidate, whether removing it would break
the reasoning chain's logical coherence. Unlike spectral (masks the loss but keeps every token
in context), Pru-CoT genuinely deletes pruned steps' text and re-tokenizes the shortened
response, but still emits the same {id, input_ids, loss_mask} jsonl shape as build_masks.py, so
train_sft.py/masked_loss.py/data_collator.py need no changes.

--config points at a yaml with the same fields as a fallback/override base; CLI flags always
win when both are given.
"""

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd
import yaml
from transformers import AutoTokenizer

from build_masks import summarize
from segmentation import record_step_spans
from step_selection import selection_stats

try:
    import json_repair
except ImportError:  # pragma: no cover - exercised only when the dependency is missing
    json_repair = None

# Ported verbatim from Pru-CoT/LLM_prune_threshold.py (paper Appendix A).
SYSTEM_PROMPT = """You are an **aggressive** CoT pruner. Your goal is to **remove all non-essential steps**.

**Core Principle:**
1.  **Default: `"prune": True`.** Prune all redundant steps, clarifications, or checks.
2.  **Exception: `"prune": False`.** ONLY keep steps that are **critically essential**. A step is essential *only if* its removal breaks the logical chain or makes the solution impossible.
3.  **Rule: If in doubt, prune.**

**Task & Format Rules:**
1.  You will see the thinking process in a table.
2.  **ONLY focus on steps that have a Number ID** in the first column. Steps with an empty ID are protected context.
3.  Respond with *only* a JSON object.
4.  **Keys:** Must be the Number ID from the table (e.g., "1", "2").
5.  **Values:** Must be `{"reasoning": (string), "prune": (boolean)}`.

**Example Response:**
{
  "1": {
    "reasoning": "Prune. Step 1 is a redundant clarification.",
    "prune": True
  },
  "2": {
    "reasoning": "Keep. Step 2 defines variable x, critical for the solution.",
    "prune": False
  }
}
"""

USER_PROMPT_TEMPLATE = """
Here is the problem:
{user_input}

Here is the final solution:
{solution}

--------------------------------------------------
**YOUR TASK:**

Below is the **full thinking process**.
Analyze **ONLY** the steps with a **Number ID** in the first column.

| ID | Thinking Step |
|---|---|
{markdown_table_rows}

For each step with an ID, decide if it can be pruned (`True`) or if it is logically essential (`False`).

Respond *only* with the JSON dictionary.
"""


def median_gate(weights: list[float], threshold: float) -> bool:
    """True => skip LLM pruning entirely, keep every step (paper's median-gate check)."""
    return bool(weights) and statistics.median(weights) > threshold


def candidate_step_indices(weights: list[float], threshold: float) -> list[int]:
    """Steps eligible for LLM-guided pruning: weight below tau (Eq. 3), excluding the first and
    last step of the scored range as structural anchors (first often holds the opening "<think>"
    tag; last is a uniform safety margin regardless of whether a "</think>" split was found)."""
    if len(weights) < 3:
        return []
    last = len(weights) - 1
    return [index for index, weight in enumerate(weights) if weight < threshold and index not in (0, last)]


def build_virtual_id_map(candidate_indices: list[int]) -> dict[int, int]:
    """Real step index -> 1-based display id, assigned only to candidates (paper: non-candidate
    rows show a blank id so the agent cannot hallucinate pruning a protected step)."""
    return {real: position + 1 for position, real in enumerate(sorted(candidate_indices))}


def format_step_table(step_texts: list[str], virtual_ids: dict[int, int]) -> str:
    rows = []
    for index, text in enumerate(step_texts):
        content = text.replace("\n", " ").replace("|", "\\|")
        display_id = str(virtual_ids[index]) if index in virtual_ids else ""
        rows.append(f"| {display_id} | {content} |")
    return "\n".join(rows)


def parse_prune_response(text: str) -> tuple[list[int] | None, str]:
    """Virtual step ids marked prune=true in the agent's JSON reply."""
    if json_repair is None:
        return None, "parse_failed: json_repair not installed"
    try:
        parsed = json_repair.loads(text)
    except Exception as error:  # noqa: BLE001 - any malformed LLM output lands here
        return None, f"parse_failed: {error}"
    if not isinstance(parsed, dict):
        return None, "parse_failed: not a JSON object"

    pruned = []
    for key, value in parsed.items():
        try:
            virtual_id = int(key)
        except (TypeError, ValueError):
            continue
        should_prune = value.get("prune", False) if isinstance(value, dict) else bool(value)
        if should_prune:
            pruned.append(virtual_id)
    return sorted(set(pruned)), "ok"


def missing_weight_ids(records: list[dict], weights_by_id: dict[int, list[float]]) -> list[int]:
    """Record ids that would be silently dropped if Pru-CoT weights are incomplete."""
    return [int(record["id"]) for record in records if int(record["id"]) not in weights_by_id]


def reconstruct_pruned_text(step_texts: list[str], prune_indices: set[int]) -> str:
    """Concatenating the kept steps reproduces the exact original substring (minus the pruned
    ones): split_into_step_texts (segmentation.py) guarantees the pieces concatenate losslessly."""
    return "".join(text for index, text in enumerate(step_texts) if index not in prune_indices)


def decode_step_texts(tokenizer, record: dict) -> list[str]:
    """Step text isn't stored (data_prep.py) -- recover it by decoding the same span used to
    build it, exactly like data_prep.py's own verify_span_alignment check."""
    return [tokenizer.decode(record["input_ids"][start:end]) for start, end in record_step_spans(record)]


def emit_prucot_record(
    record: dict,
    prune_indices: set[int],
    n_scored: int = 0,
    student_tokenizer=None,
    step_texts: list[str] | None = None,
) -> dict:
    """{id, input_ids, loss_mask}, same shape as build_masks.py's output (so train_sft.py needs
    no changes). `prune_indices` index into step_texts[:n_scored]; the solution tail is never
    touched."""
    prompt_len = record["response_token_span"][0]
    if prune_indices:
        kept_thinking = reconstruct_pruned_text(step_texts[:n_scored], prune_indices)
        solution_tail = "".join(step_texts[n_scored:])
        response_ids = student_tokenizer(kept_thinking + solution_tail, add_special_tokens=False)["input_ids"]
        input_ids = record["input_ids"][:prompt_len] + response_ids
    else:
        input_ids = record["input_ids"]
    response_len = len(input_ids) - prompt_len
    return {"id": record["id"], "input_ids": input_ids, "loss_mask": [0] * prompt_len + [1] * response_len}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="optional yaml base; CLI flags below override it")
    parser.add_argument("--data-path", help="segmented jsonl from data_prep.py (Phase 2 output)")
    parser.add_argument("--weights-path", help="step-weights parquet from prucot_weight.py")
    parser.add_argument("--tokenizer", help="student tokenizer -- must match data_prep.py's --tokenizer")
    parser.add_argument("--pruning-agent", help="HF model id/path for the vLLM LLM-guided pruning agent")
    parser.add_argument("--output-path", help="pruned {id, input_ids, loss_mask} jsonl (train_sft.py input)")
    parser.add_argument("--candidate-threshold", type=float, help="Eq. 3 tau (paper: 0.5)")
    parser.add_argument("--median-gate-threshold", type=float, help="skip LLM pruning above this median weight (paper: 1.0)")
    parser.add_argument("--temperature", type=float, help="paper: 0.9")
    parser.add_argument("--top-p", type=float, help="paper: 0.95")
    parser.add_argument("--max-tokens", type=int, help="max tokens for the agent's JSON reply")
    parser.add_argument("--max-length", type=int, default=None,
                        help="drop records longer than this many tokens (paper filtered >8192); unset/None keeps every sample")
    parser.add_argument("--max-model-len", type=int, help="vLLM context window for the pruning agent")
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--print-llm-responses", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    overrides = {
        "data_path": args.data_path,
        "weights_path": args.weights_path,
        "tokenizer": args.tokenizer,
        "pruning_agent": args.pruning_agent,
        "output_path": args.output_path,
        "candidate_threshold": args.candidate_threshold,
        "median_gate_threshold": args.median_gate_threshold,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "seed": args.seed,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    config.setdefault("candidate_threshold", 0.5)
    config.setdefault("median_gate_threshold", 1.0)
    config.setdefault("temperature", 0.9)
    config.setdefault("top_p", 0.95)
    config.setdefault("max_tokens", 8192)
    config.setdefault("max_model_len", 32768)
    config.setdefault("gpu_memory_utilization", 0.9)
    config.setdefault("tensor_parallel_size", 1)
    config.setdefault("seed", 42)

    from vllm import LLM, SamplingParams

    student_tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    agent_tokenizer = AutoTokenizer.from_pretrained(config["pruning_agent"])

    with open(config["data_path"]) as handle:
        records = [json.loads(line) for line in handle]
    if args.max_length is not None:
        kept = [r for r in records if len(r["input_ids"]) <= args.max_length]
        print(f"length filter (<= {args.max_length} tokens): kept {len(kept)}/{len(records)} records")
        records = kept
    weights_frame = pd.read_parquet(config["weights_path"])
    weights_by_id = {int(row.id): list(row.step_weights) for row in weights_frame.itertuples()}
    missing_weights = missing_weight_ids(records, weights_by_id)
    if missing_weights:
        preview = ", ".join(str(record_id) for record_id in missing_weights[:10])
        raise RuntimeError(
            f"missing Pru-CoT weights for {len(missing_weights)}/{len(records)} records "
            f"(first ids: {preview}); refusing to drop training samples"
        )

    kept_as_is, queue = [], []
    for record in records:
        weights = weights_by_id[int(record["id"])]
        if median_gate(weights, config["median_gate_threshold"]):
            kept_as_is.append(record)
            continue
        candidates = candidate_step_indices(weights, config["candidate_threshold"])
        if not candidates:
            kept_as_is.append(record)
            continue

        step_texts = decode_step_texts(student_tokenizer, record)
        n_scored = len(weights)
        thinking_steps = step_texts[:n_scored]
        solution_tail = step_texts[n_scored:]
        # Real solution if a "</think>" split was found; otherwise fall back to the last scored CoT step.
        solution_text = "".join(solution_tail) if solution_tail else (thinking_steps[-1] if thinking_steps else "")
        virtual_ids = build_virtual_id_map(candidates)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            user_input=record["prompt"],
            solution=solution_text,
            markdown_table_rows=format_step_table(thinking_steps, virtual_ids),
        )
        prompt_text = agent_tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        queue.append({
            "record": record, "step_texts": step_texts, "n_scored": n_scored,
            "virtual_ids": virtual_ids, "prompt": prompt_text,
        })

    print(f"median-gated/no-candidates (kept as-is): {len(kept_as_is)}; LLM pruning queue: {len(queue)}")

    real_prune_by_id: dict[int, list[int]] = {}
    failed = []
    if queue:
        llm = LLM(
            model=config["pruning_agent"],
            trust_remote_code=True,
            tensor_parallel_size=config["tensor_parallel_size"],
            max_model_len=config["max_model_len"],
            gpu_memory_utilization=config["gpu_memory_utilization"],
        )
        sampling = SamplingParams(
            temperature=config["temperature"], top_p=config["top_p"],
            max_tokens=config["max_tokens"], seed=config["seed"],
        )
        outputs = llm.generate([item["prompt"] for item in queue], sampling)

        for item, output in zip(queue, outputs):
            text = output.outputs[0].text.strip()
            if args.print_llm_responses:
                print(f"\n{text}\n")

            virtual_pruned, status = parse_prune_response(text)
            record_id = item["record"]["id"]
            if virtual_pruned is None:
                failed.append({"id": record_id, "error": status, "llm_output": text})
                real_prune_by_id[record_id] = []  # fallback: keep everything for this sample
                continue

            pruned_set = set(virtual_pruned)
            real_prune_by_id[record_id] = [
                real for real, virtual in item["virtual_ids"].items() if virtual in pruned_set
            ]

    # Uniform (record, prune_indices, n_scored, tokenizer, step_texts) shape for both branches --
    # kept_as_is records use the emit_prucot_record()/selection_stats() no-op defaults (empty
    # prune_indices), so both cases go through the same emit loop below.
    emit_items = [(record, set(), 0, None, None) for record in kept_as_is] + [
        (
            item["record"],
            set(real_prune_by_id.get(item["record"]["id"], [])),
            item["n_scored"],
            student_tokenizer,
            item["step_texts"],
        )
        for item in queue
    ]

    output_records, stats = [], []
    for record, prune_indices, n_scored, tokenizer, step_texts in emit_items:
        step_spans = record_step_spans(record)
        selected = [index for index in range(len(step_spans)) if index not in prune_indices]
        stats.append(selection_stats(step_spans, selected))
        output_records.append(emit_prucot_record(record, prune_indices, n_scored, tokenizer, step_texts))

    output_records.sort(key=lambda row: row["id"])
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for row in output_records:
            handle.write(json.dumps(row) + "\n")

    if failed:
        (output_path.parent / "prucot-failed-log.json").write_text(json.dumps(failed, indent=2))

    summary = summarize(stats)
    print(f"wrote {len(output_records)} samples -> {output_path} "
          f"({len(queue) - len(failed)} LLM-pruned, {len(failed)} parse-failed/kept-all, "
          f"{len(kept_as_is)} median-gated/no-candidates)")
    print(f"step drop: mean={summary['step_drop_mean']:.1%} median={summary['step_drop_median']:.1%}  "
          f"token drop: mean={summary['token_drop_mean']:.1%} median={summary['token_drop_median']:.1%}")
    (output_path.parent / "prucot-selection-stats.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
