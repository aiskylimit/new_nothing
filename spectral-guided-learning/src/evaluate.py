"""Phase 6: generate with vLLM and score both checkpoints on the paper's benchmarks.

    python src/evaluate.py --rescore results/spectral/raw/math500.jsonl   # no GPU needed

Raw generations are persisted under --results-dir/<tag>/raw/ so scoring can be revised with
--rescore. A run always regenerates them; it never reuses a file from an earlier run.
"""

import argparse
import json
from pathlib import Path

import yaml

import palign_grader
from answer_scoring import score_generation
from benchmarks import BENCHMARKS


def build_prompts(model_path: str, records: list[dict], config: dict) -> list[str]:
    """Wrap each benchmark's plain instruction into a chat-templated prompt when requested.

    Instruct/hybrid-thinking models need the chat template so generation starts from the
    assistant turn as trained, rather than continuing raw text like a base model.
    """
    if not config.get("chat_template"):
        return [record["prompt"] for record in records]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.get("base_model") or model_path)
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": record["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config.get("enable_thinking", True),
        )
        for record in records
    ]


def generate(model_path: str, records: list[dict], config: dict):
    """Yield (records_slice, generations) a batch at a time.

    Batched rather than one llm.generate() over the whole benchmark so the caller can persist
    finished problems as they land: a 1500-generation run is hours long, and a stop partway
    through should not throw away everything already produced.
    """
    from vllm import LLM, SamplingParams

    prompts = build_prompts(model_path, records, config)

    # lora_adapter: model_path is adapter-only (--no-lora-merge); load base weights once via
    # vLLM's native LoRA support instead of a ~16GB merged copy per model.
    if config.get("lora_adapter"):
        from vllm.lora.request import LoRARequest

        llm = LLM(
            model=config["base_model"],
            max_model_len=config["max_model_len"],
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.9),
            dtype="bfloat16",
            enable_lora=True,
            max_lora_rank=config.get("lora_r", 16),
            enforce_eager=config.get("enforce_eager", True),
        )
        lora_request = LoRARequest("adapter", 1, model_path)
    else:
        llm = LLM(
            model=model_path,
            max_model_len=config["max_model_len"],
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.9),
            dtype="bfloat16",
            enforce_eager=config.get("enforce_eager", True),
        )
        lora_request = None

    sampling = SamplingParams(
        n=config["n_samples"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        repetition_penalty=config.get("repetition_penalty", 1.0),
        max_tokens=config["max_tokens"],
        seed=config.get("seed", 42),
    )
    batch_size = config.get("batch_size") or len(records)
    for start in range(0, len(records), batch_size):
        stop = start + batch_size
        outputs = llm.generate(prompts[start:stop], sampling, lora_request=lora_request)
        yield records[start:stop], [
            [
                {
                    "text": completion.text,
                    "finish_reason": completion.finish_reason,
                    "n_tokens": len(completion.token_ids),  # generated length ("Length" metric)
                }
                for completion in output.outputs
            ]
            for output in outputs
        ]


def label_generations(texts: list[str], gold: str, task_type: str, grader: str) -> list[int]:
    """0/1 per generation. grader="palign" uses math_verify OR oat_math_grader on math tasks."""
    if grader == "palign" and task_type == "math":
        return palign_grader.grade(texts, gold)
    return [int(score_generation(text, gold, task_type)) for text in texts]


def score_file(path: Path, grader: str = "palign") -> dict:
    """Score a raw generations file; returns Pass@1 / Pass@k summary."""
    with path.open() as handle:
        rows = [json.loads(line) for line in handle]

    correct = total = truncated = unboxed = token_sum = 0
    solved = 0  # problems with >= 1 correct sample -- the Pass@k numerator
    samples_per_problem = 0
    for row in rows:
        texts = [generation["text"] for generation in row["generations"]]
        labels = label_generations(texts, row["gold"], row["task_type"], grader)
        row_correct = sum(labels)
        for text, generation in zip(texts, row["generations"]):
            total += 1
            # hit the max_tokens cap = the model never finished its CoT (paper caps at 32k)
            truncated += int(generation.get("finish_reason") == "length")
            unboxed += int(len(text) > 0 and "\\boxed" not in text)
            token_sum += generation.get("n_tokens", 0)  # 0 for pre-existing files without the field
        correct += row_correct
        solved += int(row_correct > 0)
        samples_per_problem = max(samples_per_problem, len(row["generations"]))

    # Pass@1 = mean over all k samples; Pass@k = share of problems with any sample correct.
    summary = {
        "benchmark": path.stem,
        "accuracy": correct / total if total else 0.0,  # == pass@1, kept for older summaries
        "pass@1": correct / total if total else 0.0,
        "samples_per_problem": samples_per_problem,
        "length": token_sum / total if total else 0.0,  # mean generated tokens ("Length" metric)
        "n_problems": len(rows),
        "n_generations": total,
        "truncation_rate": truncated / total if total else 0.0,
        "no_boxed_answer_rate": unboxed / total if total else 0.0,
    }
    if samples_per_problem > 1:
        summary[f"pass@{samples_per_problem}"] = solved / len(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--model", help="checkpoint path")
    parser.add_argument("--tag", help="short name used in output filenames (default: dir name)")
    parser.add_argument("--benchmarks", default="all")
    parser.add_argument("--rescore", help="score an existing raw generations file and exit")
    parser.add_argument(
        "--shard",
        help="run only part of each benchmark, as i/n (e.g. 0/2). Problems are taken "
             "round-robin, so every shard gets the same difficulty mix and finishes in "
             "about the same time. Raw/summary files are suffixed -shard<i>of<n>.",
    )
    # generation/sampling settings (override the yaml, or stand in for it entirely)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="problems per generate() call; finished ones are written after each batch so a "
             "stop keeps them (default: the whole benchmark in one call)",
    )
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--results-dir", help="root dir; each run writes under <results-dir>/<tag>/")
    # chat template (instruct/hybrid-thinking models) + LoRA adapter (--no-lora-merge checkpoints)
    parser.add_argument("--chat-template", action=argparse.BooleanOptionalAction)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction)
    parser.add_argument("--base-model", help="base model for chat template / LoRA adapter loading")
    parser.add_argument("--lora-adapter", action=argparse.BooleanOptionalAction)
    parser.add_argument("--lora-r", type=int, help="adapter rank, for vLLM's max_lora_rank")
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument(
        "--grader",
        default="palign",
        choices=("palign", "builtin"),
        help="palign: math_verify OR oat_math_grader (P-ALIGN's own); builtin: this repo's scorer",
    )
    args = parser.parse_args()

    if args.rescore:
        print(json.dumps(score_file(Path(args.rescore), args.grader), indent=2))
        return

    config = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    overrides = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "n_samples": args.n_samples,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "batch_size": args.batch_size,
        "enforce_eager": args.enforce_eager,
        "seed": args.seed,
        "results_dir": args.results_dir,
        "chat_template": args.chat_template,
        "enable_thinking": args.enable_thinking,
        "base_model": args.base_model,
        "lora_adapter": args.lora_adapter,
        "lora_r": args.lora_r,
        "repetition_penalty": args.repetition_penalty,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    tag = args.tag or Path(args.model).name
    names = list(BENCHMARKS) if args.benchmarks == "all" else args.benchmarks.split(",")

    shard_index, shard_count = 0, 1
    if args.shard:
        shard_index, shard_count = (int(part) for part in args.shard.split("/"))
        if not 0 <= shard_index < shard_count:
            parser.error(f"--shard {args.shard}: need 0 <= i < n")
    suffix = f"-shard{shard_index}of{shard_count}" if shard_count > 1 else ""

    run_dir = Path(config["results_dir"]) / tag
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    # Always regenerate: a stale raw file from an earlier run can hold a different sample
    # count, prompt shape or sampling regime, and silently mixing those into one summary
    # produces numbers that look fine but are not comparable. Use --rescore to re-score a
    # kept file on purpose.
    for name in names:
        raw_path = raw_dir / f"{name}{suffix}.jsonl"
        records = BENCHMARKS[name]()[shard_index::shard_count]
        print(f"[{name}] generating for {len(records)} problems x {config['n_samples']}")
        generations = generate(args.model, records, config)
        with raw_path.open("w") as handle:
            for record, completions in zip(records, generations):
                handle.write(
                    json.dumps(
                        {
                            "id": record["id"],
                            "gold": record["gold"],
                            "task_type": record["task_type"],
                            "generations": completions,
                        }
                    )
                    + "\n"
                )

        summary = score_file(raw_path, args.grader)
        summary["model"] = tag
        summaries.append(summary)
        k = summary["samples_per_problem"]
        print(
            f"[{name}] pass@1 = {summary['pass@1']:.1%}  "
            + (f"pass@{k} = {summary[f'pass@{k}']:.1%}  " if k > 1 else "")
            + f"length = {summary['length']:.0f} tok  "
            f"truncated = {summary['truncation_rate']:.1%}"
        )

    results_path = run_dir / f"summary{suffix}.json"
    results_path.write_text(json.dumps(summaries, indent=2))
    average = sum(s["pass@1"] for s in summaries) / len(summaries)
    avg_length = sum(s["length"] for s in summaries) / len(summaries)
    line = f"\n{tag}: Overall pass@1 = {average:.1%}"

    # Only average Pass@k over benchmarks that actually carry that k -- reused raw files can
    # hold a different sample count than this run requested, and treating a missing key as 0%
    # would quietly understate the aggregate.
    k = max(s["samples_per_problem"] for s in summaries)
    if k > 1:
        at_k = [s[f"pass@{k}"] for s in summaries if f"pass@{k}" in s]
        line += f"  Overall pass@{k} = {sum(at_k) / len(at_k):.1%}"
        if len(at_k) != len(summaries):
            line += f" (over {len(at_k)}/{len(summaries)} benchmarks at k={k})"
    print(f"{line}  Overall length = {avg_length:.0f} tok -> {results_path}")


if __name__ == "__main__":
    main()
