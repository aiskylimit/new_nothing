"""Run native lm-evaluation-harness MATH-500 and GSM8K tasks, saving samples."""
import argparse
import importlib.metadata
import subprocess
import sys
import os
from pathlib import Path
from .data import save_json
from .local_assets import write_eval_tasks


def command(args):
    model_args = f"pretrained={args.model},dtype={args.dtype},enable_thinking=False"
    if args.adapter:
        model_args += f",peft={args.adapter}"
    if args.revision:
        model_args += f",revision={args.revision}"
    mapping = getattr(args, "task_map", None) or {"minerva_math500": "minerva_math500", "gsm8k_cot": "gsm8k_cot"}
    cmd = [sys.executable, "-m", "lm_eval", "--model", "hf", "--model_args", model_args,
           "--tasks", ",".join(mapping.values()), "--batch_size", str(args.batch_size),
           "--device", args.device, "--output_path", args.out, "--log_samples",
           "--seed", ",".join([str(args.seed)] * 4),
           "--gen_kwargs", f"max_gen_toks={args.max_new_tokens},do_sample=False",
           "--apply_chat_template", "--fewshot_as_multiturn"]
    if getattr(args, "local_data_root", None):
        cmd += ["--include_path", str(Path(args.out).resolve() / "local_tasks")]
    if args.num_fewshot is not None:
        cmd += ["--num_fewshot", str(args.num_fewshot)]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Base model ID or full checkpoint")
    p.add_argument("--adapter", default=None, help="LoRA checkpoint; --model must be base model")
    p.add_argument("--revision", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--num-fewshot", type=int, default=None, help="Omit to keep task defaults: MATH=4, GSM=8")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="Debug subset; never label this full benchmark")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--local-data-root", default=None, help="Local raw/ directory containing math500 and gsm8k test parquet")
    args = p.parse_args()
    if args.local_data_root:
        os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        cache = Path(args.local_data_root).resolve().parent / "cache/huggingface"
        os.environ.setdefault("HF_HOME", str(cache))
        os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))
        args.task_map = write_eval_tasks(args.local_data_root, Path(args.out) / "local_tasks")
    else:
        args.task_map = {"minerva_math500": "minerva_math500", "gsm8k_cot": "gsm8k_cot"}
    cmd = command(args)
    try:
        version = importlib.metadata.version("lm_eval")
    except importlib.metadata.PackageNotFoundError:
        version = None
    save_json(Path(args.out) / "evaluation_manifest.json",
              {**vars(args), "command": cmd, "lm_eval_version": version,
               "scope": "debug_subset" if args.limit is not None else "full_test_splits",
               "math500_samples": 500, "gsm8k_samples": 1319,
               "decoding": "greedy; chat template; enable_thinking=False; native task graders"})
    print(" ".join(cmd), flush=True)
    if not args.dry_run:
        if version is None:
            raise RuntimeError("Install requirements-opd-eval.txt in a separate eval environment")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
