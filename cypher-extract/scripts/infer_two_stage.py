#!/usr/bin/env python3
"""Run selector -> predicted sub-schema -> Cypher generation inference."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from distillation.utils import seed_everything  # noqa: E402
from schema_grounding.inference.checkpoints import (  # noqa: E402
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_METHODS,
    DEFAULT_MODEL_FAMILY,
    SUPPORTED_METHODS,
    resolve_checkpoint_directory,
    resolve_last_checkpoint,
)
from schema_grounding.inference.data import default_dataset_specs  # noqa: E402
from schema_grounding.inference.model import ModelRunner  # noqa: E402
from schema_grounding.inference.pipeline import (  # noqa: E402
    InferenceOptions,
    model_runner_required,
    prepare_run_directory,
    run_dataset_pipeline,
)
from schema_grounding.inference.prompting import PromptTemplates, chat_template_metadata  # noqa: E402

DEFAULT_INFERENCE_SEEDS = (10, 42, 50, 100, 1234)


def comma_separated(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seeds(value: str) -> list[int]:
    raw_seeds = comma_separated(value)
    if not raw_seeds:
        raise ValueError("--seeds must contain at least one seed")
    try:
        seeds = [int(seed) for seed in raw_seeds]
    except ValueError as error:
        raise ValueError("--seeds must be comma-separated integers") from error
    if any(seed < 0 for seed in seeds):
        raise ValueError("--seeds must contain only non-negative integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds contains duplicates")
    return seeds


def parse_args() -> argparse.Namespace:
    datasets = default_dataset_specs(REPOSITORY_ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="Local root containing <model-family>/<method>/checkpoint-N directories.",
    )
    parser.add_argument("--model-family", default=DEFAULT_MODEL_FAMILY)
    parser.add_argument(
        "--methods",
        default="all",
        help=(
            "Comma-separated methods or 'all'. 'all' preserves the default LoRA matrix; "
            "teacher_full can be requested explicitly."
        ),
    )
    parser.add_argument(
        "--datasets",
        default=",".join(datasets),
        help=f"Comma-separated datasets. Choices: {', '.join(datasets)}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Inference output root. Defaults to results/inference/lora/<model-family>.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--selector-batch-size", type=int, default=128)
    parser.add_argument("--generator-batch-size", type=int, default=16)
    parser.add_argument("--generator-max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_INFERENCE_SEEDS),
        help="Comma-separated inference seeds; each seed is written to its own seed<value> folder.",
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument(
        "--no-merge-adapter",
        action="store_true",
        help="Keep LoRA modules separate instead of merging them into the base model.",
    )
    parser.add_argument(
        "--no-relation-endpoint-closure",
        action="store_true",
        help="Do not add endpoint nodes for selected relationships.",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path("results/inference/lora") / args.model_family
    return args


def validate_choices(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    methods = list(DEFAULT_METHODS) if args.methods == "all" else comma_separated(args.methods)
    if not methods:
        raise ValueError("--methods must contain at least one method")
    unknown_methods = sorted(set(methods).difference(SUPPORTED_METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown methods: {', '.join(unknown_methods)}")
    if len(methods) != len(set(methods)):
        raise ValueError("--methods contains duplicates")

    available_datasets = default_dataset_specs(REPOSITORY_ROOT)
    datasets = comma_separated(args.datasets)
    if not datasets:
        raise ValueError("--datasets must contain at least one dataset")
    unknown_datasets = sorted(set(datasets).difference(available_datasets))
    if unknown_datasets:
        raise ValueError(f"Unknown datasets: {', '.join(unknown_datasets)}")
    if len(datasets) != len(set(datasets)):
        raise ValueError("--datasets contains duplicates")
    return methods, datasets


def build_seed_first_run_groups(
    *,
    methods: list[str],
    dataset_names: list[str],
    seeds: list[int],
    output_root: Path,
    options: InferenceOptions,
) -> list[tuple[int, str, list[tuple[str, Path, InferenceOptions]]]]:
    """Plan every dataset run, completing one seed before moving to the next."""

    groups = []
    for seed in seeds:
        seed_options = replace(options, seed=seed)
        for method in methods:
            runs = [
                (
                    dataset_name,
                    output_root / f"seed{seed}" / method / dataset_name,
                    seed_options,
                )
                for dataset_name in dataset_names
            ]
            groups.append((seed, method, runs))
    return groups


def main() -> None:
    args = parse_args()
    methods, dataset_names = validate_choices(args)
    seeds = parse_seeds(args.seeds)
    specs = default_dataset_specs(REPOSITORY_ROOT)
    templates = PromptTemplates.from_repository(REPOSITORY_ROOT)
    options = InferenceOptions(
        selector_batch_size=args.selector_batch_size,
        generator_batch_size=args.generator_batch_size,
        generator_max_new_tokens=args.generator_max_new_tokens,
        close_relation_endpoints=not args.no_relation_endpoint_closure,
        generator_do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
    )
    options.validate()

    print(
        json.dumps(
            {
                "checkpoint_root": str(args.checkpoint_root.resolve()),
                "methods": methods,
                "datasets": dataset_names,
                "seeds": seeds,
                "selector_decoding": {
                    "labels": ["YES", "NO"],
                    "output_format": {"label": "YES|NO"},
                    "do_sample": False,
                    "num_beams": 1,
                    "max_new_tokens": options.selector_max_new_tokens,
                },
                "generator_decoding": {
                    "do_sample": options.generator_do_sample,
                    "temperature": options.temperature,
                    "top_p": options.top_p,
                    "top_k": options.top_k,
                    "num_beams": options.num_beams,
                },
                "chat_template": chat_template_metadata(args.model_family)["name"],
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    checkpoints = {}
    checkpoint_paths = {}
    for method in methods:
        checkpoints[method] = resolve_last_checkpoint(
            method,
            checkpoint_root=args.checkpoint_root,
            model_family=args.model_family,
        )
        checkpoint = checkpoints[method]
        print(
            f"[{method}] resolved {checkpoint.uri} "
            f"(step {checkpoint.step}, fingerprint {checkpoint.fingerprint[:12]})"
        )
        # Validate every local model/adapter before creating any resumable
        # output directory or run_config.json file.
        checkpoint_paths[method] = resolve_checkpoint_directory(checkpoint)

    run_groups = build_seed_first_run_groups(
        methods=methods,
        dataset_names=dataset_names,
        seeds=seeds,
        output_root=args.output_dir.resolve(),
        options=options,
    )
    for _seed, method, planned_runs in run_groups:
        checkpoint = checkpoints[method]
        for dataset_name, output_directory, seed_options in planned_runs:
            prepare_run_directory(
                method=method,
                checkpoint=checkpoint,
                spec=specs[dataset_name],
                templates=templates,
                output_directory=output_directory,
                options=seed_options,
            )

    for seed in seeds:
        print(f"[seed{seed}] starting all methods and datasets")
        for group_seed, method, planned_runs in run_groups:
            if group_seed != seed:
                continue
            checkpoint = checkpoints[method]
            runner = None
            if any(model_runner_required(output_directory) for _, output_directory, _ in planned_runs):
                runner = ModelRunner.from_checkpoint(
                    checkpoint_paths[method],
                    dtype=args.dtype,
                    device=args.device,
                    merge_adapter=not args.no_merge_adapter,
                    model_family=args.model_family,
                )
            else:
                print(f"[seed{seed}/{method}] all model-backed stages are complete; skipping model load")
            try:
                for dataset_name, output_directory, seed_options in planned_runs:
                    seed_everything(seed, rank_offset=False)
                    print(f"[seed{seed}/{method}/{dataset_name}] starting")
                    run_dataset_pipeline(
                        method=method,
                        checkpoint=checkpoint,
                        spec=specs[dataset_name],
                        runner=runner,
                        templates=templates,
                        output_directory=output_directory,
                        options=seed_options,
                    )
                    print(f"[seed{seed}/{method}/{dataset_name}] completed")
            finally:
                if runner is not None:
                    del runner
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        print(f"[seed{seed}] completed all methods and datasets")


if __name__ == "__main__":
    main()
