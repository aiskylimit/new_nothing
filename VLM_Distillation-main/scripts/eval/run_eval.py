#!/usr/bin/env python3
"""Validated single-checkpoint evaluation launcher."""

import argparse
import hashlib
import os
import shlex
import subprocess
import sys
from pathlib import Path

from eval_utils import (PINNED_VLMEVAL_COMMIT, EvalError, atomic_json, checkpoint_identity,
                        classify_checkpoint, detect_architecture, git_commit, output_dir,
                        md5_file, parse_overrides, read_json, sanitize, select_checkpoint, sha256_file,
                        run_identity_name, suite_config, update_summary, utc_now)
from prepare_eval_assets import check_environment


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-dir", type=Path, required=True)
    p.add_argument("--python-bin", type=Path, required=True)
    p.add_argument("--vlmevalkit-dir", type=Path, required=True)
    p.add_argument("--lmu-data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--suite", default="requested_benchmarks")
    p.add_argument("--suite-config", type=Path)
    p.add_argument("--model-name")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--outputs-root", type=Path)
    p.add_argument("--mode", choices=("all", "infer", "eval"), default="all")
    p.add_argument("--attention-backend", choices=("eager", "sdpa", "flash_attention_2"), default="eager")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-samples", type=int,
                   help="Limit each dataset through a project-owned smoke dataset adapter.")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--merge-device-map", default="cpu")
    p.add_argument("--base-model", type=Path,
                   help="Explicit local base model for an adapter checkpoint; overrides its embedded base path.")
    p.add_argument("--base-model-override", action="append", default=[], metavar="OLD=NEW")
    p.add_argument("--judge", default=os.environ.get("JUDGE") or None,
                   help="Opt-in judge model; suite files never silently enable it.")
    p.add_argument("--reuse", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prepare-missing-assets", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-asset-check", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def validate_manifest(args, cfg, commit):
    path = args.project_dir / "outputs/eval/manifests" / f"{cfg.get('name', args.suite)}.json"
    if not path.is_file():
        raise EvalError(f"Asset manifest absent: {path}. Run scripts/eval/prepare_eval_assets.sh.")
    data = read_json(path)
    errors = []
    if not data.get("valid"): errors.append("marked invalid")
    if data.get("suite_config_sha256") != sha256_file(args.suite_config): errors.append("suite changed")
    if data.get("vlmevalkit_commit") != commit: errors.append("VLMEvalKit commit changed")
    if Path(data.get("lmu_data", "")).resolve() != args.lmu_data: errors.append("LMUData path changed")
    entries = {row.get("dataset"): row for row in data.get("datasets", [])}
    for name in cfg["datasets"]:
        row = entries.get(name)
        if not row or row.get("status") != "ok":
            errors.append(f"invalid dataset {name}")
            continue
        cache = Path(row.get("cache_path", ""))
        if not cache.is_file() or cache.stat().st_size == 0:
            errors.append(f"missing/empty cache {name}")
        elif row.get("actual_checksum") and md5_file(cache) != row["actual_checksum"]:
            errors.append(f"cache checksum changed {name}")
    if errors:
        raise EvalError("Asset manifest stale/invalid: " + "; ".join(errors) +
                        ". Run scripts/eval/prepare_eval_assets.sh.")
    return path, {name: entries[name]["resolved_class"] for name in cfg["datasets"]}


def merged_complete(path, expected):
    meta = path / "lora_merge_metadata.json"
    if classify_checkpoint(path)[0] != "merged/full" or not meta.is_file(): return False
    actual = read_json(meta)
    return actual.get("status") == "complete" and all(actual.get(k) == v for k, v in expected.items())


def merge_adapter(args, checkpoint, run_name, architecture, base, revision):
    identity = checkpoint_identity(checkpoint, run_name, base)
    merged = args.project_dir / "outputs/eval/merged_checkpoints" / sanitize(run_name) / f"{checkpoint.name}-{identity}"
    expected = {"adapter": str(checkpoint), "base_model": base, "revision": revision,
                "architecture": architecture, "identity": identity}
    if merged_complete(merged, expected): return merged, expected
    if merged.exists() and any(merged.iterdir()):
        raise EvalError(f"Will not overwrite incomplete or mismatched merge: {merged}")
    cmd = [str(args.python_bin), str(args.project_dir / "scripts/eval/merge_lora.py"),
           "--adapter", str(checkpoint), "--output", str(merged), "--base-model", base,
           "--device-map", args.merge_device_map, "--identity", identity,
           "--expected-architecture", architecture]
    if revision: cmd += ["--revision", revision]
    subprocess.run(cmd, check=True, cwd=args.project_dir)
    if not merged_complete(merged, expected): raise EvalError(f"Merged checkpoint failed validation: {merged}")
    return merged, expected


def write_config(path, name, checkpoint, architecture, datasets, dataset_classes, args):
    if args.mode == "eval":
        cls = "OfflineScoringModel"
    elif architecture == "fast_vlm":
        cls = "FastVLMChat"
    elif architecture == "qwen3_vl":
        cls = "ProjectQwen3VLChat"
    else:
        cls = "ProjectQwen2VLChat"
    model = {"class": cls, "model_path": str(checkpoint.resolve()),
             "attention_backend": args.attention_backend, "max_new_tokens": args.max_new_tokens,
             "do_sample": False, "use_custom_prompt": architecture != "fast_vlm", "device_map": args.device_map}
    if architecture != "fast_vlm": model["model_backbone"] = architecture
    data = {}
    for dataset in datasets:
        entry = {"class": dataset_classes[dataset], "dataset": dataset}
        max_samples = getattr(args, "max_samples", None)
        if max_samples is not None:
            if max_samples <= 0:
                raise EvalError("--max-samples must be a positive integer")
            entry["max_samples"] = max_samples
        data[dataset] = entry
    atomic_json(path, {"model": {name: model}, "data": data})


def mode_is_complete(summary_path, mode):
    try:
        summary = read_json(summary_path)
        rows = summary.get("benchmarks") or []
        if not rows:
            return False
        if mode == "infer":
            return all(row.get("inference_status") == "complete" for row in rows)
        if mode == "eval":
            return all(row.get("scoring_status") == "complete" for row in rows)
        return summary.get("overall_status") == "complete"
    except EvalError:
        return False


def effective_judge(requested):
    """Prevent VLMEvalKit's MCQ/Y/N default from silently selecting an API judge."""
    return requested or "exact_matching"


def main():
    args = parse_args()
    # Keep the virtualenv launcher path intact.  Path.resolve() dereferences
    # .venv/bin/python to uv's base interpreter, which then loses the venv's
    # site-packages in child processes such as merge_lora.py.
    args.python_bin = args.python_bin.expanduser().absolute()
    for name in ("project_dir", "vlmevalkit_dir", "lmu_data", "checkpoint"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.suite_config = (args.suite_config or args.project_dir / "configs/eval" / f"{args.suite}.json").resolve()
    args.outputs_root = (args.outputs_root or args.project_dir / "outputs").resolve()
    cfg = suite_config(args.suite_config)
    environment_failures = check_environment(args.project_dir)
    if environment_failures:
        raise EvalError("Selected repository Python failed environment validation: " +
                        "; ".join(environment_failures) +
                        ". No system-Python fallback is used.")
    commit = git_commit(args.vlmevalkit_dir)
    if commit != PINNED_VLMEVAL_COMMIT:
        print(f"WARNING: VLMEvalKit commit {commit}, expected {PINNED_VLMEVAL_COMMIT}", file=sys.stderr)
    manifest, dataset_classes = None, None
    if not args.skip_asset_check:
        try:
            manifest, dataset_classes = validate_manifest(args, cfg, commit)
        except EvalError:
            if not args.prepare_missing_assets: raise
            env = {**os.environ, "PYTHON_BIN": str(args.python_bin), "VLMEVALKIT_DIR": str(args.vlmevalkit_dir),
                   "LMUData": str(args.lmu_data), "SUITE_CONFIG": str(args.suite_config)}
            subprocess.run([str(args.project_dir / "scripts/eval/prepare_eval_assets.sh")], check=True,
                           cwd=args.project_dir, env=env)
            manifest, dataset_classes = validate_manifest(args, cfg, commit)
    if dataset_classes is None:
        # Test-only hidden mode still emits structurally explicit configs.
        dataset_classes = {name: "ImageBaseDataset" for name in cfg["datasets"]}
    class_overrides = cfg.get("dataset_class_overrides", {})
    if not isinstance(class_overrides, dict):
        raise EvalError("dataset_class_overrides must be a JSON object")
    unknown_overrides = sorted(set(class_overrides) - set(cfg["datasets"]))
    if unknown_overrides:
        raise EvalError("Dataset class override names are not in the suite: " + ", ".join(unknown_overrides))
    dataset_classes.update(class_overrides)
    checkpoint = args.checkpoint
    if classify_checkpoint(checkpoint)[0] not in {"adapter-only", "merged/full"}:
        checkpoint = select_checkpoint(checkpoint) or checkpoint
    kind, reasons = classify_checkpoint(checkpoint)
    if kind not in {"adapter-only", "merged/full"}:
        raise EvalError(f"Checkpoint {checkpoint} is {kind}: {', '.join(reasons)}")
    base_overrides = parse_overrides(args.base_model_override)
    if args.base_model is not None:
        if kind != "adapter-only":
            raise EvalError("--base-model is only valid for an adapter-only LoRA checkpoint")
        forced_base = args.base_model.expanduser().resolve()
        if not (forced_base / "config.json").is_file():
            raise EvalError(f"Explicit local base model is invalid: {forced_base}")
        adapter_cfg = read_json(checkpoint / "adapter_config.json")
        embedded_base = adapter_cfg.get("base_model_name_or_path")
        if not embedded_base:
            raise EvalError(f"Adapter does not declare base_model_name_or_path: {checkpoint}")
        base_overrides[str(embedded_base)] = str(forced_base)
    architecture, base, revision = detect_architecture(checkpoint, base_overrides)
    if architecture == "fast_vlm" and args.attention_backend != "eager":
        print("WARNING: FastVLM's Timm vision tower does not support SDPA; using eager attention.",
              file=sys.stderr)
        args.attention_backend = "eager"
    run_name = run_identity_name(checkpoint, args.outputs_root)
    effective, merge_meta = checkpoint, None
    if kind == "adapter-only" and args.mode != "eval" and not args.dry_run:
        effective, merge_meta = merge_adapter(args, checkpoint, run_name, architecture, base, revision)
    model_name = sanitize(args.model_name or f"{run_name}-{checkpoint.name}")
    if args.output_dir:
        run_dir = args.output_dir.resolve()
    else:
        run_dir = output_dir(args.outputs_root, cfg.get("name", args.suite), run_name, checkpoint)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "vlmeval_config.json"
    write_config(config_path, model_name, effective, architecture, cfg["datasets"], dataset_classes, args)
    native = run_dir / "native"
    cmd = [str(args.python_bin), str(args.project_dir / "scripts/eval/vlmeval_entrypoint.py"),
           "--config", str(config_path), "--work-dir", str(native), "--mode", args.mode]
    if args.reuse: cmd.append("--reuse")
    resolved_judge = effective_judge(args.judge)
    cmd += ["--judge", resolved_judge]
    context = {"run_name": run_name, "checkpoint": str(checkpoint), "checkpoint_kind": kind,
               "effective_checkpoint": str(effective), "architecture": architecture, "model_name": model_name,
               "suite": cfg.get("name", args.suite), "datasets": cfg["datasets"], "created_at": utc_now(),
               "vlmevalkit_commit": commit, "dataset_manifest": str(manifest) if manifest else None,
               "generated_config": str(config_path), "run_dir": str(run_dir), "native_work_dir": str(native),
               "commands": {"vlmeval": shlex.join(cmd)}, "merge_metadata": merge_meta,
               "runtime": {"mode": args.mode, "reuse": args.reuse, "attention_backend": args.attention_backend,
                           "max_new_tokens": args.max_new_tokens, "device_map": args.device_map,
                           "max_samples": args.max_samples,
                           "merge_device_map": args.merge_device_map, "judge_model": args.judge,
                           "effective_judge": resolved_judge,
                           "explicit_base_model": str(args.base_model.resolve()) if args.base_model else None,
                           "base_model_overrides": list(args.base_model_override)}}
    context_path = run_dir / "run_context.json"
    atomic_json(context_path, context)
    update_summary(context_path)
    print(f"Run directory: {run_dir}\nArchitecture: {architecture}; checkpoint: {kind}\nCommand: {shlex.join(cmd)}")
    if args.dry_run: return 0
    env = {**os.environ, "PROJECT_DIR": str(args.project_dir), "VLMEVALKIT_DIR": str(args.vlmevalkit_dir),
           "LMUData": str(args.lmu_data), "EVAL_CONTEXT": str(context_path)}
    result = subprocess.run(cmd, cwd=args.vlmevalkit_dir, env=env)
    summary_path = update_summary(context_path)
    if result.returncode:
        return result.returncode
    if not mode_is_complete(summary_path, args.mode):
        print(f"ERROR: VLMEvalKit exited successfully but mode {args.mode!r} is incomplete; see {summary_path}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (EvalError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
