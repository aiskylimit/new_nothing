#!/usr/bin/env python3
"""Resolve, prepare, and validate VLMEvalKit datasets without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from unittest import mock
from pathlib import Path

from eval_utils import (PINNED_VLMEVAL_COMMIT, EvalError, atomic_json, git_commit,
                        parse_overrides, read_json, suite_config, utc_now)


REQUIRED_IMPORTS = {
    "pandas": "Install this repository's evaluation dependencies into .venv (pandas is required by VLMEvalKit).",
    "numpy": "Install numpy into the repository .venv.",
    "PIL": "Install Pillow into the repository .venv.",
    "torch": "Install the project dependencies into the repository .venv.",
    "transformers": "Install the project dependencies into the repository .venv.",
    "peft": "Install peft into the repository .venv for LoRA preparation.",
    "qwen_vl_utils": "Install qwen-vl-utils into the repository .venv.",
    "tabulate": "Install the VLMEvalKit requirements into the repository .venv.",
    "openpyxl": "Install the VLMEvalKit requirements into the repository .venv.",
    "xlsxwriter": "Install the VLMEvalKit requirements into the repository .venv.",
    "dotenv": "Install python-dotenv into the repository .venv.",
    "validators": "Install the VLMEvalKit requirements into the repository .venv.",
    "portalocker": "Install the VLMEvalKit requirements into the repository .venv.",
    "json_repair": "Install the VLMEvalKit requirements into the repository .venv.",
    "Levenshtein": "Install the VLMEvalKit requirements into the repository .venv.",
    "distance": "Install the VLMEvalKit requirements into the repository .venv.",
    "rouge_score": "Install rouge-score into the selected evaluation environment.",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--project-dir", type=Path, required=True)
    result.add_argument("--vlmevalkit-dir", type=Path, required=True)
    result.add_argument("--lmu-data", type=Path, required=True)
    result.add_argument("--suite-config", type=Path, required=True)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Validate existing assets; no network and no writes.")
    mode.add_argument("--dry-run", action="store_true", help="Resolve and print planned assets; no network and no writes.")
    mode.add_argument("--offline", action="store_true",
                      help="Validate local assets and write the manifest without dataset/model downloads.")
    result.add_argument("--checkpoints", nargs="*", type=Path, default=[])
    result.add_argument("--download-models", action="store_true", help="Explicitly prefetch missing adapter bases.")
    result.add_argument("--base-model-override", action="append", default=[], metavar="OLD=NEW")
    return result


def check_environment(project_dir: Path | None = None) -> list[str]:
    failures = []
    for module, advice in REQUIRED_IMPORTS.items():
        if importlib.util.find_spec(module) is None:
            failures.append(f"missing Python package '{module}': {advice}")
    if not failures and project_dir is not None:
        sys.path.insert(0, str(project_dir))
        for module in ("src.arguments", "src.model.processor"):
            try:
                __import__(module)
            except Exception as exc:
                failures.append(f"cannot import {module}: {exc}")
    return failures


def md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - upstream supplies MD5 asset identities
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_registry_classes(classes, names):
    resolved = {}
    for cls in classes:
        try:
            supported = cls.supported_datasets()
        except Exception:
            continue
        for name in names:
            if name in supported and name not in resolved:
                resolved[name] = cls
    missing = [name for name in names if name not in resolved]
    if missing:
        raise EvalError("Unsupported datasets in this VLMEvalKit checkout: " + ", ".join(missing))
    return resolved


def registry(vlmevalkit: Path, names: list[str], lmu_data: Path | None = None):
    sys.path.insert(0, str(vlmevalkit))
    original_exists = os.path.exists
    explicit = os.path.abspath(str(lmu_data)) if lmu_data is not None else None

    def explicit_cache_exists(path):
        if explicit is not None and os.path.abspath(os.fspath(path)) == explicit:
            return True
        return original_exists(path)

    # Upstream ignores $LMUData when the directory does not yet exist. In
    # no-write modes, make only this exact configured path appear present while
    # importing the registry so no module falls back to $HOME/LMUData.
    try:
        with mock.patch("os.path.exists", side_effect=explicit_cache_exists):
            from vlmeval.dataset import DATASET_CLASSES
    except Exception as exc:
        raise EvalError(f"Cannot import VLMEvalKit dataset registry from {vlmevalkit}: {exc}") from exc
    return resolve_registry_classes(DATASET_CLASSES, names)


def validate_dataset(name, cls, lmu_data: Path, instance=None) -> tuple[dict, list[str]]:
    import pandas as pd

    expected = getattr(cls, "DATASET_MD5", {}).get(name)
    source = getattr(cls, "DATASET_URL", {}).get(name)
    path = Path(getattr(instance, "data_path", lmu_data / f"{name}.tsv"))
    failures = []
    actual = None
    rows = 0
    if not path.is_file() or path.stat().st_size == 0:
        failures.append(f"missing or zero-byte TSV: {path}")
    else:
        actual = md5(path)
        if expected and actual != expected:
            failures.append(f"checksum mismatch: expected {expected}, got {actual}")
        try:
            data = instance.data if instance is not None and hasattr(instance, "data") else pd.read_csv(path, sep="\t")
            rows = len(data)
            columns = {str(col).lower() for col in data.columns}
            original_columns = {str(col).lower(): col for col in data.columns}
            if not rows:
                failures.append("TSV contains zero rows")
            for required in ("index", "question"):
                if required not in columns:
                    failures.append(f"missing required column: {required}")
                elif not data[original_columns[required]].notna().any():
                    failures.append(f"required column has no values: {required}")
            visual_columns = {"image", "image_path"} & columns
            if not visual_columns:
                failures.append("missing image or image_path column")
            elif not any(data[original_columns[col]].notna().any() for col in visual_columns):
                failures.append("image and image_path columns contain no values")
        except Exception as exc:
            failures.append(f"unreadable TSV: {exc}")
    return ({
        "dataset": name,
        "resolved_class": cls.__name__,
        "source": source,
        "expected_checksum": expected,
        "actual_checksum": actual,
        "row_count": rows,
        "cache_path": str(path.resolve()),
        "status": "ok" if not failures else "failed",
        "errors": failures,
    }, failures)


def prepare_base_models(checkpoints, overrides, download):
    records, failures = [], []
    if not checkpoints:
        return records, failures
    from huggingface_hub import snapshot_download
    from eval_utils import classify_checkpoint, resolve_base, select_checkpoint

    for raw in checkpoints:
        checkpoint = raw.expanduser().resolve()
        kind, reasons = classify_checkpoint(checkpoint)
        if kind not in {"adapter-only", "merged/full"}:
            checkpoint = select_checkpoint(checkpoint) or checkpoint
            kind, reasons = classify_checkpoint(checkpoint)
        if kind != "adapter-only":
            records.append({"checkpoint": str(checkpoint), "kind": kind, "status": "not-required", "details": reasons})
            continue
        try:
            base, revision = resolve_base(checkpoint, overrides)
            local = Path(base).expanduser()
            if local.is_dir():
                resolved = str(local.resolve())
            else:
                resolved = snapshot_download(repo_id=base, revision=revision, local_files_only=not download)
            records.append({"checkpoint": str(checkpoint), "base_model": base, "revision": revision,
                            "resolved_path": resolved, "status": "ok"})
        except Exception as exc:
            message = f"{checkpoint}: base model unavailable: {exc}"
            failures.append(message)
            records.append({"checkpoint": str(checkpoint), "status": "failed", "error": str(exc)})
    return records, failures


def main() -> int:
    args = parser().parse_args()
    args.project_dir = args.project_dir.resolve()
    args.vlmevalkit_dir = args.vlmevalkit_dir.resolve()
    args.lmu_data = args.lmu_data.resolve()
    args.suite_config = args.suite_config.resolve()
    cfg = suite_config(args.suite_config)
    failures = check_environment(args.project_dir)
    if failures:
        print("Environment check failed for the selected Python interpreter:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 2
    try:
        commit = git_commit(args.vlmevalkit_dir)
        print(f"VLMEvalKit commit: {commit}")
        if commit != PINNED_VLMEVAL_COMMIT:
            print(f"WARNING: expected inspected commit {PINNED_VLMEVAL_COMMIT}; continuing without modifying checkout.")
        os.environ["LMUData"] = str(args.lmu_data)
        if not args.check and not args.dry_run:
            args.lmu_data.mkdir(parents=True, exist_ok=True)
        resolved = registry(args.vlmevalkit_dir, cfg["datasets"], args.lmu_data)
    except EvalError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        for name in cfg["datasets"]:
            cls = resolved[name]
            print(f"[dry-run] {name}: {cls.__name__} -> {args.lmu_data / (name + '.tsv')}")
        if args.checkpoints:
            from eval_utils import classify_checkpoint, resolve_base, select_checkpoint
            overrides = parse_overrides(args.base_model_override)
            for raw in args.checkpoints:
                checkpoint = raw.expanduser().resolve()
                checkpoint = select_checkpoint(checkpoint) or checkpoint
                kind, reasons = classify_checkpoint(checkpoint)
                if kind == "adapter-only":
                    try:
                        base, revision = resolve_base(checkpoint, overrides)
                        print(f"[dry-run] adapter base: {checkpoint} -> {base} revision={revision or '<default>'}")
                    except EvalError as exc:
                        failures.append(str(exc))
                elif kind not in {"merged/full"}:
                    failures.append(f"{checkpoint}: {kind}: {', '.join(reasons)}")
        for failure in failures:
            print(f"[FAIL] {failure}", file=sys.stderr)
        return int(bool(failures))

    # --check must not create the cache. --offline writes a manifest but never
    # invokes build_dataset(), whose missing-file path may access the network.
    entries = []
    if not args.check and not args.offline:
        from vlmeval.dataset import build_dataset
    for name in cfg["datasets"]:
        cls = resolved[name]
        instance = None
        construction_error = None
        if not args.check and not args.offline:
            try:
                instance = build_dataset(name)
            except Exception as exc:
                construction_error = str(exc)
        entry, entry_failures = validate_dataset(name, cls, args.lmu_data, instance)
        if construction_error:
            entry["status"] = "failed"
            entry["errors"].insert(0, f"dataset construction failed: {construction_error}")
            entry_failures.insert(0, construction_error)
        entries.append(entry)
        failures.extend(f"{name}: {message}" for message in entry_failures)
        print(f"[{'ok' if not entry_failures else 'FAIL'}] {name}: rows={entry['row_count']} path={entry['cache_path']}")

    model_records, model_failures = prepare_base_models(
        args.checkpoints, parse_overrides(args.base_model_override),
        args.download_models and not args.check and not args.offline)
    failures.extend(model_failures)
    manifest = {
        "schema_version": 1,
        "suite": cfg.get("name", args.suite_config.stem),
        "suite_config": str(args.suite_config),
        "suite_config_sha256": hashlib.sha256(args.suite_config.read_bytes()).hexdigest(),
        "lmu_data": str(args.lmu_data),
        "vlmevalkit_dir": str(args.vlmevalkit_dir),
        "vlmevalkit_commit": commit,
        "timestamp": utc_now(),
        "valid": not failures,
        "datasets": entries,
        "base_models": model_records,
    }
    if not args.check:
        manifest_dir = args.project_dir / "outputs" / "eval" / "manifests"
        target = manifest_dir / f"{cfg.get('name', args.suite_config.stem)}.json"
        atomic_json(target, manifest)
        print(f"Manifest: {target}")
    if failures:
        print("Asset validation failures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
