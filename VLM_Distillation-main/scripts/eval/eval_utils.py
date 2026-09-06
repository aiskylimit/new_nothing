#!/usr/bin/env python3
"""Pure helpers shared by evaluation preparation and launch scripts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PINNED_VLMEVAL_COMMIT = "bdb6d429f3f6804eaa6cd899c341486c8a42aed8"
MODEL_WEIGHTS = (
    "model.safetensors", "pytorch_model.bin", "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")
TOKENIZER_MARKERS = ("tokenizer_config.json", "tokenizer.json", "vocab.json", "tokenizer.model")
VISION_PROCESSOR_MARKERS = ("preprocessor_config.json", "processor_config.json", "image_processor_config.json")


class EvalError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvalError(f"Expected a JSON object in {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - upstream dataset identity is MD5
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return clean or "model"


def suite_config(path: Path) -> dict[str, Any]:
    cfg = read_json(path)
    datasets = cfg.get("datasets")
    if not isinstance(datasets, list) or not datasets or not all(isinstance(x, str) and x for x in datasets):
        raise EvalError(f"Suite {path} must contain a non-empty string list named 'datasets'")
    if cfg.get("judge_model") is not None and not isinstance(cfg["judge_model"], str):
        raise EvalError(f"judge_model in {path} must be a string or null")
    return cfg


def git_commit(repo: Path) -> str:
    head = repo / ".git" / "HEAD"
    if not head.exists():
        vendored = repo / ".vendored-commit"
        if not vendored.is_file():
            raise EvalError(f"Neither Git metadata nor .vendored-commit exists in: {repo}")
        commit = vendored.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
            raise EvalError(f"Invalid vendored commit metadata: {vendored}")
        return commit.lower()
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref: "):
        ref = repo / ".git" / text[5:]
        if ref.exists():
            return ref.read_text(encoding="utf-8").strip()
        packed = repo / ".git" / "packed-refs"
        if packed.exists():
            suffix = f" {text[5:]}"
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(suffix):
                    return line.split()[0]
        raise EvalError(f"Cannot resolve git HEAD for {repo}")
    return text


def _has_any(path: Path, names: tuple[str, ...]) -> bool:
    return any((path / name).is_file() and (path / name).stat().st_size > 0 for name in names)


def classify_checkpoint(path: Path) -> tuple[str, list[str]]:
    """Return adapter-only, merged/full, incomplete, or unsupported plus reasons."""
    if not path.is_dir():
        return "unsupported", ["not a directory"]
    adapter_cfg = (path / "adapter_config.json").is_file()
    full_cfg = (path / "config.json").is_file()
    adapter_weights = _has_any(path, ADAPTER_WEIGHTS)
    full_weights = _has_any(path, MODEL_WEIGHTS)
    tokenizer = _has_any(path, TOKENIZER_MARKERS)
    vision_processor = _has_any(path, VISION_PROCESSOR_MARKERS)
    if adapter_cfg:
        missing = []
        if not adapter_weights:
            missing.append("adapter weights")
        if not tokenizer:
            missing.append("tokenizer files")
        if not vision_processor:
            missing.append("vision processor files")
        return ("adapter-only", []) if not missing else ("incomplete", missing)
    if full_cfg:
        missing = []
        if not full_weights:
            missing.append("model weights")
        if not tokenizer:
            missing.append("tokenizer files")
        if not vision_processor:
            missing.append("vision processor files")
        if missing:
            return "incomplete", missing
        return "merged/full", []
    return "unsupported", ["neither adapter_config.json nor config.json is present"]


def _architecture_from_config(cfg: dict[str, Any]) -> str | None:
    model_type = str(cfg.get("model_type", "")).lower().replace("-", "_")
    arch = " ".join(map(str, cfg.get("architectures") or [])).lower().replace("-", "_")
    joined = f"{model_type} {arch}"
    compact = re.sub(r"[^a-z0-9]", "", joined)
    if "fast_vlm" in joined or "fastvlm" in compact:
        return "fast_vlm"
    if "qwen3_vl" in joined or "qwen3vl" in compact:
        return "qwen3_vl"
    if "qwen2_5_vl" in joined or "qwen25vl" in compact:
        return "qwen2_5_vl"
    if "qwen2_vl" in joined or "qwen2vl" in compact:
        return "qwen2_vl"
    return None


def parse_overrides(items: list[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise EvalError(f"Base override must be OLD=NEW, got: {item}")
        old, new = item.split("=", 1)
        if not old or not new:
            raise EvalError(f"Base override must be OLD=NEW, got: {item}")
        result[old] = str(Path(new).expanduser().resolve()) if Path(new).expanduser().exists() else new
    return result


def resolve_base(adapter: Path, overrides: dict[str, str] | None = None) -> tuple[str, str | None]:
    cfg = read_json(adapter / "adapter_config.json")
    embedded = cfg.get("base_model_name_or_path")
    if not embedded:
        raise EvalError(f"No base_model_name_or_path in {adapter / 'adapter_config.json'}")
    resolved = (overrides or {}).get(str(embedded), str(embedded))
    revision = cfg.get("revision")
    return resolved, str(revision) if revision else None


def detect_architecture(path: Path, overrides: dict[str, str] | None = None) -> tuple[str, str | None, str | None]:
    kind, reasons = classify_checkpoint(path)
    if kind in {"unsupported", "incomplete"}:
        raise EvalError(f"Checkpoint {path} is {kind}: {', '.join(reasons)}")
    base = revision = None
    config_path = path / "config.json"
    adapter_hint = None
    if kind == "adapter-only":
        adapter_cfg = read_json(path / "adapter_config.json")
        mapping = adapter_cfg.get("auto_mapping") or {}
        adapter_hint = _architecture_from_config({"architectures": list(mapping.values())})
        base, revision = resolve_base(path, overrides)
        candidate = Path(base).expanduser()
        if candidate.is_dir():
            config_path = candidate / "config.json"
        else:
            try:
                from huggingface_hub import snapshot_download
                cached = snapshot_download(repo_id=base, revision=revision, local_files_only=True)
                config_path = Path(cached) / "config.json"
            except Exception as exc:
                raise EvalError(
                    f"Adapter base model {base!r} revision {revision!r} is not cached; "
                    "prefetch it with prepare_eval_assets.sh --checkpoints ... --download-models, "
                    "or provide --base-model-override OLD=NEW"
                ) from exc
    cfg = read_json(config_path)
    architecture = _architecture_from_config(cfg)
    if architecture is None:
        raise EvalError(f"Unsupported architecture in {config_path}")
    if adapter_hint is not None and adapter_hint != architecture:
        raise EvalError(
            f"Adapter auto_mapping reports {adapter_hint}, but base config reports {architecture}: {path}")
    return architecture, base, revision


def numeric_checkpoint(path: Path) -> int | None:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else None


def select_checkpoint(run_dir: Path) -> Path | None:
    valid = []
    for child in run_dir.iterdir() if run_dir.is_dir() else []:
        step = numeric_checkpoint(child)
        if step is not None and classify_checkpoint(child)[0] in {"adapter-only", "merged/full"}:
            valid.append((step, child))
    if valid:
        return max(valid, key=lambda pair: pair[0])[1]
    return run_dir if classify_checkpoint(run_dir)[0] in {"adapter-only", "merged/full"} else None


def checkpoint_identity(checkpoint: Path, run_name: str, base: str | None = None) -> str:
    weight = next((checkpoint / name for name in ADAPTER_WEIGHTS if (checkpoint / name).is_file()), None)
    material = json.dumps({
        "checkpoint": str(checkpoint.resolve()), "run": run_name, "base": base,
        "adapter": read_json(checkpoint / "adapter_config.json") if (checkpoint / "adapter_config.json").exists() else None,
        "adapter_weights_sha256": sha256_file(weight) if weight else None,
    }, sort_keys=True).encode()
    return hashlib.sha256(material).hexdigest()[:12]


def output_dir(root: Path, suite: str, run_name: str, checkpoint: Path) -> Path:
    step = numeric_checkpoint(checkpoint)
    checkpoint_id = f"checkpoint-{step}" if step is not None else sanitize(checkpoint.name)
    return root / "eval" / sanitize(suite) / sanitize(run_name) / checkpoint_id


def run_identity_name(checkpoint: Path, outputs_root: Path) -> str:
    run_dir = checkpoint.parent if checkpoint.name.startswith("checkpoint-") else checkpoint
    try:
        relative = run_dir.resolve().relative_to(outputs_root.resolve())
        return "__".join(relative.parts)
    except ValueError:
        return run_dir.name


def summary_from_native(context_path: Path) -> dict[str, Any]:
    context = read_json(context_path)
    previous_path = Path(context["run_dir"]) / "summary.json"
    try:
        previous = read_json(previous_path) if previous_path.is_file() else {}
    except EvalError:
        previous = {}
    previous_rows = {row.get("dataset"): row for row in previous.get("benchmarks", [])}
    native_root = Path(context["native_work_dir"])
    candidates = sorted(native_root.rglob("status.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    native = read_json(candidates[0]) if candidates else {}
    statuses = native.get("datasets", {})
    benchmarks = []
    for name in context["datasets"]:
        row = statuses.get(name, {})
        status = row.get("status", "pending")
        pred = row.get("prediction_file")
        if pred and not Path(pred).is_absolute() and candidates:
            pred = str((candidates[0].parent / pred).resolve())
        metrics = row.get("metrics") or {}
        primary_name = row.get("primary_metric")
        if isinstance(primary_name, list):
            primary_value = {key: metrics.get(key) for key in primary_name}
        else:
            primary_value = metrics.get(primary_name) if primary_name else None
        skip = row.get("skip_reason")
        error = row.get("error_message")
        missing_infer = skip in {"Incomplete infer result", "No infer result found"}
        previously_inferred = previous_rows.get(name, {}).get("inference_status") == "complete"
        inferred = bool(pred) and not missing_infer and (
            status == "eval" or (status == "done" and (not error or previously_inferred)))
        scored = bool(metrics) and not error
        benchmarks.append({
            "dataset": name,
            "inference_status": "complete" if inferred else ("failed" if error else "pending"),
            "scoring_status": "complete" if scored else ("skipped" if skip else ("failed" if error else "pending")),
            "primary_metric_name": primary_name,
            "primary_metric_value": primary_value,
            "native_result_path": pred,
            "error": error,
            "skip_reason": skip,
            "native_status": status,
        })
    values = [b["scoring_status"] for b in benchmarks]
    fully_complete = bool(benchmarks) and all(
        b["inference_status"] == "complete" and b["scoring_status"] == "complete"
        for b in benchmarks)
    overall = "complete" if fully_complete else "partial"
    if any(v == "failed" for v in values) or any(b["inference_status"] == "failed" for b in benchmarks):
        overall = "failed"
    result = {key: context[key] for key in (
        "run_name", "checkpoint", "checkpoint_kind", "architecture", "model_name", "suite",
        "created_at", "vlmevalkit_commit", "dataset_manifest", "generated_config", "commands",
        "effective_checkpoint", "merge_metadata", "runtime",
    ) if key in context}
    result.update({"updated_at": utc_now(), "overall_status": overall, "benchmarks": benchmarks})
    return result


def update_summary(context_path: Path) -> Path:
    context = read_json(context_path)
    target = Path(context["run_dir"]) / "summary.json"
    atomic_json(target, summary_from_native(context_path))
    return target
