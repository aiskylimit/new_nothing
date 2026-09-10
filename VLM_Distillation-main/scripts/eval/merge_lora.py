#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter checkpoint into a standalone HF checkpoint for eval."""

import argparse
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from peft import PeftConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.arguments import ModelArguments
from src.model.model import VLMModel
from src.model.processor import load_processor, save_processor
from eval_utils import _architecture_from_config, atomic_json, read_json


def _set_if_present(config, name, value):
    if config is not None and hasattr(config, name):
        setattr(config, name, value)


def _restore_generation_config(config):
    """Undo training-only output settings before saving the eval checkpoint."""
    for sub_config in [
        config,
        getattr(config, "text_config", None),
        getattr(config, "vision_config", None),
        getattr(config, "vision_config_2", None),
        getattr(config, "audio_config", None),
    ]:
        _set_if_present(sub_config, "use_cache", True)
        _set_if_present(sub_config, "output_hidden_states", False)
        _set_if_present(sub_config, "output_attentions", False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="Path to the PEFT adapter checkpoint directory.")
    parser.add_argument("--output", required=True, help="Directory for the merged HF checkpoint.")
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model override. Defaults to base_model_name_or_path from adapter_config.json.",
    )
    parser.add_argument("--model-backbone", default=None, help="Optional repo backbone override.")
    parser.add_argument("--expected-architecture", default=None, help="Architecture validated before merging.")
    parser.add_argument("--identity", default=None, help="Deterministic merge identity recorded in metadata.")
    parser.add_argument("--revision", default=None, help="Exact Hugging Face base-model revision.")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to from_pretrained. Use 'none' to disable.",
    )
    parser.add_argument("--max-shard-size", default="4GB", help="Shard size passed to save_pretrained.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    adapter_dir = Path(args.adapter).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not (adapter_dir / "adapter_config.json").is_file():
        raise SystemExit(f"Missing adapter_config.json in {adapter_dir}")

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is not empty: {output_dir}. Pass --overwrite to reuse it.")

    peft_config = PeftConfig.from_pretrained(str(adapter_dir))
    base_model = args.base_model or peft_config.base_model_name_or_path
    if not base_model:
        raise SystemExit("Could not infer base model. Pass --base-model explicitly.")

    base_path = Path(base_model).expanduser()
    if not base_path.is_dir():
        from huggingface_hub import snapshot_download
        try:
            base_model = snapshot_download(repo_id=base_model, revision=args.revision, local_files_only=True)
        except Exception as exc:
            raise SystemExit(
                f"Base model {base_model!r} revision {args.revision!r} is not cached. "
                "Run prepare_eval_assets.sh with --download-models explicitly first. "
                f"Details: {exc}") from exc

    base_cfg = read_json(Path(base_model) / "config.json")
    architecture = _architecture_from_config(base_cfg)
    if architecture is None:
        raise SystemExit(f"Unsupported base-model architecture in {base_model}/config.json")
    if args.expected_architecture and architecture != args.expected_architecture:
        raise SystemExit(f"Architecture mismatch: expected {args.expected_architecture}, resolved {architecture}")

    model_args = ModelArguments(
        model_name=base_model,
        processor_name=str(adapter_dir),
        model_backbone=args.model_backbone,
        checkpoint_path=str(adapter_dir),
        lora=True,
    )

    load_kwargs = {
        # Force eager regardless of what's in the model config. transformers
        # v5 sometimes ignores config._attn_implementation for Qwen2-VL during
        # from_pretrained and tries flash_attn_2 — fails on hosts without the
        # flash_attn package. eager works everywhere and matches training.
        "attn_implementation": "eager",
    }
    if args.device_map and args.device_map != "none":
        load_kwargs["device_map"] = args.device_map

    model = VLMModel.load(model_args, is_trainable=False, **load_kwargs)
    if not hasattr(model.encoder, "merge_and_unload"):
        raise SystemExit(f"Checkpoint at {adapter_dir} did not load as a mergeable PEFT model.")

    merged = model.encoder.merge_and_unload()
    _restore_generation_config(merged.config)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.incomplete-{os.getpid()}"
    if staging.exists():
        raise SystemExit(f"Refusing to reuse existing incomplete merge directory: {staging}")
    staging.mkdir()
    merged.save_pretrained(str(staging), safe_serialization=True, max_shard_size=args.max_shard_size)

    try:
        processor = load_processor(ModelArguments(
            model_name=str(adapter_dir), processor_name=str(adapter_dir), model_backbone=architecture))
    except Exception as exc:
        print(f"Adapter processor reload failed ({exc}); loading base processor before overlaying saved files.")
        processor = load_processor(ModelArguments(
            model_name=base_model, processor_name=base_model, model_backbone=architecture))
    save_processor(processor, str(staging), architecture)
    preserved_prefixes = ("tokenizer", "processor", "preprocessor", "video_preprocessor", "chat_template")
    preserved_names = {"vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json",
                       "sentencepiece.bpe.model", "spiece.model"}
    for source in adapter_dir.iterdir():
        if source.is_file() and (source.name.startswith(preserved_prefixes) or source.name in preserved_names):
            shutil.copy2(source, staging / source.name)
    if architecture == "fast_vlm":
        # An older adapter may carry KamilaMila's inconsistent tokenizer files.
        # Re-saving the normalized processor last guarantees Apple-compatible
        # EOS/PAD metadata in the standalone merged checkpoint.
        save_processor(processor, str(staging), architecture)

    # CPU-side reload checks catch missing config/processor files without allocating model weights.
    from transformers import AutoConfig, AutoProcessor
    AutoConfig.from_pretrained(str(staging), trust_remote_code=True)
    try:
        AutoProcessor.from_pretrained(str(staging), trust_remote_code=True)
    except Exception:
        load_processor(ModelArguments(model_name=str(staging), processor_name=str(staging),
                                      checkpoint_path=str(staging), model_backbone=architecture))

    metadata = {
        "adapter": str(adapter_dir),
        "base_model": args.base_model or peft_config.base_model_name_or_path,
        "revision": args.revision,
        "model_backbone": args.model_backbone,
        "architecture": architecture,
        "identity": args.identity,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(staging / "lora_merge_metadata.json", metadata)
    if output_dir.exists():
        if any(output_dir.iterdir()):
            if not args.overwrite:
                raise SystemExit(f"Output became non-empty during merge; preserving staging at {staging}")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = output_dir.parent / f"{output_dir.name}.backup-{stamp}"
            if backup.exists():
                raise SystemExit(f"Backup target already exists; preserving staging at {staging}: {backup}")
            output_dir.rename(backup)
            print(f"Preserved previous output at {backup}")
        else:
            output_dir.rmdir()
    staging.rename(output_dir)

    print(f"Merged LoRA adapter {adapter_dir} into {output_dir}")


if __name__ == "__main__":
    main()
