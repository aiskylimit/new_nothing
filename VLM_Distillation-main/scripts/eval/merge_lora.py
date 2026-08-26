#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter checkpoint into a standalone HF checkpoint for eval."""

import argparse
import json
import sys
from pathlib import Path

from peft import PeftConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.arguments import ModelArguments
from src.model.model import VLMModel
from src.model.processor import load_processor


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
    output_dir.mkdir(parents=True, exist_ok=True)

    peft_config = PeftConfig.from_pretrained(str(adapter_dir))
    base_model = args.base_model or peft_config.base_model_name_or_path
    if not base_model:
        raise SystemExit("Could not infer base model. Pass --base-model explicitly.")

    model_args = ModelArguments(
        model_name=base_model,
        processor_name=base_model,
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
    merged.save_pretrained(str(output_dir), safe_serialization=True, max_shard_size=args.max_shard_size)

    processor = load_processor(
        ModelArguments(
            model_name=base_model,
            processor_name=base_model,
            model_backbone=args.model_backbone,
        )
    )
    processor.save_pretrained(str(output_dir))

    metadata = {
        "adapter": str(adapter_dir),
        "base_model": base_model,
        "model_backbone": args.model_backbone,
    }
    with open(output_dir / "lora_merge_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Merged LoRA adapter {adapter_dir} into {output_dir}")


if __name__ == "__main__":
    main()
