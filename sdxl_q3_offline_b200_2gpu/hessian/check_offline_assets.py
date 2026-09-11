#!/usr/bin/env python3
"""Validate every runtime asset before a GPU process is started."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def required_path(name: str, *, directory: bool = False, min_bytes: int = 1) -> Path:
    raw = os.environ.get(name, "")
    if not raw:
        raise RuntimeError(f"{name} is not configured; see env.sh and download.txt")
    path = Path(raw).expanduser().resolve()
    if directory:
        if not path.is_dir():
            raise FileNotFoundError(f"{name} directory is missing: {path}")
    elif not path.is_file() or path.stat().st_size < min_bytes:
        raise FileNotFoundError(f"{name} file is missing/empty: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-wheelhouse", action="store_true")
    args = parser.parse_args()

    model = required_path("MODEL_DIR", directory=True)
    vae = required_path("VAE_DIR", directory=True)
    required = [
        model / "model_index.json",
        model / "unet" / "config.json",
        model / "scheduler" / "scheduler_config.json",
        model / "tokenizer" / "tokenizer_config.json",
        model / "tokenizer_2" / "tokenizer_config.json",
        model / "text_encoder" / "config.json",
        model / "text_encoder" / "model.safetensors",
        model / "text_encoder_2" / "config.json",
        model / "text_encoder_2" / "model.safetensors",
        model / "unet" / "diffusion_pytorch_model.safetensors",
        vae / "config.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete local SDXL snapshot; missing {path}")
    if not any(vae.glob("*.safetensors")) and not any(vae.glob("*.bin")):
        raise FileNotFoundError(f"Incomplete local VAE snapshot; no weights under {vae}")

    data_dir = required_path("DATA_DIR", directory=True)
    manifest_path = required_path("STREAM_MANIFEST")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"Manifest has no files list: {manifest_path}")
    missing = [name for name in files if not (data_dir / Path(name).name).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Offline dataset is incomplete: {len(missing)}/{len(files)} shards missing "
            f"under {data_dir}; first: {preview}"
        )

    for name in (
        "RATIO_PICKAPIC_PROMPTS_SOURCE",
        "RATIO_PARTIPROMPT_SOURCE",
        "RATIO_HPDV2_SOURCE",
    ):
        required_path(name)
    required_path("RATIO_HPSV2_CHECKPOINT", min_bytes=100 * 1024 * 1024)
    required_path("RATIO_HPSV2_BPE", min_bytes=100_000)
    required_path("RATIO_AESTHETIC_CLIP_CHECKPOINT", min_bytes=100 * 1024 * 1024)
    required_path("RATIO_AESTHETIC_MODEL_CHECKPOINT", min_bytes=100_000)
    processor = required_path("RATIO_PICKSCORE_PROCESSOR_DIR", directory=True)
    pickscore = required_path("RATIO_PICKSCORE_MODEL_DIR", directory=True)
    if not (processor / "preprocessor_config.json").is_file():
        raise FileNotFoundError(f"PickScore processor snapshot is incomplete: {processor}")
    if not any(pickscore.glob("*.safetensors")) and not any(pickscore.glob("*.bin")):
        raise FileNotFoundError(f"PickScore model snapshot contains no weights: {pickscore}")
    image_reward = required_path("RATIO_IMAGEREWARD_ROOT", directory=True)
    if not any(image_reward.rglob("*.pt")):
        raise FileNotFoundError(f"ImageReward root contains no .pt checkpoint: {image_reward}")
    if not (image_reward / "med_config.json").is_file():
        raise FileNotFoundError(f"ImageReward med_config.json is missing: {image_reward}")
    bert = required_path("RATIO_IMAGEREWARD_BERT_DIR", directory=True)
    if not (bert / "tokenizer_config.json").is_file():
        raise FileNotFoundError(f"ImageReward BERT tokenizer snapshot is incomplete: {bert}")

    if not args.skip_wheelhouse:
        wheelhouse = required_path("WHEELHOUSE_DIR", directory=True)
        wheels = list(wheelhouse.glob("*.whl"))
        if not wheels:
            raise FileNotFoundError(f"No wheels found in offline wheelhouse: {wheelhouse}")

    result = {
        "status": "ok",
        "model": str(model),
        "vae": str(vae),
        "dataset_manifest": str(manifest_path),
        "dataset_shards": len(files),
        "target_rows": manifest.get("target_rows"),
        "offline_strict": os.environ.get("RATIO_OFFLINE_STRICT") == "1",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
