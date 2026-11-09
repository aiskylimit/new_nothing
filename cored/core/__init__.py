"""Module dùng chung cho Multi-LoRA MoE Council Distillation (spec §4)."""

import os

# CORED always runs offline, including vLLM workers and PEFT metadata lookups.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
if os.environ.get("REBUTTAL_STAGE_DIR"):
    from rebuttal.telemetry import install
    install()

from . import council, data_utils, grad_dropout, losses, rep_loss, segmentation, weighting

__all__ = [
    "council",
    "data_utils",
    "grad_dropout",
    "losses",
    "rep_loss",
    "segmentation",
    "weighting",
]
