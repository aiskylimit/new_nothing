import os
from pathlib import Path

import pytest
import torch
import yaml

from distillation.arguments import DistillationArguments

CONFIG_PATHS = sorted(Path("configs").glob("*/*.yaml"))


def test_all_yaml_keys_are_accepted_by_pinned_llamafactory(monkeypatch) -> None:
    monkeypatch.setenv("DISABLE_VERSION_CHECK", "1")
    try:
        from llamafactory.hparams import (
            DataArguments,
            FinetuningArguments,
            GeneratingArguments,
            ModelArguments,
            TrainingArguments,
        )
        from transformers import HfArgumentParser
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"LlamaFactory integration dependencies are unavailable: {exc}")

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, FinetuningArguments, GeneratingArguments)
    )
    for path in CONFIG_PATHS:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        _, llamafactory_config = DistillationArguments.split_config(config)

        # Avoid initializing unavailable DeepSpeed/CUDA during a key-validation test.
        llamafactory_config.update(
            deepspeed=None,
            bf16=False,
            fp16=False,
            do_train=False,
            do_eval=False,
            predict_with_generate=False,
        )
        parser.parse_dict(llamafactory_config, allow_extra_keys=False)

    os.environ.pop("DISABLE_VERSION_CHECK", None)


def test_bf16_config_sets_llamafactory_compute_dtype(monkeypatch) -> None:
    monkeypatch.setenv("DISABLE_VERSION_CHECK", "1")
    try:
        from llamafactory.hparams import get_train_args
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"LlamaFactory integration dependencies are unavailable: {exc}")

    config = yaml.safe_load(Path("configs/llama3/fkl.yaml").read_text(encoding="utf-8"))
    _, llamafactory_config = DistillationArguments.split_config(config)
    llamafactory_config.update(
        deepspeed=None,
        use_cpu=True,
        do_train=False,
        do_eval=False,
        predict_with_generate=False,
    )
    model_args, _, training_args, _, _ = get_train_args(llamafactory_config)
    assert training_args.bf16 is True
    assert training_args.fp16 is False
    assert model_args.compute_dtype is torch.bfloat16
