import json
from pathlib import Path

import pytest
import yaml

EXPECTED_CONFIGS = {"ds_config_bf16.json"}
BASELINE_CONFIG_PATHS = sorted(Path("configs").glob("*/*.yaml"))


def test_only_supported_bf16_deepspeed_preset_is_shipped() -> None:
    actual = {path.name for path in Path("configs/deepspeed").glob("*.json")}
    assert actual == EXPECTED_CONFIGS


def test_deepspeed_batch_fields_are_owned_by_trainer() -> None:
    config = json.loads(Path("configs/deepspeed/ds_config_bf16.json").read_text(encoding="utf-8"))
    assert config["train_batch_size"] == "auto"
    assert config["train_micro_batch_size_per_gpu"] == "auto"
    assert config["gradient_accumulation_steps"] == "auto"
    assert config["gradient_clipping"] == "auto"
    assert config["zero_optimization"]["stage"] == 1


def test_deepspeed_preset_uses_bf16() -> None:
    config = json.loads(Path("configs/deepspeed/ds_config_bf16.json").read_text(encoding="utf-8"))
    assert config["bf16"]["enabled"] is True
    assert config["fp16"]["enabled"] is False


@pytest.mark.parametrize("path", BASELINE_CONFIG_PATHS)
def test_baseline_configs_reference_existing_deepspeed_config(path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert Path(config["deepspeed"]).is_file()


@pytest.mark.parametrize("path", BASELINE_CONFIG_PATHS)
def test_baseline_configs_use_bf16_only(path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["bf16"] is True
    assert config.get("fp16", False) is False
    assert config["deepspeed"] == "configs/deepspeed/ds_config_bf16.json"
