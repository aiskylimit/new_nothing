from pathlib import Path

import pytest

MODEL_WRAPPERS = {
    "qwen3": "qwen3",
    "llama3": "llama3",
    "qwen2_5_coder": "qwen2.5_coder",
}
SETTINGS = {
    "lora": "lora",
    "lora_normalized": "lora_normalized",
    "full_finetune": "full_finetune",
    "full_finetune_normalized": "full_finetune_normalized",
}
EXPECTED_WRAPPERS = {
    f"train_all_{wrapper_family}_{script_suffix}.sh"
    for wrapper_family in MODEL_WRAPPERS
    for script_suffix in SETTINGS
}


def test_train_all_has_all_family_and_setting_wrappers() -> None:
    actual = {path.name for path in Path("scripts").glob("train_all_*.sh")}
    assert actual == EXPECTED_WRAPPERS


@pytest.mark.parametrize(
    ("wrapper_family", "model_family", "script_suffix", "setting"),
    [
        (wrapper_family, model_family, script_suffix, setting)
        for wrapper_family, model_family in MODEL_WRAPPERS.items()
        for script_suffix, setting in SETTINGS.items()
    ],
)
def test_train_all_wrapper_routes_to_shared_runner(
    wrapper_family: str,
    model_family: str,
    script_suffix: str,
    setting: str,
) -> None:
    path = Path("scripts", f"train_all_{wrapper_family}_{script_suffix}.sh")
    script = path.read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert f'exec bash "${{SCRIPT_DIR}}/train_all.sh" {model_family} {setting} "$@"' in script


def test_shared_train_all_runner_maps_every_config_family_and_teacher_type() -> None:
    script = Path("scripts/train_all.sh").read_text(encoding="utf-8")

    for setting in SETTINGS.values():
        assert f"  {setting})" in script
    assert 'CONFIG_DIRECTORY="${CONFIG_FAMILY}_normalized_loss"' in script
    assert 'CONFIG_DIRECTORY="${CONFIG_FAMILY}_full_finetune"' in script
    assert 'CONFIG_DIRECTORY="${CONFIG_FAMILY}_full_finetune_normalized_loss"' in script
    assert 'CONFIG_DIRECTORY="qwen2.5_coder"' in script

    assert 'TEACHER_KIND="teacher_lora"' in script
    assert 'TEACHER_KIND="teacher_full"' in script
    assert 'TEACHER_OVERRIDE_KEY="ref_model_adapters"' in script
    assert 'TEACHER_OVERRIDE_KEY="ref_model"' in script
    assert 'DEFAULT_RESULTS_SUBDIR="results/normalized_loss"' in script
    assert 'DEFAULT_RESULTS_SUBDIR="results/full_finetune"' in script
    assert 'DEFAULT_RESULTS_SUBDIR="results/full_finetune_normalized_loss"' in script


def test_shared_train_all_runner_preserves_safety_and_output_routing() -> None:
    script = Path("scripts/train_all.sh").read_text(encoding="utf-8")

    assert 'if [[ "${override}" == resume_from_checkpoint=* ]]; then' in script
    assert "one checkpoint cannot be applied to every method" in script
    assert 'fresh_outputs=("${TEACHER_OUTPUT}")' in script
    assert 'for checkpoint in "${output_dir}"/checkpoint-*; do' in script
    assert "Choose a new RESULTS_ROOT" in script
    assert 'method_overrides=("output_dir=${output_dir}")' in script
    assert 'method_overrides+=("${TEACHER_OVERRIDE_KEY}=${TEACHER_OUTPUT}")' in script
    assert '"$@" "${method_overrides[@]}"' in script
    assert "adapter_model.safetensors" in script
    assert "has_full_model_weights" in script
