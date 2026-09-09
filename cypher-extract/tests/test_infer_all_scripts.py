from pathlib import Path

import pytest

from schema_grounding.inference.checkpoints import DEFAULT_METHODS

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
    f"infer_all_{wrapper_family}_{script_suffix}.sh"
    for wrapper_family in MODEL_WRAPPERS
    for script_suffix in SETTINGS
}


def test_infer_all_has_all_family_and_setting_wrappers() -> None:
    actual = {path.name for path in Path("scripts").glob("infer_all_*.sh")}
    assert actual == EXPECTED_WRAPPERS


@pytest.mark.parametrize(
    ("wrapper_family", "model_family", "script_suffix", "setting"),
    [
        (wrapper_family, model_family, script_suffix, setting)
        for wrapper_family, model_family in MODEL_WRAPPERS.items()
        for script_suffix, setting in SETTINGS.items()
    ],
)
def test_infer_all_wrapper_routes_to_shared_runner(
    wrapper_family: str,
    model_family: str,
    script_suffix: str,
    setting: str,
) -> None:
    path = Path("scripts", f"infer_all_{wrapper_family}_{script_suffix}.sh")
    script = path.read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert f'exec bash "${{SCRIPT_DIR}}/infer_all.sh" {model_family} {setting} "$@"' in script


def test_shared_infer_all_runner_maps_checkpoint_and_output_roots() -> None:
    script = Path("scripts/infer_all.sh").read_text(encoding="utf-8")

    for setting in SETTINGS.values():
        assert f"  {setting})" in script
    for checkpoint_root in (
        "results/lora",
        "results/lora_normalized",
        "results/full_finetune",
        "results/full_finetune_normalized",
    ):
        assert f'DEFAULT_CHECKPOINT_ROOT="{checkpoint_root}"' in script
    for output_root in (
        "results/inference/lora",
        "results/inference/lora_normalized",
        "results/inference/full_finetune",
        "results/inference/full_finetune_normalized",
    ):
        assert f'DEFAULT_OUTPUT_ROOT="{output_root}"' in script


def test_full_inference_matrix_replaces_only_the_teacher_method() -> None:
    script = Path("scripts/infer_all.sh").read_text(encoding="utf-8")
    expected = ",".join(("teacher_full", *DEFAULT_METHODS[1:]))

    assert f'FULL_METHODS="{expected}"' in script
    assert 'METHODS="all"' in script
    assert '--methods "${METHODS}"' in script
    assert '"$@"' in script
