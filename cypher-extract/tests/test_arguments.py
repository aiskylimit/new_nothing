from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from distillation.arguments import DistillationArguments
from distillation.cli import _validate_deepspeed_platform, _validate_required_placeholders
from distillation.reference import create_reference_model_at_revision

MODEL_FAMILIES = ("llama3", "qwen3", "qwen2.5_coder")
NORMALIZED_CONFIG_FAMILIES = {
    "llama3_normalized_loss": "llama3",
    "qwen2.5_normalized_loss": "qwen2.5_coder",
    "qwen3_normalized_loss": "qwen3",
}
FULL_CONFIG_FAMILIES = {
    "llama3_full_finetune": "llama3",
    "qwen2.5_full_finetune": "qwen2.5_coder",
    "qwen3_full_finetune": "qwen3",
}
FULL_NORMALIZED_CONFIG_FAMILIES = {
    "llama3_full_finetune_normalized_loss": "llama3",
    "qwen2.5_full_finetune_normalized_loss": "qwen2.5_coder",
    "qwen3_full_finetune_normalized_loss": "qwen3",
}
CONFIG_PATHS = sorted(
    path
    for family in MODEL_FAMILIES
    for path in (Path("configs") / family).glob("*.yaml")
)
ALL_TRAIN_CONFIG_PATHS = sorted(
    path
    for directory in (
        "distillation",
        *MODEL_FAMILIES,
        *NORMALIZED_CONFIG_FAMILIES,
        *FULL_CONFIG_FAMILIES,
        *FULL_NORMALIZED_CONFIG_FAMILIES,
    )
    for path in (Path("configs") / directory).glob("*.yaml")
)
REDUNDANT_RUNTIME_DEFAULTS = {
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "fp16",
    "gradient_checkpointing",
    "logging_strategy",
}
BASELINE_CONFIG_NAMES = {
    "csd.yaml",
    "amid.yaml",
    "distillm_adaptive_sfkl.yaml",
    "distillm_adaptive_srkl.yaml",
    "fdd_sfkl.yaml",
    "fdd_srkl.yaml",
    "fkl.yaml",
    "hpd.yaml",
    "rkl.yaml",
    "sfkl.yaml",
    "srkl.yaml",
    "sft.yaml",
}
MODEL_REVISIONS = {
    "qwen3": {
        "student": "c1899de289a04d12100db370d81485cdf75e47ca",
        "teacher": "cdbee75f17c01a7cc42f958dc650907174af0554",
    },
    "llama3": {
        "student": "9213176726f574b556790deb65791e0c5aa438b6",
        "teacher": "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
    },
    "qwen2.5_coder": {
        "student": "488639f1ff808d1d3d0ba301aef8c11461451ec5",
        "teacher": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
    },
}
PINNED_MODEL_REVISIONS = {
    "Qwen/Qwen3-0.6B": MODEL_REVISIONS["qwen3"]["student"],
    "Qwen/Qwen3-4B-Instruct-2507": MODEL_REVISIONS["qwen3"]["teacher"],
    "meta-llama/Llama-3.2-1B-Instruct": MODEL_REVISIONS["llama3"]["student"],
    "meta-llama/Meta-Llama-3-8B-Instruct": MODEL_REVISIONS["llama3"]["teacher"],
    "Qwen/Qwen2.5-Coder-3B-Instruct": MODEL_REVISIONS["qwen2.5_coder"]["student"],
    "Qwen/Qwen2.5-Coder-7B-Instruct": MODEL_REVISIONS["qwen2.5_coder"]["teacher"],
}


def test_yaml_configs_do_not_have_utf8_bom() -> None:
    bom = b"\xef\xbb\xbf"
    offenders = [
        str(path)
        for path in sorted(Path("configs").rglob("*.yaml"))
        if path.read_bytes().startswith(bom)
    ]
    assert offenders == []


@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_each_model_family_has_all_baseline_configs(family: str) -> None:
    config_paths = list((Path("configs") / family).glob("*.yaml"))
    actual = {path.name for path in config_paths}
    assert actual == BASELINE_CONFIG_NAMES


@pytest.mark.parametrize(("normalized_directory", "base_family"), NORMALIZED_CONFIG_FAMILIES.items())
def test_normalized_configs_match_base_presets(normalized_directory: str, base_family: str) -> None:
    base_paths = {path.name: path for path in (Path("configs") / base_family).glob("*.yaml")}
    normalized_paths = {
        path.name: path for path in (Path("configs") / normalized_directory).glob("*.yaml")
    }
    assert set(normalized_paths) == BASELINE_CONFIG_NAMES

    for name, normalized_path in normalized_paths.items():
        base_config = yaml.safe_load(base_paths[name].read_text(encoding="utf-8"))
        normalized_config = yaml.safe_load(normalized_path.read_text(encoding="utf-8"))
        distillation_args, remaining_config = DistillationArguments.split_config(normalized_config)
        assert distillation_args.uses_task_normalized_loss is True
        assert distillation_args.selector_loss_weight == 0.5
        assert "selector_loss_weight" not in remaining_config
        assert normalized_config.pop("selector_loss_weight") == 0.5
        assert normalized_config.pop("output_dir") == (
            f"results/normalized_loss/{base_family}/{normalized_path.stem}"
        )
        base_config.pop("output_dir")
        if "ref_model_adapters" in base_config:
            assert normalized_config.pop("ref_model_adapters") == (
                f"results/normalized_loss/{base_family}/teacher_lora"
            )
            base_config.pop("ref_model_adapters")
        assert normalized_config == base_config


@pytest.mark.parametrize(("full_directory", "base_family"), FULL_CONFIG_FAMILIES.items())
def test_full_finetune_configs_match_base_presets(full_directory: str, base_family: str) -> None:
    base_paths = {path.name: path for path in (Path("configs") / base_family).glob("*.yaml")}
    full_paths = {path.name: path for path in (Path("configs") / full_directory).glob("*.yaml")}
    assert set(full_paths) == BASELINE_CONFIG_NAMES

    for name, full_path in full_paths.items():
        base_config = yaml.safe_load(base_paths[name].read_text(encoding="utf-8"))
        full_config = yaml.safe_load(full_path.read_text(encoding="utf-8"))
        distillation_args, _ = DistillationArguments.split_config(full_config)
        assert distillation_args.uses_task_normalized_loss is False
        assert full_config.pop("output_dir") == f"results/full_finetune/{base_family}/{full_path.stem}"
        base_config.pop("output_dir")

        assert full_config.pop("finetuning_type") == "full"
        base_config.pop("finetuning_type")
        for key in ("lora_target", "lora_rank", "lora_alpha", "lora_dropout"):
            base_config.pop(key, None)
            assert key not in full_config

        assert full_config.pop("learning_rate") == 2e-5
        base_config.pop("learning_rate")
        if "ref_model_adapters" in base_config:
            assert full_config.pop("ref_model") == f"results/full_finetune/{base_family}/teacher_full"
            base_config.pop("ref_model")
            base_config.pop("ref_model_revision")
            base_config.pop("ref_model_adapters")
            assert "ref_model_revision" not in full_config
            assert "ref_model_adapters" not in full_config
        assert full_config == base_config


@pytest.mark.parametrize(
    ("normalized_directory", "base_family"),
    FULL_NORMALIZED_CONFIG_FAMILIES.items(),
)
def test_full_normalized_configs_match_full_finetune_presets(
    normalized_directory: str,
    base_family: str,
) -> None:
    full_directory = normalized_directory.removesuffix("_normalized_loss")
    full_paths = {path.name: path for path in (Path("configs") / full_directory).glob("*.yaml")}
    normalized_paths = {
        path.name: path for path in (Path("configs") / normalized_directory).glob("*.yaml")
    }
    assert set(normalized_paths) == BASELINE_CONFIG_NAMES

    for name, normalized_path in normalized_paths.items():
        full_config = yaml.safe_load(full_paths[name].read_text(encoding="utf-8"))
        normalized_config = yaml.safe_load(normalized_path.read_text(encoding="utf-8"))
        distillation_args, remaining_config = DistillationArguments.split_config(normalized_config)
        assert distillation_args.uses_task_normalized_loss is True
        assert distillation_args.selector_loss_weight == 0.5
        assert "selector_loss_weight" not in remaining_config
        assert normalized_config.pop("selector_loss_weight") == 0.5
        assert normalized_config.pop("output_dir") == (
            f"results/full_finetune_normalized_loss/{base_family}/{normalized_path.stem}"
        )
        full_config.pop("output_dir")
        if "ref_model" in full_config:
            assert normalized_config.pop("ref_model") == (
                f"results/full_finetune_normalized_loss/{base_family}/teacher_full"
            )
            full_config.pop("ref_model")
        assert normalized_config == full_config


@pytest.mark.parametrize("config_path", ALL_TRAIN_CONFIG_PATHS)
def test_train_configs_omit_redundant_runtime_defaults(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert REDUNDANT_RUNTIME_DEFAULTS.isdisjoint(config)


@pytest.mark.parametrize("config_path", ALL_TRAIN_CONFIG_PATHS)
def test_all_train_configs_use_the_same_effective_batch_settings(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["per_device_train_batch_size"] == 2
    assert config["gradient_accumulation_steps"] == 8


@pytest.mark.parametrize("config_path", ALL_TRAIN_CONFIG_PATHS)
def test_all_train_configs_keep_all_checkpoints(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["save_total_limit"] is None


@pytest.mark.parametrize("config_path", ALL_TRAIN_CONFIG_PATHS)
def test_all_train_configs_log_and_run_generative_eval_each_epoch(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["logging_steps"] == 20
    assert config["do_eval"] is True
    assert config["predict_with_generate"] is True
    assert config["save_strategy"] == "epoch"
    assert config["eval_strategy"] == "epoch"


@pytest.mark.parametrize("config_path", ALL_TRAIN_CONFIG_PATHS)
def test_all_remote_base_models_are_pinned_to_immutable_commits(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["model_revision"] == PINNED_MODEL_REVISIONS[config["model_name_or_path"]]
    if "ref_model" in config:
        if config["ref_model"] in PINNED_MODEL_REVISIONS:
            assert config["ref_model_revision"] == PINNED_MODEL_REVISIONS[config["ref_model"]]
        else:
            assert config["ref_model"].startswith(
                ("results/full_finetune/", "results/full_finetune_normalized_loss/")
            )
            assert "ref_model_revision" not in config


@pytest.mark.parametrize(
    ("teacher_path", "kd_path"),
    [
        ("configs/distillation/teacher_lora_qwen3.yaml", "configs/qwen3/fkl.yaml"),
        ("configs/distillation/teacher_lora_llama3.yaml", "configs/llama3/fkl.yaml"),
        ("configs/distillation/teacher_lora_qwen2.5_coder.yaml", "configs/qwen2.5_coder/fkl.yaml"),
        (
            "configs/distillation/teacher_lora_qwen3_normalized_loss.yaml",
            "configs/qwen3_normalized_loss/fkl.yaml",
        ),
        (
            "configs/distillation/teacher_lora_llama3_normalized_loss.yaml",
            "configs/llama3_normalized_loss/fkl.yaml",
        ),
        (
            "configs/distillation/teacher_lora_qwen2.5_coder_normalized_loss.yaml",
            "configs/qwen2.5_normalized_loss/fkl.yaml",
        ),
    ],
)
def test_teacher_lora_output_is_wired_into_family_kd_configs(teacher_path: str, kd_path: str) -> None:
    teacher = yaml.safe_load(Path(teacher_path).read_text(encoding="utf-8"))
    kd = yaml.safe_load(Path(kd_path).read_text(encoding="utf-8"))

    assert teacher["distill_method"] == "sft"
    assert teacher["kd_ratio"] == 0.0
    assert teacher["finetuning_type"] == "lora"
    assert "ref_model" not in teacher
    assert "ref_model_adapters" not in teacher
    assert teacher["model_name_or_path"] == kd["ref_model"]
    assert teacher["model_revision"] == kd["ref_model_revision"]
    assert teacher["output_dir"] == kd["ref_model_adapters"]
    assert teacher["dataset"] == kd["dataset"]
    assert teacher["eval_dataset"] == kd["eval_dataset"]
    assert teacher["dataset_dir"] == kd["dataset_dir"]
    assert teacher["learning_rate"] == 1e-5
    assert teacher["lr_scheduler_type"] == "cosine"
    assert teacher["warmup_ratio"] == 0.1
    assert "warmup_steps" not in teacher
    assert "lr_scheduler_kwargs" not in teacher


@pytest.mark.parametrize(
    ("teacher_path", "kd_path", "selector_weight"),
    [
        ("configs/distillation/teacher_full_qwen3.yaml", "configs/qwen3_full_finetune/fkl.yaml", None),
        ("configs/distillation/teacher_full_llama3.yaml", "configs/llama3_full_finetune/fkl.yaml", None),
        (
            "configs/distillation/teacher_full_qwen2.5_coder.yaml",
            "configs/qwen2.5_full_finetune/fkl.yaml",
            None,
        ),
        (
            "configs/distillation/teacher_full_qwen3_normalized_loss.yaml",
            "configs/qwen3_full_finetune_normalized_loss/fkl.yaml",
            0.5,
        ),
        (
            "configs/distillation/teacher_full_llama3_normalized_loss.yaml",
            "configs/llama3_full_finetune_normalized_loss/fkl.yaml",
            0.5,
        ),
        (
            "configs/distillation/teacher_full_qwen2.5_coder_normalized_loss.yaml",
            "configs/qwen2.5_full_finetune_normalized_loss/fkl.yaml",
            0.5,
        ),
    ],
)
def test_teacher_full_output_is_wired_into_full_finetune_configs(
    teacher_path: str,
    kd_path: str,
    selector_weight: float | None,
) -> None:
    teacher = yaml.safe_load(Path(teacher_path).read_text(encoding="utf-8"))
    kd = yaml.safe_load(Path(kd_path).read_text(encoding="utf-8"))

    assert teacher["distill_method"] == "sft"
    assert teacher["kd_ratio"] == 0.0
    assert teacher["finetuning_type"] == "full"
    assert not any(key.startswith("lora_") for key in teacher)
    assert "ref_model" not in teacher
    assert "ref_model_adapters" not in teacher
    assert teacher["output_dir"] == kd["ref_model"]
    assert "ref_model_revision" not in kd
    assert "ref_model_adapters" not in kd
    assert teacher["dataset"] == kd["dataset"]
    assert teacher["eval_dataset"] == kd["eval_dataset"]
    assert teacher["dataset_dir"] == kd["dataset_dir"]
    if selector_weight is None:
        assert "selector_loss_weight" not in teacher
        assert "selector_loss_weight" not in kd
    else:
        assert teacher["selector_loss_weight"] == selector_weight
        assert kd["selector_loss_weight"] == selector_weight


@pytest.mark.parametrize(
    "config_path",
    [
        Path("configs/distillation/student_sft.yaml"),
        Path("configs/llama3/sft.yaml"),
        Path("configs/qwen3/sft.yaml"),
        Path("configs/qwen2.5_coder/sft.yaml"),
    ],
)
def test_student_sft_configs_use_full_finetuning(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["finetuning_type"] == "full"
    assert not any(key.startswith("lora_") for key in config)


@pytest.mark.parametrize(
    ("family", "student_layers", "teacher_layers"),
    [
        ("llama3", 16, 32),
        ("qwen3", 28, 36),
        ("qwen2.5_coder", 36, 28),
    ],
)
def test_fdd_configs_map_middle_and_final_hidden_states(
    family: str,
    student_layers: int,
    teacher_layers: int,
) -> None:
    for config_path in (Path("configs") / family).glob("fdd_*.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["student_layer_mapping"] == [student_layers // 2, student_layers]
        assert config["teacher_layer_mapping"] == [teacher_layers // 2, teacher_layers]


@pytest.mark.parametrize("config_path", sorted(Path("configs/qwen3").glob("*.yaml")))
def test_qwen_configs_follow_template_defaults(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["model_name_or_path"] == "Qwen/Qwen3-0.6B"
    assert config["model_revision"] == MODEL_REVISIONS["qwen3"]["student"]
    if config["distill_method"] != "sft":
        assert config["ref_model"] == "Qwen/Qwen3-4B-Instruct-2507"
        assert config["ref_model_revision"] == MODEL_REVISIONS["qwen3"]["teacher"]
    assert config["template"] == "qwen3_nothink"
    assert config["cutoff_len"] == 892
    assert config["per_device_train_batch_size"] == 2
    assert config["per_device_eval_batch_size"] == 16
    assert config["gradient_accumulation_steps"] == 8
    if config["distill_method"].startswith("fdd_"):
        assert config["student_layer_mapping"] == [14, 28]
        assert config["teacher_layer_mapping"] == [18, 36]


@pytest.mark.parametrize("config_path", sorted(Path("configs/qwen2.5_coder").glob("*.yaml")))
def test_qwen2_5_coder_configs_follow_architecture_defaults(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["model_name_or_path"] == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert config["model_revision"] == MODEL_REVISIONS["qwen2.5_coder"]["student"]
    if config["distill_method"] != "sft":
        assert config["ref_model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
        assert config["ref_model_revision"] == MODEL_REVISIONS["qwen2.5_coder"]["teacher"]
    assert config["template"] == "qwen"
    assert config["cutoff_len"] == 892
    assert config["per_device_train_batch_size"] == 2
    assert config["per_device_eval_batch_size"] == 8
    assert config["gradient_accumulation_steps"] == 8
    if config["distill_method"].startswith("fdd_"):
        assert config["student_layer_mapping"] == [18, 36]
        assert config["teacher_layer_mapping"] == [14, 28]


def test_adaptive_method_requires_student_generation() -> None:
    with pytest.raises(ValueError, match="requires student_gen=true"):
        DistillationArguments(distill_method="adaptive_srkl")


def test_static_method_rejects_student_generation() -> None:
    with pytest.raises(ValueError, match="only valid for adaptive"):
        DistillationArguments(distill_method="rkl", student_gen=True)


def test_fdd_requires_two_matching_layer_lists() -> None:
    with pytest.raises(ValueError, match="at least two"):
        DistillationArguments(
            distill_method="fdd_srkl",
            fdd_weight=0.2,
            student_layer_mapping=[7],
            teacher_layer_mapping=[15],
        )


def test_fdd_sfkl_uses_sfkl_token_divergence() -> None:
    args = DistillationArguments(
        distill_method="fdd_sfkl",
        fdd_weight=0.2,
        student_layer_mapping=[7, 15],
        teacher_layer_mapping=[15, 31],
    )
    assert args.uses_fdd is True
    assert args.base_method == "sfkl"


def test_method_alias_is_canonicalized() -> None:
    args = DistillationArguments(distill_method="adaptive-srkl", student_gen=True)
    assert args.distill_method == "adaptive_srkl"
    assert args.base_method == "srkl"


def test_amid_method_is_supported() -> None:
    args = DistillationArguments(distill_method="amid", amid_div_name="ab", amid_div_order="rq")
    assert args.base_method == "amid"


def test_hpd_method_is_supported() -> None:
    args = DistillationArguments(distill_method="hpd", hpd_sample_in_fp32=False)
    assert args.base_method == "hpd"
    assert args.hpd_sample_in_fp32 is False


def test_explicit_sft_mode_is_teacher_free() -> None:
    args = DistillationArguments(distill_method="sft", kd_ratio=0.0)
    assert args.is_sft is True
    assert args.uses_kd is False
    assert args.base_method == "sft"


def test_reference_model_uses_its_own_revision_and_restores_student_revision() -> None:
    model_args = SimpleNamespace(model_revision="student-commit")
    observed = {}

    def factory(received_model_args, finetuning_args):
        observed["revision"] = received_model_args.model_revision
        observed["finetuning_args"] = finetuning_args
        return "teacher"

    finetuning_args = object()
    result = create_reference_model_at_revision(model_args, finetuning_args, "teacher-commit", factory)

    assert result == "teacher"
    assert observed == {"revision": "teacher-commit", "finetuning_args": finetuning_args}
    assert model_args.model_revision == "student-commit"


def test_local_reference_model_does_not_inherit_student_revision(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher_full"
    teacher.mkdir()
    model_args = SimpleNamespace(model_revision="student-commit")
    finetuning_args = SimpleNamespace(ref_model=str(teacher))
    observed = {}

    def factory(received_model_args, _finetuning_args):
        observed["revision"] = received_model_args.model_revision
        return "teacher"

    assert create_reference_model_at_revision(model_args, finetuning_args, None, factory) == "teacher"
    assert observed["revision"] is None
    assert model_args.model_revision == "student-commit"


def test_zero_kd_ratio_ignores_stale_adaptive_options() -> None:
    args = DistillationArguments(
        distill_method="adaptive_srkl",
        kd_ratio=0.0,
        student_gen=True,
        fdd_weight=0.2,
        student_layer_mapping=[7, 15],
        teacher_layer_mapping=[15, 31],
    )
    assert args.is_sft is True
    assert args.is_adaptive is False
    assert args.uses_fdd is False


def test_explicit_sft_rejects_nonzero_kd_ratio() -> None:
    with pytest.raises(ValueError, match="distill_method=sft requires kd_ratio=0"):
        DistillationArguments(distill_method="sft", kd_ratio=0.5)


def test_task_normalized_loss_is_opt_in() -> None:
    default_args = DistillationArguments()
    normalized_args = DistillationArguments(selector_loss_weight=0.5)
    assert default_args.uses_task_normalized_loss is False
    assert normalized_args.uses_task_normalized_loss is True


def test_eval_prediction_artifacts_are_enabled_by_default() -> None:
    assert DistillationArguments().save_eval_predictions is True


@pytest.mark.parametrize("weight", [0.0, 1.0, -0.1, 1.1])
def test_selector_loss_weight_must_keep_both_tasks_active(weight: float) -> None:
    with pytest.raises(ValueError, match="selector_loss_weight"):
        DistillationArguments(selector_loss_weight=weight)


@pytest.mark.parametrize("field, value", [("amid_div_name", "js"), ("amid_div_order", "pp")])
def test_amid_arguments_validate_choices(field: str, value: str) -> None:
    with pytest.raises(ValueError, match="amid"):
        DistillationArguments(**{field: value})


@pytest.mark.parametrize("config_path", CONFIG_PATHS)
def test_baseline_config_student_generation_matrix(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    distillation_args, _ = DistillationArguments.split_config(config)
    assert distillation_args.uses_task_normalized_loss is False
    if distillation_args.uses_kd:
        expected_adapter = {
            "qwen3": "results/qwen3/teacher_lora",
            "llama3": "results/llama3/teacher_lora",
            "qwen2.5_coder": "results/qwen2.5_coder/teacher_lora",
        }[config_path.parent.name]
        assert config["ref_model_adapters"] == expected_adapter
        assert distillation_args.ref_model_revision == MODEL_REVISIONS[config_path.parent.name]["teacher"]
    else:
        assert "ref_model_adapters" not in config
    assert config["dataset"] == "cypher_prepared_train"
    assert config["eval_dataset"] == "cypher_prepared_eval"
    assert config["dataset_dir"] == "data/llamafactory"
    expected_template = {
        "qwen3": "qwen3_nothink",
        "llama3": "llama3",
        "qwen2.5_coder": "qwen",
    }[config_path.parent.name]
    assert config["template"] == expected_template
    assert config["packing"] is False
    assert config["ignore_pad_token_for_loss"] is True
    assert config["train_on_prompt"] is False
    assert "mask_history" not in config
    assert config["predict_with_generate"] is True
    assert config["dataloader_num_workers"] == 1
    assert config["dataloader_drop_last"] is False
    assert config["optim"] == "adamw_torch"
    assert config["lr_scheduler_type"] == "cosine_with_min_lr"
    assert config["lr_scheduler_kwargs"] == {"min_lr_rate": 0.001}
    if distillation_args.is_adaptive:
        assert config["student_gen"] is True
        expected_context_length = {"qwen3": 797, "llama3": 810, "qwen2.5_coder": 797}[config_path.parent.name]
        assert config["rollout_context_length"] == expected_context_length
        assert config["repetition_penalty"] == 1.0
    else:
        adaptive_only = {
            "student_gen",
            "gen_top_p",
            "init_threshold",
            "loss_eps",
            "capacity",
            "replay_ratio",
        }
        assert adaptive_only.isdisjoint(config)
        assert "rollout_context_length" not in config


def test_unresolved_required_placeholders_are_rejected() -> None:
    with pytest.raises(ValueError, match="ref_model, ref_model_adapters, dataset, eval_dataset"):
        _validate_required_placeholders(
            {
                "ref_model": "REPLACE_WITH_TEACHER_MODEL",
                "ref_model_adapters": "REPLACE_WITH_TEACHER_ADAPTER",
                "dataset": "REPLACE_WITH_TRAIN_DATASET",
                "eval_dataset": "REPLACE_WITH_EVAL_DATASET",
            }
        )


def test_overridden_required_placeholders_are_accepted() -> None:
    _validate_required_placeholders(
        {
            "ref_model": "Qwen/Qwen3-4B-Instruct-2507",
            "ref_model_adapters": "hf://owner/repo/adapter",
            "dataset": "my_train",
            "eval_dataset": "my_eval",
        }
    )


def test_sft_does_not_require_teacher_placeholder() -> None:
    _validate_required_placeholders(
        {
            "distill_method": "sft",
            "kd_ratio": 0.0,
            "ref_model": "REPLACE_WITH_TEACHER_MODEL",
            "ref_model_adapters": "REPLACE_WITH_TEACHER_ADAPTER",
            "dataset": "my_train",
            "eval_dataset": "my_val",
        }
    )


def test_deepspeed_platform_validation_allows_linux(monkeypatch) -> None:
    monkeypatch.setattr("distillation.cli.sys.platform", "linux")
    _validate_deepspeed_platform("configs/deepspeed/ds_config_bf16.json")


def test_deepspeed_platform_validation_rejects_windows(monkeypatch) -> None:
    monkeypatch.setattr("distillation.cli.sys.platform", "win32")
    with pytest.raises(RuntimeError, match="Linux/WSL2"):
        _validate_deepspeed_platform("configs/deepspeed/ds_config_bf16.json")
    _validate_deepspeed_platform(None)


def test_runtime_capacity_validation() -> None:
    args = DistillationArguments(
        distill_method="adaptive_sfkl",
        student_gen=True,
        capacity=2,
        rollout_context_length=50,
    )
    with pytest.raises(ValueError, match="one per-device"):
        args.validate_runtime(cutoff_len=100, per_device_train_batch_size=4)


def test_adaptive_method_validates_rollout_context_limit() -> None:
    args = DistillationArguments(distill_method="adaptive_sfkl", student_gen=True, rollout_context_length=128)
    with pytest.raises(ValueError, match="rollout_context_length"):
        args.validate_runtime(cutoff_len=128, per_device_train_batch_size=1)
