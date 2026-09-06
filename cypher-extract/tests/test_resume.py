import json
from pathlib import Path

import pytest

from distillation.resume import (
    ResumeCheckpointError,
    canonical_resume_config,
    full_model_checkpoint_identity,
    validate_fresh_output_dir,
    validate_resume_checkpoint,
    write_resume_manifest,
)


def _write(path: Path, content: str = "state") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_checkpoint(
    output_dir: Path,
    *,
    step: int = 10,
    world_size: int = 1,
    deepspeed: bool = False,
    distillm: bool = False,
) -> Path:
    checkpoint = output_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    _write(
        checkpoint / "trainer_state.json",
        json.dumps({"global_step": step, "train_batch_size": 2}),
    )
    if world_size == 1:
        _write(checkpoint / "rng_state.pth")
    else:
        for rank in range(world_size):
            _write(checkpoint / f"rng_state_{rank}.pth")

    if deepspeed:
        tag = f"global_step{step}"
        _write(checkpoint / "latest", tag)
        _write(checkpoint / tag / "mp_rank_00_model_states.pt")
        for rank in range(world_size):
            _write(checkpoint / tag / f"zero_pp_rank_{rank}_mp_rank_00_optim_states.pt")
    else:
        _write(checkpoint / "adapter_model.safetensors")
        _write(checkpoint / "optimizer.pt")
        _write(checkpoint / "scheduler.pt")

    if distillm:
        for rank in range(world_size):
            _write(checkpoint / f"distillm_state_rank{rank}.pt")
    return checkpoint


def _validate(
    checkpoint: Path | str | bool | None,
    output_dir: Path,
    *,
    world_size: int = 1,
    deepspeed: bool = False,
    distillm: bool = False,
) -> Path | None:
    return validate_resume_checkpoint(
        checkpoint,
        output_dir=output_dir,
        world_size=world_size,
        deepspeed_enabled=deepspeed,
        require_distillm_state=distillm,
        train_batch_size=2,
    )


def test_no_resume_returns_none(tmp_path: Path) -> None:
    assert _validate(None, tmp_path) is None
    assert _validate(False, tmp_path) is None


def test_validates_complete_standard_checkpoint(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path)

    assert _validate(checkpoint, tmp_path) == checkpoint.resolve()


def test_validates_complete_distributed_deepspeed_and_distillm_checkpoint(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path, world_size=2, deepspeed=True, distillm=True)

    assert _validate(checkpoint, tmp_path, world_size=2, deepspeed=True, distillm=True) == checkpoint.resolve()


@pytest.mark.parametrize("automatic_value", [True, "true", "auto", "latest"])
def test_rejects_automatic_latest_resume(tmp_path: Path, automatic_value: str | bool) -> None:
    with pytest.raises(ResumeCheckpointError, match="explicit checkpoint path|checkpoint-N explicitly"):
        _validate(automatic_value, tmp_path)


def test_rejects_checkpoint_from_another_output_directory(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path / "fkl")

    with pytest.raises(ResumeCheckpointError, match="directly inside output_dir"):
        _validate(checkpoint, tmp_path / "rkl")


def test_rejects_trainer_step_or_batch_size_mismatch(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path, step=10)
    _write(checkpoint / "trainer_state.json", json.dumps({"global_step": 9, "train_batch_size": 4}))

    with pytest.raises(ResumeCheckpointError) as exc_info:
        _validate(checkpoint, tmp_path)

    message = str(exc_info.value)
    assert "does not match checkpoint step 10" in message
    assert "train_batch_size changed from 4 to 2" in message


def test_rejects_incomplete_standard_training_state(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path)
    (checkpoint / "optimizer.pt").unlink()
    (checkpoint / "scheduler.pt").unlink()

    with pytest.raises(ResumeCheckpointError) as exc_info:
        _validate(checkpoint, tmp_path)

    message = str(exc_info.value)
    assert "missing optimizer state" in message
    assert "missing scheduler.pt" in message


def test_rejects_different_distributed_world_size(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path, world_size=2, deepspeed=True)

    with pytest.raises(ResumeCheckpointError) as exc_info:
        _validate(checkpoint, tmp_path, world_size=1, deepspeed=True)

    message = str(exc_info.value)
    assert "distributed RNG states" in message
    assert "unexpected ranks [1]" in message


def test_rejects_missing_distillm_rank_state(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path, world_size=2, deepspeed=True, distillm=True)
    (checkpoint / "distillm_state_rank1.pt").unlink()

    with pytest.raises(ResumeCheckpointError, match=r"DistiLLM states: missing ranks \[1\]"):
        _validate(checkpoint, tmp_path, world_size=2, deepspeed=True, distillm=True)


def test_rejects_deepspeed_latest_tag_mismatch(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path, deepspeed=True)
    _write(checkpoint / "latest", "global_step9")

    with pytest.raises(ResumeCheckpointError, match="expected 'global_step10'"):
        _validate(checkpoint, tmp_path, deepspeed=True)


def test_fresh_training_rejects_output_directory_with_old_checkpoints(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-10").mkdir()

    with pytest.raises(ResumeCheckpointError, match="Refusing to start fresh training"):
        validate_fresh_output_dir(tmp_path)


def test_fresh_training_accepts_new_or_checkpoint_free_output_directory(tmp_path: Path) -> None:
    validate_fresh_output_dir(tmp_path / "new-run")
    (tmp_path / "logs").mkdir()
    validate_fresh_output_dir(tmp_path)


def test_resume_manifest_ignores_checkpoint_path_revision_pins_and_retention_limit(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path)
    original = {
        "distill_method": "fkl",
        "gradient_accumulation_steps": 8,
        "resume_from_checkpoint": None,
    }
    resumed = {
        "distill_method": "fkl",
        "gradient_accumulation_steps": 8,
        "resume_from_checkpoint": str(checkpoint),
        "model_revision": "student-commit",
        "ref_model_revision": "teacher-commit",
        "save_total_limit": None,
    }
    write_resume_manifest(
        checkpoint,
        config=original,
        runtime_identity={"dataset": "fingerprint-a"},
        world_size=1,
        global_step=10,
    )

    result = validate_resume_checkpoint(
        checkpoint,
        output_dir=tmp_path,
        world_size=1,
        deepspeed_enabled=False,
        require_distillm_state=False,
        train_batch_size=2,
        expected_config=canonical_resume_config(resumed),
        expected_runtime={"dataset": "fingerprint-a"},
    )

    assert result == checkpoint.resolve()


def test_resume_manifest_rejects_changed_training_config(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path)
    write_resume_manifest(
        checkpoint,
        config={"distill_method": "fkl", "gradient_accumulation_steps": 8},
        runtime_identity={"dataset": "fingerprint-a"},
        world_size=1,
        global_step=10,
    )

    with pytest.raises(ResumeCheckpointError, match="changed fields: gradient_accumulation_steps"):
        validate_resume_checkpoint(
            checkpoint,
            output_dir=tmp_path,
            world_size=1,
            deepspeed_enabled=False,
            require_distillm_state=False,
            train_batch_size=2,
            expected_config={"distill_method": "fkl", "gradient_accumulation_steps": 4},
            expected_runtime={"dataset": "fingerprint-a"},
        )


def test_resume_manifest_rejects_changed_runtime_content(tmp_path: Path) -> None:
    checkpoint = _make_checkpoint(tmp_path)
    write_resume_manifest(
        checkpoint,
        config={"distill_method": "fkl"},
        runtime_identity={"train_dataset": {"fingerprint": "old"}, "teacher_adapters": [{"sha256": "old"}]},
        world_size=1,
        global_step=10,
    )

    with pytest.raises(ResumeCheckpointError, match="changed sections: teacher_adapters, train_dataset"):
        validate_resume_checkpoint(
            checkpoint,
            output_dir=tmp_path,
            world_size=1,
            deepspeed_enabled=False,
            require_distillm_state=False,
            train_batch_size=2,
            expected_config={"distill_method": "fkl"},
            expected_runtime={
                "train_dataset": {"fingerprint": "new"},
                "teacher_adapters": [{"sha256": "new"}],
            },
        )


def test_full_teacher_checkpoint_identity_changes_with_weights(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher_full"
    _write(teacher / "config.json", "{}")
    _write(teacher / "model.safetensors", "first")
    first = full_model_checkpoint_identity(str(teacher))

    _write(teacher / "model.safetensors", "second")
    second = full_model_checkpoint_identity(str(teacher))

    assert first != second
