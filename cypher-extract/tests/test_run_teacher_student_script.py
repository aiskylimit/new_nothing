from pathlib import Path


def test_teacher_student_runner_skips_completed_training_by_default() -> None:
    script = Path("scripts/run_teacher_student.sh").read_text(encoding="utf-8")

    assert 'SKIP_COMPLETED="${SKIP_COMPLETED:-1}"' in script
    assert "--no-skip-completed" in script
    assert '[[ -s "${output_dir}/train_results.json" ]]' in script
    assert '[[ -s "${output_dir}/trainer_state.json" ]]' in script
    assert 'has_final_model_weights "${output_dir}"' in script
    assert '"${model_family} teacher: ${setting}"' in script
    assert '"${model_family} student: ${setting}/${method}"' in script
    assert script.count("run_training \\") == 2


def test_teacher_student_runner_can_force_retraining_and_reinference() -> None:
    script = Path("scripts/run_teacher_student.sh").read_text(encoding="utf-8")

    assert 'RETRAIN="${RETRAIN:-0}"' in script
    assert 'REINFER="${REINFER:-0}"' in script
    assert "    --retrain)\n" in script
    assert "    --reinfer)\n" in script
    # Retraining clears the old output_dir because fresh training rejects stale checkpoints.
    assert 'remove_training_output "${label}" "${output_dir}"' in script
    assert 'rm -rf -- "${normalized}"' in script
    # Old inference outputs belong to the deleted checkpoints, so retrain implies reinfer.
    assert 'if [[ "${RETRAIN}" == "1" ]]; then\n  REINFER=1\nfi' in script
    assert "inference_args+=(--overwrite)" in script


def test_project_command_retrains_and_reinfers() -> None:
    command = Path("project_command.sh").read_text(encoding="utf-8")

    assert "--retrain" in command
    assert "--reinfer" in command
