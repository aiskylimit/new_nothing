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
