import json

from evaluate import score_file


def write_raw(tmp_path, name, rows):
    path = tmp_path / f"{name}.jsonl"
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def problem(gold, answers):
    return {
        "id": 0,
        "gold": gold,
        "task_type": "math",
        "generations": [
            {"text": rf"so \boxed{{{answer}}}", "finish_reason": "stop", "n_tokens": 10}
            for answer in answers
        ],
    }


def test_pass_at_1_averages_over_samples(tmp_path):
    # one problem, 3 samples, 1 correct -> pass@1 = 1/3, pass@3 = 1 (any correct)
    path = write_raw(tmp_path, "aime24", [problem("42", ["42", "7", "7"])])
    summary = score_file(path, grader="builtin")
    assert summary["pass@1"] == 1 / 3
    assert summary["pass@3"] == 1.0
    assert summary["samples_per_problem"] == 3


def test_pass_at_k_counts_problems_not_generations(tmp_path):
    rows = [
        problem("42", ["42", "7", "7"]),  # solved
        problem("13", ["1", "2", "3"]),  # unsolved
    ]
    path = write_raw(tmp_path, "aime25", rows)
    summary = score_file(path, grader="builtin")
    assert summary["pass@3"] == 0.5
    assert summary["pass@1"] == 1 / 6


def test_pass_at_1_equals_accuracy_when_single_sample(tmp_path):
    path = write_raw(tmp_path, "math500", [problem("42", ["42"]), problem("13", ["7"])])
    summary = score_file(path, grader="builtin")
    assert summary["pass@1"] == summary["accuracy"] == 0.5
    assert summary["samples_per_problem"] == 1
    assert "pass@1" in summary
