import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner():
    return load_file("shared_eval", ROOT / "palign/src/run_eval.py")


def test_changed_sample_count_keeps_pass3_distinct_from_passn(runner):
    result = runner.summarize([{"label": [1, 0, 0, 0]}], 4)
    assert result["pass@1"] == 25
    assert result["pass@3"] == 75
    assert result["pass@4"] == 100
    assert runner.summarize([{"label": [0, 0, 0]}], 3)["pass@3"] == 0
    with pytest.raises(ValueError, match="expected 3"):
        runner.summarize([{"label": [1]}], 3)


def test_common_pipeline_saves_generations_and_scores_arbitrary_ood(runner, tmp_path, monkeypatch):
    source = tmp_path / "ood.jsonl"
    source.write_text(json.dumps({"question": "Q?", "answer": "18"}) + "\n")
    result_dir = tmp_path / "result"
    args = runner.parse_args([
        "--model", "checkpoint", "--input-files", str(source),
        "--result-dir", str(result_dir), "--n", "4",
    ])
    real_run = subprocess.run
    calls = []

    def fake_generation(command, **kwargs):
        calls.append(command)
        if Path(command[1]).name == "test.py":
            output = Path(command[command.index("--output_files") + 1])
            output.write_text(json.dumps({
                "answer": "18", "output": ["$\\boxed{18}$", "", "$\\boxed{3}$", ""],
            }) + "\n")
            return SimpleNamespace(returncode=0)
        return real_run(command, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", fake_generation)
    summary = runner.run(args)
    assert summary["ood"]["pass@1"] == 25
    assert summary["ood"]["pass@3"] == 75
    assert summary["ood"]["pass@4"] == 100
    assert json.loads((result_dir / "ood.jsonl").read_text())["output"][1] == ""
    config = json.loads((result_dir / "evaluation_config.json").read_text())
    assert config["backend"] == "vllm"
    assert config["grader"] == "math-verify"
    assert config["input_sha256"]["ood"]
    assert "--max_model_len" in calls[0]
    assert calls[0][calls[0].index("--n") + 1] == "4"
    with pytest.raises(FileExistsError):
        runner.run(args)


def test_context_has_generation_room(runner, tmp_path):
    args = runner.parse_args([
        "--model", "x", "--result-dir", str(tmp_path), "--max-model-len", "4096",
    ])
    with pytest.raises(ValueError, match="room for the prompt"):
        runner.run(args)


def test_generator_preserves_exact_palign_prompt_and_checks_candidate_count(tmp_path, monkeypatch):
    module = load_file("palign_generation", ROOT / "palign/src/test.py")
    source = tmp_path / "questions.jsonl"
    source.write_text(json.dumps({"question": "Q?", "answer": "1"}) + "\n")
    seen = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            seen.append(messages)
            return messages[0]["content"]

    class LLM:
        def generate(self, prompts, params, **kwargs):
            return [SimpleNamespace(outputs=[SimpleNamespace(text="answer")] * params)]

    module.process_data(str(source), str(tmp_path / "raw.jsonl"), LLM(), 1, Tokenizer(), 3, 3)
    assert seen[0] == [{
        "role": "user",
        "content": "Please reason step by step, and put your final answer within \\boxed{}.Q?",
    }]
    with pytest.raises(RuntimeError, match="expected 3 candidates"):
        module.process_data(str(source), str(tmp_path / "bad.jsonl"), LLM(), 1, Tokenizer(), 2, 3)


def test_vllm_adapter_is_enabled_with_the_same_seed(monkeypatch):
    module = load_file("palign_generation", ROOT / "palign/src/test.py")
    seen = {}
    monkeypatch.setattr(module, "LLM", lambda **kwargs: seen.update(kwargs) or object())
    module._make_llm("base", 32768, 1, 0.8, seed=42, adapter_config={"r": 16})
    assert seen["enable_lora"] is True
    assert seen["model"] == "base"
    assert seen["seed"] == 42
    assert seen["max_model_len"] == 32768


def test_ssft_counters_are_initialized_before_model_load():
    path = ROOT / "SegmentSelectiveSFT/SelectiveSFT/train_mask.py"
    run = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.FunctionDef) and node.name == "run")

    class ReachedModelLoad(Exception):
        pass

    def from_pretrained(**kwargs):
        raise ReachedModelLoad

    namespace = {
        "os": SimpleNamespace(environ={}),
        "FastLanguageModel": SimpleNamespace(from_pretrained=from_pretrained),
    }
    exec(compile(ast.Module(body=[run], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 - trusted local function
    args = SimpleNamespace(model_name_or_path="base", max_seq_length=32768,
                           load_in_4bit=False, use_lora=True)
    with pytest.raises(ReachedModelLoad):
        namespace["run"](args)


def test_onepass_isolates_checkpoints_and_refuses_reuse(tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps({"question": "Q", "solution": "R"}) + "\n")
    script = ROOT / "SegmentSelectiveSFT/one_pass.py"
    for name in ("run-a", "run-b"):
        result = subprocess.run([
            sys.executable, str(script), "--train-data", str(source),
            "--output-dir", str(tmp_path / "runs"), "--run-id", name, "--dry-run",
        ], check=False, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        checkpoint = tmp_path / "runs" / name / "checkpoint-final"
        checkpoint.mkdir()
        (checkpoint / "adapter_config.json").write_text("{}")
    result = subprocess.run([
        sys.executable, str(script), "--train-data", str(source),
        "--output-dir", str(tmp_path / "runs"), "--run-id", "run-a",
    ], check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "already contains a run" in result.stderr
    assert (tmp_path / "runs/run-b/checkpoint-final/adapter_config.json").exists()


def test_ssft_shell_routes_sample_override_to_shared_evaluator():
    env = os.environ.copy()
    env.update(EVAL_PYTHON="/bin/echo", EVAL_N="5")
    result = subprocess.run([
        "bash", str(ROOT / "SegmentSelectiveSFT/eval.sh"), "--model", "adapter",
    ], env=env, check=True, capture_output=True, text=True)
    assert "palign/src/run_eval.py" in result.stdout
    assert "--n 5" in result.stdout
    assert "--model adapter" in result.stdout


def test_ssft_training_launch_uses_three_workers_and_absolute_data(tmp_path):
    source = tmp_path / "selected.jsonl"
    source.write_text("{}\n")
    result = subprocess.run([
        "bash", str(ROOT / "SegmentSelectiveSFT/train.sh"), "--skip-setup",
        "--dry-run", "--data", str(source),
    ], check=True, capture_output=True, text=True)
    assert "--standalone --nproc_per_node 3" in result.stdout
    assert "--per_device_train_batch_size 11" in result.stdout
    assert "--gradient_accumulation_steps 1" in result.stdout
    assert "--seed 42" in result.stdout
    assert f"--data_names {source}" in result.stdout
