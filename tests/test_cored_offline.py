"""Exercise offline asset loading with Hub/cache/network entry points blocked."""
import importlib
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cored"))
from core.local_assets import load_local_rows, local_model_path
from core.model_utils import load_base_model, load_experts, load_tokenizer
from eval.benchmarks import load_benchmark


@pytest.fixture(autouse=True)
def offline(tmp_path, monkeypatch):
    import datasets
    import huggingface_hub

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline runtime attempted a network/Hub loader")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(datasets, "load_dataset", forbidden)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)
    monkeypatch.setenv("CORED_ASSET_ROOT", str(tmp_path / "cored"))


def write_parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def model_snapshot(tmp_path, name="Qwen2.5-7B-Instruct"):
    path = tmp_path / "cored/models" / name
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    return path


def test_snapshot_split_selection_and_missing_split(tmp_path):
    root = tmp_path / "cored/datasets/s1K-1.1"
    rows = [{"question": "training", "answer": "42"}]
    write_parquet(root / "data/train-00000-of-00001.parquet", rows)
    write_parquet(root / "data/test-00000-of-00001.parquet", [{"question": "test"}])
    assert load_local_rows("datasets/s1K-1.1", split="train") == rows
    with pytest.raises(FileNotFoundError, match="validation"):
        load_local_rows(root, split="validation")
    with pytest.raises(FileNotFoundError, match="Local dataset missing"):
        load_local_rows("simplescaling/s1K-1.1", split="train")


@pytest.mark.parametrize("name", ["Qwen2.5-7B-Instruct", "DeepSeek-R1-Distill-Qwen-1.5B"])
def test_transformer_loaders_get_absolute_local_paths_and_offline_flag(tmp_path, monkeypatch, name):
    import torch

    path = model_snapshot(tmp_path, name)
    tokenizer = SimpleNamespace(pad_token_id=None, eos_token="EOS", is_fast=True)
    model = torch.nn.Linear(1, 1)
    model.config = SimpleNamespace(use_cache=True)
    token_loader = Mock(return_value=tokenizer)
    model_loader = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=token_loader),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_loader)))
    assert load_tokenizer(f"models/{name}") is tokenizer
    assert load_base_model(f"models/{name}") is model
    for loader in (token_loader, model_loader):
        assert loader.call_args.args == (str(path),)
        assert loader.call_args.kwargs["local_files_only"] is True
    assert tokenizer.pad_token == "EOS"
    assert not model.config.use_cache
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_missing_model_or_hub_id_fails_before_library_loader(tmp_path, monkeypatch):
    loader = Mock(side_effect=AssertionError("Should validate local path first"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=loader)))
    for path in ("models/missing", "Qwen/Qwen2.5-7B-Instruct"):
        with pytest.raises(FileNotFoundError, match="Local model missing"):
            load_tokenizer(path)
    loader.assert_not_called()
    with pytest.raises(ValueError):
        local_model_path("")


def test_expert_adapters_are_local_and_missing_adapter_fails(tmp_path, monkeypatch):
    dirs = [tmp_path / f"expert_{i}" for i in range(2)]
    for path in dirs:
        path.mkdir()
        (path / "adapter_config.json").write_text("{}")
    model = SimpleNamespace(load_adapter=Mock())
    loader = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        PeftModel=SimpleNamespace(from_pretrained=loader)))
    assert load_experts(object(), [str(p) for p in dirs])[0] is model
    assert loader.call_args.kwargs["local_files_only"] is True
    assert model.load_adapter.call_args.kwargs["local_files_only"] is True
    with pytest.raises(FileNotFoundError, match="Local adapter missing"):
        load_experts(object(), [str(tmp_path / "absent")])


def test_vllm_receives_local_backbone_and_adapter_paths(tmp_path, monkeypatch):
    from eval.run_eval import _VLLMRunner

    model = model_snapshot(tmp_path, "DeepSeek-R1-Distill-Qwen-1.5B")
    adapter = tmp_path / "student"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    engine = Mock()
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=engine))
    args = SimpleNamespace(model_name="models/DeepSeek-R1-Distill-Qwen-1.5B",
                           max_lora_rank=16, gpu_memory_utilization=0.8,
                           max_model_len=32768, max_tokens=4096, seed=42)
    runner = _VLLMRunner(args, str(adapter))
    assert engine.call_args.kwargs["model"] == str(model)
    assert engine.call_args.kwargs["enable_lora"] is True
    assert runner.adapter_path == str(adapter)


def test_benchmarks_use_expected_local_files_and_answer_formats(tmp_path):
    write_parquet(tmp_path / "palign/datasets/aimo-validation-amc/data/train-00000.parquet",
                  [{"problem": "AMC", "answer": 142.0}])
    assert load_benchmark("amc")[0].answer == "142"
    math = tmp_path / "palign/datasets/MATH-500/test.jsonl"
    math.parent.mkdir(parents=True)
    math.write_text(json.dumps({"problem": "MATH", "answer": "7"}) + "\n")
    assert load_benchmark("math500")[0].question == "MATH"
    write_parquet(tmp_path / "cored/datasets/aime_2024/data/train-00000.parquet",
                  [{"problem": "AIME24", "answer": "0"}])
    assert load_benchmark("aime24")[0].answer == "0"
    write_parquet(tmp_path / "cored/datasets/gsm8k/main/test-00000.parquet",
                  [{"question": "GSM", "answer": "reasoning\n#### 1,234"}])
    assert load_benchmark("gsm8k")[0].answer == "1234"


def test_aime25_requires_both_parts_and_30_unique_questions(tmp_path):
    root = tmp_path / "cored/datasets/AIME2025"
    root.mkdir(parents=True)
    for part in ("I", "II"):
        (root / f"aime2025-{part}.jsonl").write_text("".join(
            json.dumps({"question": f"{part}-{i}", "answer": i}) + "\n" for i in range(15)))
    assert len(load_benchmark("aime25")) == 30
    assert len(load_benchmark("aime25", limit=4)) == 4
    (root / "aime2025-II.jsonl").write_bytes((root / "aime2025-I.jsonl").read_bytes())
    with pytest.raises(ValueError, match="30 unique"):
        load_benchmark("aime25")
    (root / "aime2025-II.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        load_benchmark("aime25")


def test_curiosft_resolves_against_asset_root(tmp_path):
    path = tmp_path / "cored/data/curiosft_raw/test.math.parquet"
    write_parquet(path, [{"data_source": source, "prompt": [{"content": source}],
                          "reward_model": {"ground_truth": "42"}}
                         for source in ("math", "amc")])
    assert [item.question for item in load_benchmark("cs_amc")] == ["amc"]


def test_s1k_preparation_preserves_longcot_fields(tmp_path, monkeypatch):
    prepare = importlib.import_module("data.prepare_s1k")
    write_parquet(tmp_path / "cored/datasets/s1K-1.1/data/train-00000.parquet", [{
        "question": "Q", "deepseek_thinking_trajectory": "Step1\n\nStep2",
        "deepseek_attempt": "Final", "solution": "42", "source_type": "math"}])
    monkeypatch.setattr(prepare, "load_tokenizer", lambda model: lambda *args, **kwargs: {"input_ids": [1, 2]})
    out = tmp_path / "train.jsonl"
    monkeypatch.setattr(sys, "argv", ["prepare_s1k.py", "--out", str(out)])
    prepare.main()
    row = json.loads(out.read_text())
    assert row["answer"] == "42"
    assert row["answer_text"] == "Final"
    assert row["full_cot"] == "<think>\nStep1\n\nStep2\n</think>\n\nFinal"
    assert row["steps"] == ["Step1", "Step2"]
