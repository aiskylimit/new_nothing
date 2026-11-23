"""Downloaded model paths and training must work with network access blocked."""

import importlib
import json
from pathlib import Path
import socket

import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizer

from utils.model_paths import MODEL_REPOS, downloaded_model_path, resolve_local_model


@pytest.mark.parametrize("key", MODEL_REPOS)
def test_alias_and_repo_id_resolve_to_downloaded_directory(tmp_path, monkeypatch, key):
    monkeypatch.setenv("MRL_MODELS_DIR", str(tmp_path))
    expected = tmp_path / MODEL_REPOS[key].split("/")[-1]
    expected.mkdir()
    assert downloaded_model_path(key) == str(expected)
    assert resolve_local_model(key) == str(expected)
    assert resolve_local_model(MODEL_REPOS[key]) == str(expected)


def test_default_location_does_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("MRL_MODELS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    project = Path(__file__).resolve().parent.parent
    assert downloaded_model_path("bgem3") == str(project / "models" / "bge-m3")
    with pytest.raises(FileNotFoundError, match="no Hub download"):
        resolve_local_model(str(tmp_path / "missing"))


def test_explicit_directory_takes_priority_over_alias(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "bert").mkdir()
    assert resolve_local_model("bert") == str(tmp_path / "bert")


@pytest.mark.parametrize("method", ["MRL", "MIPIC", "FlagMRL", "ESE"])
def test_train_from_downloaded_encoder_without_network(tmp_path, monkeypatch, method):
    def blocked(*args, **kwargs):
        raise AssertionError("Model loading attempted network access")

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    module = importlib.import_module(f"{method}.train")
    model_dir = tmp_path / "downloaded-encoder"
    model_dir.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "good", "bad", "day", "night"]
    (model_dir / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tokenizer = BertTokenizer(str(model_dir / "vocab.txt"))
    tokenizer.save_pretrained(model_dir)
    original = BertModel(BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=3,
        num_attention_heads=2, intermediate_size=32))
    original.save_pretrained(model_dir)
    csv_path = tmp_path / "pairs.csv"
    pd.DataFrame({"anchor": ["good day", "bad night", "good night", "bad day"],
                  "positive": ["good night", "bad day", "good day", "bad night"]}).to_csv(csv_path, index=False)
    output = tmp_path / "checkpoint"
    args = module.parse_args([
        "--model_name", str(model_dir), "--data_path", str(csv_path),
        "--output_dir", str(output), "--device", "cpu", "--epochs", "1",
        "--batch_size", "4", "--num_workers", "0", "--max_length", "8",
        "--matryoshka_dims", "4", "16", "--run_eval", "0",
    ])
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        module.train(args)
        metadata = json.loads((output / "training_config.json").read_text())
        assert metadata["model_name"] == str(model_dir)
        from utils.mteb_eval import evaluate

        # Exercise the actual checkpoint/model/tokenizer loading paths while
        # keeping dataset preparation and MTEB scoring outside this regression.
        monkeypatch.setattr("utils.mteb_eval.require_local_tasks", lambda *a, **k: None)

        def inspect_loaded(model, loaded_tokenizer, *args, **kwargs):
            batch = loaded_tokenizer(["good day"], return_tensors="pt")
            assert model(**batch).last_hidden_state.shape == (1, 4, 16)
            return {"loaded": True}

        monkeypatch.setattr("utils.mteb_eval.evaluate_model", inspect_loaded)
        assert evaluate(checkpoint_dir=output, method=method.lower(), eval_tasks=["STSBenchmark"],
                        device="cpu") == {"loaded": True}
        assert evaluate(model_name=str(model_dir), method="original", eval_tasks=["STSBenchmark"],
                        device="cpu") == {"loaded": True}
    finally:
        torch.set_num_threads(previous_threads)
