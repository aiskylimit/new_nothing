"""Notebook parity, sampled-layer gradients, and offline checkpoint regressions."""

import ast
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import torch
from torch.nn import functional as F
from transformers import BertConfig, BertModel, BertTokenizer

from ESE.loss import ese_loss, pool_hidden, resolve_dims
from ESE.train import parse_args, train


NOTEBOOKS = sorted(Path(__file__).parent.glob("*.ipynb"))


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module", params=NOTEBOOKS, ids=lambda path: path.stem)
def notebook_loss(request):
    """Execute only the two reference functions, directly from each notebook."""
    notebook = json.loads(request.param.read_text())
    functions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and "def EPRESSO_SimCSE(" in "".join(cell["source"]):
            tree = ast.parse("".join(cell["source"]))
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                         and node.name in {"EPRESSO_SimCSE", "EPRESSO_SimCSE_with_layers"}]
            break
    assert len(functions) == 2
    namespace = {"torch": torch, "F": F, "math": math, "random": random}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(request.param), "exec"), namespace)
    return namespace["EPRESSO_SimCSE_with_layers"]


@pytest.mark.parametrize("blocks,sampled,enabled", [
    (1, 1, True), (2, 1, True), (4, 1, True), (4, 2, True),
    (4, 0, True), (4, -1, True), (4, 99, True), (4, 1, False),
])
@pytest.mark.parametrize("dims", [[2, 4, 8], [8, 2]])
def test_loss_and_gradients_match_notebooks(notebook_loss, blocks, sampled, enabled, dims):
    torch.manual_seed(11)
    first = [torch.randn(4, 3, 8, requires_grad=True) for _ in range(blocks + 1)]
    second = [torch.randn(4, 5, 8, requires_grad=True) for _ in range(blocks + 1)]
    ref_first = [hidden.detach().clone().requires_grad_() for hidden in first]
    ref_second = [hidden.detach().clone().requires_grad_() for hidden in second]
    mask_first = torch.tensor([[1, 1, 0]]).repeat(4, 1)
    mask_second = torch.tensor([[1, 1, 1, 0, 0]]).repeat(4, 1)

    class ReferenceEncoder:
        def __call__(self, input_ids, **kwargs):
            return SimpleNamespace(hidden_states=ref_first if input_ids is first else ref_second)

    random.seed(17)
    actual, metrics = ese_loss(first, second, mask_first, mask_second, matryoshka_dims=dims,
                               n_layers_per_step=sampled, use_intermediate_layers=enabled)
    actual_rng = random.getstate()
    random.seed(17)
    expected, losses, accuracies = notebook_loss(
        ReferenceEncoder(), first, mask_first, second, mask_second, temperature=0.07,
        matryoshka_dims=dims, n_layers_per_step=sampled, use_intermediate_layers=enabled)
    assert random.getstate() == actual_rng
    torch.testing.assert_close(actual, expected)
    for key, value in {**losses, **accuracies}.items():
        assert metrics[key].item() == pytest.approx(value, rel=1e-5, abs=1e-6)
    actual.backward()
    expected.backward()
    for actual_states, ref_states in ((first, ref_first), (second, ref_second)):
        for hidden, ref_hidden in zip(actual_states, ref_states):
            if ref_hidden.grad is None:
                assert hidden.grad is None
            else:
                torch.testing.assert_close(hidden.grad, ref_hidden.grad)


def test_only_sampled_and_final_blocks_receive_direct_supervision():
    torch.manual_seed(11)
    first = [torch.randn(4, 3, 8, requires_grad=True) for _ in range(5)]
    second = [torch.randn(4, 3, 8, requires_grad=True) for _ in range(5)]
    mask = torch.tensor([[1, 1, 0]]).repeat(4, 1)
    with patch("ESE.loss.random.sample", return_value=[2]) as sample:
        loss, metrics = ese_loss(first, second, mask, mask, matryoshka_dims=[2, 4, 8])
    sample.assert_called_once_with([1, 2, 3], 1)
    loss.backward()
    for states in (first, second):
        for layer, hidden in enumerate(states):
            if layer in (2, 4):
                assert torch.isfinite(hidden.grad).all()
                assert hidden.grad[:, :2].abs().sum() > 0
                assert hidden.grad[:, -1].abs().sum() == 0
            else:
                assert hidden.grad is None
    torch.testing.assert_close(loss.detach(), metrics["final_loss"] + metrics["intermediate_loss"])
    assert metrics["sampled_intermediate_layers"] == 1


def test_pooling_ignores_padding():
    hidden = torch.tensor([[[2.0, 4.0], [4.0, 8.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    torch.testing.assert_close(pool_hidden(hidden, mask, "mean"), torch.tensor([[3.0, 6.0]]))
    torch.testing.assert_close(pool_hidden(hidden, mask, "cls"), hidden[:, 0])


def test_dimensions_preserve_weight_order_and_custom_training_set():
    assert resolve_dims(768) == [16, 32, 64, 128, 256, 512, 768]
    assert resolve_dims(1024)[-1] == 1024
    assert resolve_dims(8, [4, 2]) == [4, 2]
    with pytest.warns(UserWarning, match="Filtered"):
        assert resolve_dims(8, [16, 4, 2]) == [4, 2]
    for invalid in ([], [16], [0], [True], [2.5]):
        with pytest.raises(ValueError):
            resolve_dims(8, invalid)


def test_offline_train_save_and_reload(tmp_path, monkeypatch):
    # Keep CPU smoke checks small and independent of model downloads/GPU access.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model_dir = tmp_path / "encoder"
        model_dir.mkdir()
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "good", "bad", "happy", "sad",
                 "day", "night", "sun", "rain"]
        (model_dir / "vocab.txt").write_text("\n".join(vocab) + "\n")
        tokenizer = BertTokenizer(str(model_dir / "vocab.txt"))
        assert tokenizer("good day")["input_ids"] != tokenizer("bad night")["input_ids"]
        tokenizer.save_pretrained(model_dir)
        config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=3,
                            num_attention_heads=2, intermediate_size=32, max_position_embeddings=512)
        torch.manual_seed(3)
        original = BertModel(config)
        original.save_pretrained(model_dir)
        texts = ["good day", "bad night", "happy sun", "sad rain", "good sun", "bad rain",
                 "happy day", "sad night"]
        training_csv = tmp_path / "train.csv"
        pd.DataFrame({"anchor": texts, "positive": texts[::-1]}).to_csv(training_csv, index=False)
        output_dir = tmp_path / "checkpoint"
        args = parse_args([
            "--model_name", str(model_dir), "--data_path", str(training_csv),
            "--output_dir", str(output_dir), "--device", "cpu", "--epochs", "1",
            "--batch_size", "4", "--num_workers", "0", "--max_length", "16",
            "--matryoshka_dims", "4", "16",
            "--gradient_checkpointing", "--run_eval", "0",
        ])
        forward = BertModel.forward
        forward_options = []

        def record_forward(model, *positional, **kwargs):
            forward_options.append(kwargs.get("output_hidden_states"))
            return forward(model, *positional, **kwargs)

        monkeypatch.setattr(BertModel, "forward", record_forward)
        trained = train(args)
        assert forward_options == [True] * 4
        assert trained.config.use_cache is False
        assert not torch.equal(original.embeddings.word_embeddings.weight,
                               trained.embeddings.word_embeddings.weight)
        assert (output_dir / "model.safetensors").is_file()
        assert (output_dir / "tokenizer_config.json").is_file()
        saved_config = json.loads((output_dir / "training_config.json").read_text())
        assert saved_config["pooling"] == "mean"
        assert saved_config["training_dims"] == [4, 16]
        assert saved_config["ese_variant"] == "notebook_epresso_simcse"
        assert saved_config["intermediate_layers"] == "sampled"
        assert saved_config["n_layers_per_step"] == 1
        assert saved_config["dimension_weights"] == pytest.approx([1.0, 1.0 / (1 + math.log(2))])
        assert saved_config["training_data"]["num_pairs"] == len(texts)
        rows = [json.loads(row) for row in (output_dir / "training_metrics.jsonl").read_text().splitlines()]
        assert len(rows) == 2 and rows[-1]["optimizer_step"] == 2
        assert all(math.isfinite(row["loss"]) and row["intermediate_loss"] > 0
                   and row["sampled_intermediate_layers"] == 1 for row in rows)
        reloaded = BertModel.from_pretrained(output_dir)
        torch.testing.assert_close(reloaded.embeddings.word_embeddings.weight,
                                   trained.embeddings.word_embeddings.weight)
    finally:
        torch.set_num_threads(previous_threads)
