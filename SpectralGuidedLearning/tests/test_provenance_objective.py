import torch
import torch.nn.functional as F

from sgl.training.collator import MaskedSFTCollator
from sgl.training.losses import IGNORE_INDEX, OBJECTIVE_DFT, OBJECTIVE_NLL, masked_cross_entropy


def _random_batch(batch_size=2, seq_len=12, vocab=17, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch_size, seq_len, vocab), torch.randint(0, vocab, (batch_size, seq_len))


def _target_prob(logits, labels):
    return torch.softmax(logits[:, :-1].float(), -1).gather(-1, labels[:, 1:, None])[..., 0]


def test_all_nll_objective_equals_default_loss():
    logits, labels = _random_batch()
    objective = torch.full_like(labels, OBJECTIVE_NLL)
    assert torch.allclose(
        masked_cross_entropy(logits, labels, token_objective=objective), masked_cross_entropy(logits, labels)
    )


def test_dft_has_the_gradient_of_minus_p():
    """-sg(p) log p and -p share the gradient -p * grad(log p) (DFT, arXiv 2508.05629)."""
    logits, labels = _random_batch()
    dft_logits = logits.clone().requires_grad_(True)
    masked_cross_entropy(dft_logits, labels, token_objective=torch.full_like(labels, OBJECTIVE_DFT)).backward()

    p_logits = logits.clone().requires_grad_(True)
    n = labels[:, 1:].numel()
    (-_target_prob(p_logits, labels).sum() / n).backward()
    assert torch.allclose(dft_logits.grad, p_logits.grad, atol=1e-6)


def test_mixed_objective_applies_per_token_and_keeps_denominator():
    logits, labels = _random_batch()
    labels = labels.clone()
    labels[:, :3] = IGNORE_INDEX
    objective = torch.full_like(labels, OBJECTIVE_NLL)
    objective[:, 7:] = OBJECTIVE_DFT

    nll = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)).float(), labels[:, 1:].reshape(-1),
        ignore_index=IGNORE_INDEX, reduction="none",
    ).view(labels.size(0), -1)
    coef = torch.where(objective[:, 1:] == OBJECTIVE_DFT, torch.exp(-nll), torch.ones_like(nll))
    expected = (nll * coef).sum() / (labels[:, 1:] != IGNORE_INDEX).sum()
    got = masked_cross_entropy(logits, labels, token_objective=objective)
    assert torch.allclose(got, expected, atol=1e-6)


def test_objective_composes_with_token_weights():
    logits, labels = _random_batch()
    weights = torch.rand(labels.shape) + 0.5
    objective = torch.full_like(labels, OBJECTIVE_DFT)
    nll = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)).float(), labels[:, 1:].reshape(-1), reduction="none"
    ).view(labels.size(0), -1)
    expected = (nll * torch.exp(-nll) * weights[:, 1:]).sum() / nll.numel()
    got = masked_cross_entropy(logits, labels, token_weights=weights, token_objective=objective)
    assert torch.allclose(got, expected, atol=1e-6)


def test_collator_pads_token_source_with_zero():
    collator = MaskedSFTCollator(pad_token_id=0)
    batch = collator([
        {"input_ids": [5, 6, 7, 8], "loss_mask": [0, 1, 1, 1], "token_source": [0, 1, 2, 2]},
        {"input_ids": [5, 6], "loss_mask": [0, 1], "token_source": [0, 2]},
    ])
    assert batch["token_source"].tolist() == [[0, 1, 2, 2], [0, 2, 0, 0]]


def test_junction_tokens_get_their_own_source(tmp_path, monkeypatch):
    import json

    from sgl.transforms.provenance import build

    class Tok:  # char-level tokenizer: token i == character i
        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            out = {"input_ids": [ord(c) for c in text]}
            if return_offsets_mapping:
                out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
            return out

    import sgl.transforms.provenance as module

    monkeypatch.setattr(module, "AutoTokenizer", type("A", (), {"from_pretrained": staticmethod(lambda name: Tok())}))
    response = "ab<End_of_Prefix>cdefg"
    prompt_ids = [1, 2]
    ids = prompt_ids + [ord(c) for c in response] + [0]
    start, end = len(prompt_ids), len(prompt_ids) + len(response)
    (tmp_path / "seg.jsonl").write_text(json.dumps(
        {"id": 0, "input_ids": ids, "response": response, "response_token_span": [start, end]}) + "\n")
    mask = [0, 0] + [1] * (len(response) + 1)
    (tmp_path / "van.jsonl").write_text(json.dumps({"id": 0, "input_ids": ids, "loss_mask": mask}) + "\n")
    build(tmp_path / "seg.jsonl", tmp_path / "van.jsonl", tmp_path / "out.jsonl", "x", junction_tokens=2)
    source = json.loads((tmp_path / "out.jsonl").read_text())["token_source"]
    n_prefix = len("ab<End_of_Prefix>")
    assert source == [0, 0] + [1] * n_prefix + [3, 3] + [2] * 3 + [2]
