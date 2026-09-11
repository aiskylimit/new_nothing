import pytest
import torch
import torch.nn.functional as F

from data_collator import MaskedSFTCollator
from masked_loss import IGNORE_INDEX, masked_cross_entropy
from training_utils import compensate_global_token_mean, set_training_seed


def _random_batch(batch_size=2, seq_len=12, vocab=17, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch_size, seq_len, vocab), torch.randint(0, vocab, (batch_size, seq_len))


def test_full_mask_equals_standard_sft_loss():
    """The spectral objective must reduce exactly to vanilla SFT when nothing is masked."""
    logits, labels = _random_batch()
    reference = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)).float(), labels[:, 1:].reshape(-1)
    )
    assert torch.allclose(masked_cross_entropy(logits, labels), reference, atol=1e-6)


def test_masked_positions_contribute_no_gradient():
    logits, labels = _random_batch()
    logits = logits.requires_grad_(True)
    masked_labels = labels.clone()
    masked_labels[:, 6:] = IGNORE_INDEX

    masked_cross_entropy(logits, masked_labels).backward()

    # position i's logits predict token i+1; masking tokens 6.. frees logits from index 5 on
    assert torch.all(logits.grad[:, 5:] == 0)
    assert torch.any(logits.grad[:, :5] != 0)


def test_loss_ignores_masked_tokens_values():
    """Changing the identity of masked-out target tokens must not change the loss."""
    logits, labels = _random_batch()
    labels_a, labels_b = labels.clone(), labels.clone()
    labels_a[:, 8:] = IGNORE_INDEX
    labels_b[:, 8:] = IGNORE_INDEX
    loss_a = masked_cross_entropy(logits, labels_a)
    labels_b[:, 3] = (labels_b[:, 3] + 1) % logits.size(-1)  # unmasked change -> must differ
    assert not torch.isclose(loss_a, masked_cross_entropy(logits, labels_b))


def test_every_supervised_token_weighs_the_same():
    """Z is a token count over the batch, so mask density must not reweight sequences."""
    vocab = 5
    logits = torch.zeros(2, 9, vocab)  # uniform -> every token loss = log(vocab)
    labels = torch.zeros(2, 9, dtype=torch.long)
    labels[0, 3:] = IGNORE_INDEX  # 2 supervised targets after the shift
    labels[1, 1] = IGNORE_INDEX  # 7 supervised targets after the shift
    # per-sequence averaging would also give log(vocab) here, so check the token counts too
    loss = masked_cross_entropy(logits, labels)
    assert torch.isclose(loss, torch.tensor(vocab, dtype=torch.float).log(), atol=1e-6)
    assert (labels[:, 1:] != IGNORE_INDEX).sum() == 9


def test_explicit_denominator_overrides_batch_count():
    logits, labels = _random_batch()
    supervised = int((labels[:, 1:] != IGNORE_INDEX).sum())
    default = masked_cross_entropy(logits, labels)
    assert torch.allclose(
        masked_cross_entropy(logits, labels, denominator=2 * supervised), default / 2, atol=1e-6
    )


def test_uniform_token_weights_recover_the_spectral_loss_exactly():
    logits, labels = _random_batch()
    labels[:, 7:] = IGNORE_INDEX
    weights = (labels != IGNORE_INDEX).float()
    assert torch.allclose(
        masked_cross_entropy(logits, labels, token_weights=weights),
        masked_cross_entropy(logits, labels),
        atol=1e-6,
    )


def test_token_weights_redistribute_loss_without_changing_denominator():
    logits = torch.zeros(1, 4, 2)
    labels = torch.tensor([[0, 0, 1, IGNORE_INDEX]])
    weights = torch.tensor([[0.0, 2.0, 0.5, 0.0]])
    # Both selected targets have CE=log(2), so the numerator has coefficient 2.5
    # while Z remains the two selected tokens.
    expected = 1.25 * torch.tensor(2.0).log()
    assert torch.allclose(masked_cross_entropy(logits, labels, token_weights=weights), expected)


def test_shared_denominator_makes_microbatches_sum_to_the_full_batch_loss():
    """Gradient accumulation: two microbatches under one Z must equal the single-batch loss."""
    logits, labels = _random_batch(batch_size=4, seq_len=10)
    labels[0, 5:] = IGNORE_INDEX  # uneven mask density across the batch
    labels[3, 2:7] = IGNORE_INDEX

    full = masked_cross_entropy(logits, labels)
    total_supervised = int((labels[:, 1:] != IGNORE_INDEX).sum())
    halves = sum(
        masked_cross_entropy(logits[start:start + 2], labels[start:start + 2], total_supervised)
        for start in (0, 2)
    )
    assert torch.allclose(halves, full, atol=1e-6)


def test_collator_builds_labels_from_loss_mask_and_pads():
    collator = MaskedSFTCollator(pad_token_id=0)
    batch = collator(
        [
            {"input_ids": [5, 6, 7, 8], "loss_mask": [0, 0, 1, 1]},
            {"input_ids": [9, 10], "loss_mask": [0, 1]},
        ]
    )

    assert batch["input_ids"].tolist() == [[5, 6, 7, 8], [9, 10, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert batch["labels"].tolist() == [
        [IGNORE_INDEX, IGNORE_INDEX, 7, 8],
        [IGNORE_INDEX, 10, IGNORE_INDEX, IGNORE_INDEX],
    ]


def test_collator_rejects_mask_length_mismatch():
    with pytest.raises(ValueError):
        MaskedSFTCollator(pad_token_id=0)([{"input_ids": [1, 2, 3], "loss_mask": [1, 1]}])


def test_collator_pads_optional_loss_weights():
    batch = MaskedSFTCollator(pad_token_id=0)(
        [
            {"input_ids": [5, 6, 7], "loss_mask": [0, 1, 1], "loss_weights": [0.0, 0.5, 1.5]},
            {"input_ids": [8, 9], "loss_mask": [0, 1], "loss_weights": [0.0, 1.0]},
        ]
    )
    assert batch["loss_weights"].tolist() == [[0.0, 0.5, 1.5], [0.0, 1.0, 0.0]]


def test_trainer_recovers_global_token_mean_under_ddp_averaging():
    # The gathered global Z is 4 but this rank contributes two selected tokens. DDP
    # averages gradients later, so a custom sum/global-Z loss must multiply by world size.
    assert torch.allclose(
        compensate_global_token_mean(
            torch.tensor(0.5), average_tokens_across_devices=True,
            num_items_in_batch=4, num_processes=2,
        ),
        torch.tensor(1.0),
    )


def test_training_seed_is_repeatable_before_adapter_construction():
    set_training_seed(42)
    first = torch.rand(5)
    set_training_seed(42)
    assert torch.equal(first, torch.rand(5))
