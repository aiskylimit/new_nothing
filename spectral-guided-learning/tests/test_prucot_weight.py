import torch

from prucot_weight import (
    best_epoch_weights,
    build_labels,
    build_replacement_mask,
    fuse_embeddings,
    probe_scope,
)


def test_build_replacement_mask_uses_step_weight_over_its_span():
    weights = torch.tensor([0.2, 0.8])
    mask = build_replacement_mask(6, [(0, 3), (3, 6)], weights, device="cpu")

    assert mask.shape == (1, 6, 1)
    assert torch.allclose(mask[0, 0:3, 0], torch.full((3,), 0.2))
    assert torch.allclose(mask[0, 3:6, 0], torch.full((3,), 0.8))


def test_build_replacement_mask_leaves_gaps_at_one():
    """Positions outside every step span (the prompt) are never replaced by noise."""
    weights = torch.tensor([0.0])
    mask = build_replacement_mask(5, [(2, 4)], weights, device="cpu")

    assert mask[0, 0, 0] == 1.0
    assert mask[0, 1, 0] == 1.0
    assert mask[0, 4, 0] == 1.0


def test_build_replacement_mask_gradient_flows_to_weights():
    """The in-place slice-assignment trick must stay differentiable w.r.t. the step weights."""
    weights = torch.tensor([0.5, 0.5], requires_grad=True)
    mask = build_replacement_mask(4, [(0, 2), (2, 4)], weights, device="cpu")

    mask.sum().backward()

    assert weights.grad is not None
    assert torch.allclose(weights.grad, torch.tensor([2.0, 2.0]))


def test_fuse_embeddings_recovers_original_when_weight_is_one():
    embeddings = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    noise = torch.tensor([[9.0, 9.0]])
    mask = torch.tensor([[[1.0], [1.0]]])

    fused = fuse_embeddings(embeddings, mask, noise)

    assert torch.allclose(fused, embeddings)


def test_fuse_embeddings_recovers_noise_when_weight_is_zero():
    embeddings = torch.tensor([[[1.0, 2.0]]])
    noise = torch.tensor([[9.0, 9.0]])
    mask = torch.tensor([[[0.0]]])

    fused = fuse_embeddings(embeddings, mask, noise)

    assert torch.allclose(fused, noise)


def test_fuse_embeddings_interpolates_at_intermediate_weight():
    embeddings = torch.tensor([[[0.0, 0.0]]])
    noise = torch.tensor([[10.0, 10.0]])
    mask = torch.tensor([[[0.3]]])

    fused = fuse_embeddings(embeddings, mask, noise)

    assert torch.allclose(fused, torch.tensor([[[7.0, 7.0]]]))


def test_best_epoch_weights_picks_lowest_loss_epoch():
    weight_history = [torch.tensor([0.9]), torch.tensor([0.1]), torch.tensor([0.5])]
    loss_history = [2.0, 0.3, 1.0]

    assert torch.equal(best_epoch_weights(loss_history, weight_history), torch.tensor([0.1]))


def test_build_labels_masks_outside_response_span():
    input_ids = torch.tensor([[10, 11, 12, 13, 14]])

    labels = build_labels(5, (2, 4), input_ids)

    assert labels.tolist() == [[-100, -100, 12, 13, -100]]


def test_probe_scope_scores_only_steps_before_the_tag_and_supervises_only_the_solution():
    """Paper Eq. 2 / cot_weight.py's prepare_data_with_indices: chat AND think labels are both
    -100, only the solution is supervised -- and only the CoT steps before it are scored."""
    step_spans = [(10, 20), (20, 35), (35, 50), (50, 60)]
    step_texts = ["First. ", "Second. ", "Ends with </think> tag here. ", "Solution text."]

    cot_step_spans, supervised_span = probe_scope(step_spans, step_texts, response_token_span=(10, 60))

    assert cot_step_spans == [(10, 20), (20, 35)]
    assert supervised_span == (35, 60)


def test_probe_scope_falls_back_to_whole_response_without_a_tag():
    step_spans = [(10, 20), (20, 35)]
    step_texts = ["First. ", "Second, no closing tag."]

    cot_step_spans, supervised_span = probe_scope(step_spans, step_texts, response_token_span=(10, 35))

    assert cot_step_spans == step_spans
    assert supervised_span == (10, 35)
