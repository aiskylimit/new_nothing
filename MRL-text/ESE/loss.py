import math
import random
import warnings

import torch

from utils.loss import info_nce as pair_info_nce


def resolve_dims(full_dim, dims=None):
    """Keep training dimension order: log weights depend on prefix position."""
    if dims is None:
        return [d for d in (16, 32, 64, 128, 256, 512) if d < full_dim] + [full_dim]
    dims = list(dims)
    if any(isinstance(d, bool) or not isinstance(d, int) or d < 1 for d in dims):
        raise ValueError("Matryoshka dimensions must be positive integers.")
    valid = [d for d in dims if d <= full_dim]
    if not valid:
        raise ValueError("At least one Matryoshka dimension must be <= hidden_size.")
    if len(valid) < len(dims):
        warnings.warn(f"Filtered Matryoshka dimensions from {dims} to {valid} (hidden_size={full_dim}).",
                      stacklevel=2)
    return valid


def pool_hidden(hidden, attention_mask, pooling):
    if pooling == "cls":
        return hidden[:, 0, :].float()
    if pooling != "mean":
        raise ValueError("pooling must be cls or mean.")
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden.float() * mask).sum(1) / mask.sum(1).clamp_min(1e-9)


def info_nce(first, second, temperature):
    return pair_info_nce(first, second, temperature)[0]


def matryoshka_info_nce(first, second, dims, temperature):
    total = first.new_zeros((), dtype=torch.float32)
    metrics = {}
    for index, dim in enumerate(dims):
        loss, logits = pair_info_nce(first[:, :dim], second[:, :dim], temperature)
        weighted = loss / (1.0 + math.log(index + 1))
        total = total + weighted
        metrics[f"loss_dim_{dim}"] = loss.detach()
        metrics[f"weighted_loss_dim_{dim}"] = weighted.detach()
        labels = torch.arange(first.size(0), device=first.device)
        metrics[f"acc_dim_{dim}"] = (logits.detach().argmax(-1) == labels).float().mean()
    return total, metrics


def ese_loss(hidden_first, hidden_second, mask_first, mask_second, *,
             matryoshka_dims=None, temperature=0.07, n_layers_per_step=1,
             use_intermediate_layers=True, pooling="mean"):

    if len(hidden_first) != len(hidden_second) or len(hidden_first) < 2:
        raise ValueError("Expected embedding states followed by at least one Transformer block.")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite.")
    if isinstance(n_layers_per_step, bool) or not isinstance(n_layers_per_step, int):
        raise ValueError("n_layers_per_step must be an integer (<=0 selects all intermediate blocks).")
    dims = resolve_dims(hidden_first[-1].size(-1), matryoshka_dims)
    num_layers = len(hidden_first)
    with torch.autocast(device_type=mask_first.device.type, enabled=False):
        first = pool_hidden(hidden_first[-1], mask_first, pooling)
        second = pool_hidden(hidden_second[-1], mask_second, pooling)
        final_loss, metrics = matryoshka_info_nce(first, second, dims, temperature)
        intermediate_loss = final_loss.new_zeros(())
        layer_indices = []
        if use_intermediate_layers and num_layers > 2:
            layer_indices = list(range(1, num_layers - 1))
            if 0 < n_layers_per_step < len(layer_indices):
                layer_indices = random.sample(layer_indices, n_layers_per_step)
            for layer in layer_indices:
                first = pool_hidden(hidden_first[layer], mask_first, pooling)
                second = pool_hidden(hidden_second[layer], mask_second, pooling)
                layer_loss, _ = matryoshka_info_nce(first, second, dims, temperature)
                intermediate_loss = intermediate_loss + layer_loss / (1.0 + math.log(num_layers - layer))
        total = final_loss + intermediate_loss
    metrics.update(final_loss=final_loss.detach(), intermediate_loss=intermediate_loss.detach(),
                   sampled_intermediate_layers=final_loss.new_tensor(len(layer_indices)))
    return total, metrics
