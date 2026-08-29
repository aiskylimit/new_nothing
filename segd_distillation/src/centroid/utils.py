from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.arguments import ModelArguments


def make_teacher_model_args(model_args):
    teacher_args = ModelArguments(
        model_name=model_args.teacher_model_name or model_args.model_name,
        model_type=model_args.model_type,
        processor_name=model_args.processor_name,
        checkpoint_path=getattr(model_args, "teacher_checkpoint_path", None),
        pooling=model_args.teacher_pooling,
        normalize=model_args.teacher_normalize,
        lora=model_args.teacher_lora,
        lora_r=model_args.teacher_lora_r,
        lora_alpha=model_args.teacher_lora_alpha,
        lora_dropout=model_args.teacher_lora_dropout,
        lora_target_modules=model_args.teacher_lora_target_modules,
        model_backbone=model_args.teacher_backbone or model_args.model_backbone,
        student_hidden_dim=model_args.teacher_hidden_dim,
        teacher_hidden_dim=model_args.teacher_hidden_dim,
        modality_gated_pooling=model_args.teacher_modality_gated_pooling,
    )
    teacher_args.temperature = model_args.temperature
    return teacher_args


def make_runtime_data_args(data_args):
    runtime_args = deepcopy(data_args)
    runtime_args.caching_dir = None
    return runtime_args


def freeze_module(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def move_to_device(obj, device):
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        values = [move_to_device(v, device) for v in obj]
        return tuple(values) if isinstance(obj, tuple) else values
    if hasattr(obj, "to") and callable(obj.to):
        return obj.to(device)
    return obj


def get_special_token_mask(inputs, tokenizer):
    input_ids = inputs.get("input_ids")
    if tokenizer is None or input_ids is None:
        return torch.zeros_like(inputs["attention_mask"], dtype=torch.bool)

    eos_ids = tokenizer.eos_token_id
    eos_ids = [] if eos_ids is None else eos_ids if isinstance(eos_ids, list) else [eos_ids]
    special_ids = set(tokenizer.all_special_ids) - set(eos_ids)
    if not special_ids:
        return torch.zeros_like(inputs["attention_mask"], dtype=torch.bool)

    special_ids = torch.tensor(sorted(special_ids), device=input_ids.device, dtype=input_ids.dtype)
    return torch.isin(input_ids, special_ids)


def build_count_token_mask(inputs, hidden_state, image_features, tokenizer):
    attention_mask = inputs["attention_mask"]
    batch_size, hidden_len = hidden_state.shape[:2]
    device = hidden_state.device

    special_mask = get_special_token_mask(inputs, tokenizer)
    token_mask = torch.zeros(batch_size, hidden_len, dtype=torch.bool, device=device)
    left_padding = attention_mask[:, 0].eq(0) & attention_mask[:, -1].eq(1)

    for idx in range(batch_size):
        num_image_tokens = 0
        if image_features is not None and idx < len(image_features) and image_features[idx] is not None:
            num_image_tokens = image_features[idx].size(0)

        sample_text_mask = ~special_mask[idx]
        sample_mask = torch.cat(
            [
                torch.ones(num_image_tokens, dtype=torch.bool, device=device),
                sample_text_mask.to(device),
            ],
            dim=0,
        )

        if sample_mask.numel() == 0:
            raise ValueError("No valid token found for centroid descriptor.")
        if bool(left_padding[idx].item()):
            sample_mask = sample_mask[-hidden_len:]
            token_mask[idx, -sample_mask.numel():] = sample_mask
        else:
            sample_mask = sample_mask[:hidden_len]
            token_mask[idx, :sample_mask.numel()] = sample_mask

    return token_mask


def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous())
    gathered[rank] = tensor
    return torch.cat(gathered, dim=0)


def contrastive_loss_and_accuracy(qry_desc, pos_desc, temperature):
    all_qry = gather_tensor(qry_desc)
    all_pos = gather_tensor(pos_desc)
    logits = torch.matmul(all_qry, all_pos.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=1) == labels).float().mean()
    return loss, acc


def encode_centroid_descriptor(
    teacher,
    centroid_head,
    inputs,
    tokenizer=None,
    layer_idx=-1,
    drop_special_tokens=False,
):
    with torch.no_grad():
        _, image_features, _, hidden_states = teacher.encode_input(inputs)
    token_hidden = hidden_states[layer_idx].detach()

    token_mask = build_count_token_mask(inputs, token_hidden, image_features, tokenizer)
    return centroid_head(token_hidden, token_mask)
