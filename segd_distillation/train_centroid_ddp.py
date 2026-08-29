import os
import sys
from dataclasses import asdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import HfArgumentParser

from src.arguments import DataArguments, ModelArguments, TrainingArguments
from src.centroid import CentroidArguments, SinkhornCentroidDescriptor
from src.centroid.utils import (
    contrastive_loss_and_accuracy,
    encode_centroid_descriptor,
    freeze_module,
    make_runtime_data_args,
    make_teacher_model_args,
    move_to_device,
)
from src.model.model import MMEBModel
from src.model.processor import load_processor
from src.single_wrapper import SingleCollator, SingleDataset
from src.utils import print_rank


def ddp_setup():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_count = torch.cuda.device_count()
    if local_rank >= device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {device_count} CUDA device(s) are visible. "
            "Lower --nproc_per_node/NUM_GPUS_PER_NODE or set CUDA_VISIBLE_DEVICES to expose more GPUs."
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def load_teacher(model_args, device):
    teacher_args = make_teacher_model_args(model_args)
    teacher = MMEBModel.load(teacher_args, is_trainable=False).to(device)
    freeze_module(teacher)
    processor = load_processor(teacher_args, None)
    tokenizer = getattr(processor, "tokenizer", None)
    return teacher, teacher_args, processor, tokenizer


def get_hidden_dim(model_args, teacher):
    for attr in ("hidden_size",):
        value = getattr(teacher.config, attr, None)
        if value is not None:
            return value
    for attr in ("text_config", "language_config"):
        config = getattr(teacher.config, attr, None)
        value = getattr(config, "hidden_size", None)
        if value is not None:
            return value
    return model_args.teacher_hidden_dim


def build_dataloader(data_args, model_args, processor, training_args, shuffle, drop_last):
    runtime_data_args = make_runtime_data_args(data_args)
    dataset = SingleDataset(runtime_data_args, model_args)
    sampler = DistributedSampler(dataset, shuffle=shuffle, seed=training_args.seed)
    collator = SingleCollator(
        processor=processor,
        model_args=model_args,
        data_args=runtime_data_args,
        training_args=training_args,
    )
    return DataLoader(
        dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=collator,
        drop_last=drop_last,
        pin_memory=False,
    )


def save_checkpoint(path, centroid_head, centroid_args, model_args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    module = centroid_head.module if isinstance(centroid_head, DDP) else centroid_head
    torch.save(
        {
            "model": module.state_dict(),
            "centroid_args": asdict(centroid_args),
            "teacher_model_name": model_args.teacher_model_name or model_args.model_name,
            "input_dim": module.input_dim,
            "hidden_dim": module.hidden_dim,
            "descriptor_dim": module.descriptor_dim,
        },
        path,
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, CentroidArguments))
    model_args, data_args, training_args, centroid_args = parser.parse_args_into_dataclasses()

    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    teacher, teacher_args, processor, tokenizer = load_teacher(model_args, device)
    train_loader = build_dataloader(data_args, teacher_args, processor, training_args, shuffle=True, drop_last=True)

    centroid_head = SinkhornCentroidDescriptor(
        input_dim=get_hidden_dim(model_args, teacher),
        hidden_dim=centroid_args.centroid_hidden_dim,
        num_centroids=centroid_args.num_centroids,
        centroid_dim=centroid_args.centroid_dim,
        sinkhorn_epsilon=centroid_args.sinkhorn_epsilon,
        sinkhorn_iters=centroid_args.sinkhorn_iters,
        dustbin_mass=centroid_args.dustbin_mass,
    ).to(device)
    if centroid_args.centroid_checkpoint:
        state = torch.load(centroid_args.centroid_checkpoint, map_location="cpu")
        centroid_head.load_state_dict(state["model"])

    centroid_head = DDP(centroid_head, device_ids=[device.index])
    optimizer = AdamW(
        centroid_head.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    total_steps = (
        len(train_loader.dataset)
        // (training_args.per_device_train_batch_size * dist.get_world_size())
        // training_args.gradient_accumulation_steps
        * training_args.num_train_epochs
    )
    from transformers import get_constant_schedule_with_warmup

    scheduler = get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(training_args.warmup_ratio * total_steps),
    )

    global_step = 0
    for epoch in range(training_args.num_train_epochs):
        train_loader.sampler.set_epoch(epoch)
        centroid_head.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_main_process())
        optimizer.zero_grad()
        for step, batch in enumerate(progress):
            batch = move_to_device(batch, device)
            qry_desc, qry_aux = encode_centroid_descriptor(
                teacher,
                centroid_head,
                batch["qry"],
                tokenizer=tokenizer,
                layer_idx=centroid_args.centroid_layer_idx,
                drop_special_tokens=centroid_args.drop_special_tokens,
            )
            pos_desc, _ = encode_centroid_descriptor(
                teacher,
                centroid_head,
                batch["pos"],
                tokenizer=tokenizer,
                layer_idx=centroid_args.centroid_layer_idx,
                drop_special_tokens=centroid_args.drop_special_tokens,
            )
            loss, acc = contrastive_loss_and_accuracy(
                qry_desc,
                pos_desc,
                centroid_args.centroid_temperature,
            )
            (loss / training_args.gradient_accumulation_steps).backward()

            if (step + 1) % training_args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if is_main_process():
                progress.set_postfix(
                    loss=f"{loss.detach().item():.4f}",
                    acc=f"{acc.detach().item():.4f}",
                    dustbin=f"{qry_aux['dustbin_mass'].mean().item():.3f}",
                )

        if is_main_process() and training_args.save_strategy == "epoch":
            save_checkpoint(
                os.path.join(training_args.output_dir, f"checkpoint-epoch-{epoch}", "centroid.pt"),
                centroid_head,
                centroid_args,
                model_args,
            )

    if is_main_process():
        save_checkpoint(
            os.path.join(training_args.output_dir, "checkpoint-final", "centroid.pt"),
            centroid_head,
            centroid_args,
            model_args,
        )


if __name__ == "__main__":
    ddp_setup()
    main()
    dist.destroy_process_group()
