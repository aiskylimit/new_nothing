import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from data_utils.lm_datasets import LMTrainDataset
from distillm.forward import ModelFeatures, forward_features
from distillm.losses import topk_kl_loss
from distillm.trajectory import velocity_loss_from_steps
from utils import print_rank, save_rank, get_rank, all_gather
from training_runtime import (
    configure_training_steps,
    make_loader,
    save_checkpoint,
    setup_model_and_optimizer,
)

COMPONENTS = ("ce", "mag", "gram", "logit")


@dataclass
class DistillationResult:
    loss: torch.Tensor
    components: dict
    counts: dict
    student: ModelFeatures
    teacher: ModelFeatures


def get_teacher_model(args, device):
    if args.model_parallel:
        raise NotImplementedError("does not support model parallelism")

    config = AutoConfig.from_pretrained(args.teacher_model_path)
    config.is_model_parallel = False
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp32:
        dtype = torch.float32
    else:
        dtype = torch.float16

    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_path,
        config=config,
        device_map={"": device},
        torch_dtype=dtype,
    )
    if args.teacher_peft_path is not None:
        from peft import PeftModel

        teacher_model = PeftModel.from_pretrained(teacher_model, args.teacher_peft_path)
        teacher_model = teacher_model.merge_and_unload()

    teacher_model.requires_grad_(False)
    teacher_model.eval()
    parameter_count = sum(
        parameter.numel() for parameter in teacher_model.parameters()
    )
    print_rank(f" > teacher parameters: {parameter_count:,}", flush=True)
    return teacher_model


def prepare_dataset(args, tokenizer, teacher_tokenizer=None):
    """Pair training examples with a teacher; evaluate student-only datasets."""
    datasets = {}
    if args.do_train:
        datasets["train"] = LMTrainDataset(
            args,
            tokenizer,
            args.data_dir,
            "train",
            args.train_num,
            args.train_ratio,
            teacher_tokenizer=teacher_tokenizer,
        )
        datasets["valid"] = LMTrainDataset(
            args,
            tokenizer,
            args.data_dir,
            "valid",
            args.dev_num,
            args.dev_ratio,
            with_teacher=False,
        )
    if args.do_eval:
        datasets["test"] = LMTrainDataset(
            args,
            tokenizer,
            args.data_dir,
            "test",
            args.dev_num,
            args.dev_ratio,
            with_teacher=False,
        )
    return datasets


def require_token_alignment(
    model_batch, no_model_batch, teacher_batch, teacher_metadata
):
    """Reject incompatible tokenizations instead of comparing unrelated positions."""
    for key in ("input_ids", "attention_mask"):
        if not torch.equal(model_batch[key], teacher_batch[key]):
            raise ValueError(
                "Top-k needs identical student/teacher tokenization and length limits; "
                "set --logit-weight 0 to use only trajectory alignment"
            )
    if not torch.equal(no_model_batch["label"], teacher_metadata["label"]):
        raise ValueError("Top-k needs identical response prediction positions")


def get_distil_loss(args, model, teacher_model, model_batch, no_model_batch, teacher_batch, teacher_metadata, sample_mask=None):
    use_logits = args.logit_weight > 0
    if use_logits:
        require_token_alignment(model_batch, no_model_batch, teacher_batch, teacher_metadata)
    teacher_model.eval()

    with torch.no_grad():
        teacher = forward_features(
            teacher_model,
            teacher_batch,
            teacher_metadata,
            pooling=args.step_pooling,
            top_k=args.distill_top_k,
            sample_mask=sample_mask,
        )
    student = forward_features(
        model,
        model_batch,
        no_model_batch,
        pooling=args.step_pooling,
        top_k=args.distill_top_k,
        support=teacher.topk_indices if use_logits else None,
        compute_ce=args.ce_weight > 0,
        sample_mask=sample_mask,
    )
    magnitude, gram = velocity_loss_from_steps(
        student.step_representations,
        teacher.step_representations,
        student.step_mask,
        teacher.step_mask,
        args.magnitude_normalization,
        args.eps,
    )
    logit_loss = student.step_representations.sum() * 0
    if use_logits:
        logit_loss = topk_kl_loss(
            student.topk_logits,
            teacher.topk_logits,
            student.token_mask & teacher.token_mask,
            args.distill_temperature,
        )
    loss = (
        args.ce_weight * student.ce_loss
        + args.mag_weight * magnitude
        + args.gram_weight * gram
        + args.logit_weight * logit_loss
    )
    common_steps = student.step_mask & teacher.step_mask
    velocity_counts = (common_steps[:, 1:] & common_steps[:, :-1]).sum(-1)
    return DistillationResult(
        loss=loss,
        components={
            "ce": student.ce_loss,
            "mag": magnitude,
            "gram": gram,
            "logit": logit_loss,
        },
        counts={
            "ce": student.token_mask.sum(),
            "mag": (velocity_counts > 0).sum(),
            "gram": (velocity_counts > 1).sum(),
            "logit": student.token_mask.sum() if use_logits else 0,
        },
        student=student,
        teacher=teacher,
    )


def accumulate_metrics(totals, result):
    for index, name in enumerate(COMPONENTS):
        count = torch.as_tensor(
            result.counts[name], device=totals.device, dtype=totals.dtype
        )
        totals[index, 0] += result.components[name].detach().double() * count
        totals[index, 1] += count


def report_metrics(args, totals, prefix):
    if dist.is_initialized():
        dist.all_reduce(totals)
    values = (totals[:, 0] / totals[:, 1].clamp_min(1)).tolist()
    metrics = dict(zip(COMPONENTS, values))
    metrics["total"] = sum(
        getattr(args, f"{name}_weight") * metrics[name] for name in COMPONENTS
    )
    losses = " | ".join(f"{name}_loss: {value:.6f}" for name, value in metrics.items())
    log_str = f"{prefix} | {losses}"
    print_rank(log_str, flush=True)
    if not dist.is_initialized() or get_rank() == 0:
        Path(args.save).mkdir(parents=True, exist_ok=True)
    save_rank(log_str, Path(args.save) / "log.txt")
    return metrics


def evaluate(args, tokenizer, model, dataset: LMTrainDataset, split, epoch, device, adaptive_threshold=None):
    """Evaluate student CE and optionally generate and score student responses."""
    if args.model_parallel:
        raise NotImplementedError("does not support model parallelism")
    dp_world_size = dist.get_world_size() if dist.is_initialized() else 1
    dp_rank = get_rank() if dist.is_initialized() else 0
    dp_group = None
    loss_func = nn.CrossEntropyLoss(ignore_index=-100)
    print_rank("dp size", dp_world_size)

    generation_config = None
    if args.eval_gen:
        eos_token_ids = [tokenizer.eos_token_id]
        if args.model_type == "qwen":
            end_of_text = tokenizer.get_vocab().get("<|endoftext|>")
            if end_of_text is not None and end_of_text not in eos_token_ids:
                eos_token_ids.append(end_of_text)
        generation_config = GenerationConfig(
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty or 1.0,
            max_length=args.max_length,
            min_length=None,
            eos_token_id=eos_token_ids,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )

    sampler = DistributedSampler(
        dataset, shuffle=False, drop_last=False,
        rank=dp_rank, num_replicas=dp_world_size,
    )
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size,
        num_workers=args.num_workers, collate_fn=dataset.collate,
    )
    model.eval()
    all_loss = 0.0
    step = 0
    all_response_ids = []

    with torch.no_grad():
        for it, (model_batch, no_model_batch, gen_data, _, _) in enumerate(
            tqdm(dataloader, desc="Evaluating", disable=(dp_rank != 0))
        ):
            print_rank(f"{it}/{len(dataloader)}")
            dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            logits = model(
                **model_batch, return_dict=True, use_cache=False,
                output_hidden_states=False,
            ).logits
            loss = loss_func(
                logits.float().reshape(-1, logits.shape[-1]),
                no_model_batch["label"].reshape(-1),
            )
            del logits

            if args.eval_gen:
                generation_model = model if callable(getattr(model, "generate", None)) else model.module
                gen_out = generation_model.generate(
                    **gen_data,
                    generation_config=generation_config,
                    max_new_tokens=args.max_length - gen_data["input_ids"].size(1),
                    synced_gpus=dp_world_size > 1,
                )
                full_ids = F.pad(
                    gen_out.sequences,
                    (0, args.max_length - gen_out.sequences.shape[1]),
                    value=tokenizer.pad_token_id,
                )
                response_ids = full_ids[:, gen_data["input_ids"].size(1):]
                all_response_ids.append(response_ids)

            if dist.is_initialized():
                dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            all_loss += loss.item() / dp_world_size
            step += 1

    if step == 0:
        raise ValueError(f"{split} contains no evaluation batches")
    if args.eval_gen:
        all_response_ids = torch.cat(all_response_ids, dim=0)
        if dist.is_initialized():
            all_response_ids = all_gather(
                all_response_ids, dim=1, world_size=dp_world_size,
                group=dp_group, op="stack",
            )
            all_response_ids = all_response_ids.reshape(-1, all_response_ids.size(-1))

    avg_loss = all_loss / step
    res = {}
    if dp_rank == 0:
        if args.eval_gen:
            from rouge_metric import compute_metrics

            responses = tokenizer.batch_decode(all_response_ids, skip_special_tokens=True)
            references = dataset.answers
            responses = responses[:len(references)]
            res = compute_metrics(responses, references)
            eval_dir = Path(args.save) / "eval" / str(epoch)
            print_rank(eval_dir)
            eval_dir.mkdir(parents=True, exist_ok=True)
            with (eval_dir / "answers.jsonl").open("w", encoding="utf-8") as handle:
                for response in responses:
                    handle.write(json.dumps({"text": response}, ensure_ascii=False) + "\n")

        log_str = f"{split} | avg_loss: {avg_loss} | {res}"
        if "adaptive" in args.type:
            log_str += f" | threshold: {adaptive_threshold}"
        print_rank(log_str)
        Path(args.save).mkdir(parents=True, exist_ok=True)
        save_rank(log_str, Path(args.save) / "log.txt")
    return res

def finetune(args,tokenizer: AutoTokenizer, model, optimizer, lr_scheduler, dataset, device, teacher_model=None,):
    if teacher_model is None:
        raise ValueError("Distillation requires a teacher model")
    print_rank("Start fine-tuning", flush=True)
    train_dataloader = make_loader(args, dataset["train"], training=True)
    global_step = 0
    last_saved_step = None
    optimizer_steps_since_log = 0
    elapsed_since_log = 0.0
    totals = torch.zeros(len(COMPONENTS), 2, dtype=torch.float64, device=device)

    evaluate(args, tokenizer, model, dataset["valid"], "valid", 0, device)
    
    for epoch in range(args.epochs):
        train_dataloader.sampler.set_epoch(epoch)
        model.train()
        max_micro_batches = (
            args.train_iters_per_epoch * args.gradient_accumulation_steps
        )

        for micro_step, batch in enumerate(train_dataloader):
            if micro_step >= max_micro_batches:
                break

            started_at = time.perf_counter()
            model_batch, no_model_batch, _, teacher_batch, teacher_metadata = batch
            train_dataloader.dataset.move_to_device(model_batch, no_model_batch, None, device,)
            train_dataloader.dataset.move_to_device(teacher_batch, teacher_metadata, None, device)

            result = get_distil_loss(args, model, teacher_model, model_batch, no_model_batch, teacher_batch, teacher_metadata)
            loss = result.loss
            model.backward(loss)
            accumulation_boundary = model.is_gradient_accumulation_boundary()
            model.step()
            accumulate_metrics(totals, result)
            elapsed_since_log += time.perf_counter() - started_at
            del loss, result

            if not accumulation_boundary:
                continue

            global_step += 1
            optimizer_steps_since_log += 1
            should_log = (
                global_step % args.log_interval == 0 or global_step == args.total_iters
            )
            if should_log:
                step_time = elapsed_since_log / max(optimizer_steps_since_log, 1)
                report_metrics(
                    args,
                    totals,
                    f"train | epoch: {epoch + 1:3d} | global step: "
                    f"{global_step:6d}/{args.total_iters:6d} | "
                    f"lr: {lr_scheduler.get_last_lr()[0]:.4e} | "
                    f"scale: {getattr(optimizer, 'cur_scale', 1.0):.4f} | "
                    f"step time: {step_time:.3f}",
                )
                totals.zero_()
                optimizer_steps_since_log = 0
                elapsed_since_log = 0.0

            if args.save_interval and global_step % args.save_interval == 0:
                save_checkpoint(args, tokenizer, model, global_step)
                last_saved_step = global_step

            if args.eval_interval and global_step % args.eval_interval == 0:
                evaluate(args, tokenizer, model, dataset["valid"], "valid", global_step, device)
                model.train()

            if global_step >= args.total_iters:
                break

        if global_step >= args.total_iters:
            break

    if last_saved_step != global_step:
        save_checkpoint(args, tokenizer, model, global_step)
    return global_step


def validate_args(args):
    if args.type != "rvd":
        raise ValueError("finetune.py supports RVD only; run CoT SFT with sft.py")
    if not args.do_train and not args.do_eval:
        raise ValueError("Specify --do-train and/or --do-eval")
    if not args.model_path or not args.data_dir or not args.save:
        raise ValueError("Provide --model-path, --data-dir and --save")
    if args.do_train and not args.teacher_model_path:
        raise ValueError("Provide --teacher-model-path for distillation training")
    if args.model_parallel:
        raise ValueError("RVD does not support --model-parallel")
    if args.lm_data_dir or args.student_gen:
        raise ValueError(
            "RVD uses paired training forwards; omit --lm-data-dir and --student-gen"
        )
    if not args.deepspeed or not args.deepspeed_config:
        raise ValueError("Provide --deepspeed and --deepspeed_config")
    if any(
        value < 1
        for value in (
            args.batch_size,
            args.eval_batch_size,
            args.gradient_accumulation_steps,
            args.log_interval,
        )
    ):
        raise ValueError("Batch sizes, accumulation and log interval must be positive")
    if args.save_interval < -1 or args.eval_interval < -1:
        raise ValueError("Save/eval intervals must be -1, 0, or positive")

    if args.do_train:
        weights = tuple(getattr(args, f"{name}_weight") for name in COMPONENTS)
        if (
            any(not math.isfinite(weight) or weight < 0 for weight in weights)
            or sum(weights) == 0
        ):
            raise ValueError(
                "RVD loss weights must be finite, nonnegative, and not all zero"
            )
        if args.distill_top_k < 1 or (args.logit_weight > 0 and args.distill_top_k < 2):
            raise ValueError("--distill-top-k must be at least 2 when top-k KL is enabled")
        if not math.isfinite(args.distill_temperature) or args.distill_temperature <= 0:
            raise ValueError("--distill-temperature must be finite and positive")
        if not math.isfinite(args.eps) or args.eps <= 0:
            raise ValueError("--eps must be finite and positive")
    if not 0 < args.max_prompt_length < args.max_length:
        raise ValueError("Require 0 < --max-prompt-length < --max-length")
    if args.do_train and not 0 < args.t_max_prompt_length < args.t_max_length:
        raise ValueError("Require 0 < --t-max-prompt-length < --t-max-length")

    if args.do_train:
        if args.lr is None or not math.isfinite(args.lr) or args.lr <= 0:
            raise ValueError("Provide a finite positive --lr")
        if args.epochs is None and args.total_iters is None:
            args.epochs = 3
        if any(
            value is not None and value < 1 for value in (args.epochs, args.total_iters)
        ):
            raise ValueError("Epochs and total iterations must be positive")


def main():
    from arguments import get_args
    from utils import get_tokenizer, initialize, print_args

    args = get_args(default_type="rvd")
    validate_args(args)

    with open(args.deepspeed_config, encoding="utf-8") as handle:
        ds_config = json.load(handle)
    if ds_config.get("zero_optimization", {}).get("stage", 0) > 2:
        raise ValueError("HF/PEFT checkpoint saving supports ZeRO stages 0, 1 and 2")
    ds_config.update(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        train_micro_batch_size_per_gpu=args.batch_size,
        gradient_clipping=args.clip_grad,
        steps_per_print=1000,
    )
    if not args.do_train:
        ds_config.setdefault("zero_optimization", {})["stage"] = 0

    fp16_enabled = ds_config.get("fp16", {}).get("enabled", False)
    bf16_enabled = ds_config.get("bf16", {}).get("enabled", False)
    args.fp32 = not fp16_enabled and not bf16_enabled
    args.bf16 = bf16_enabled
    initialize(args)

    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "=" * 30 + f" EXP at {cur_time} " + "=" * 30,
              Path(args.save) / "log.txt")

    device = torch.device("cuda", torch.cuda.current_device())
    tokenizer = get_tokenizer(args)
    teacher_tokenizer = None
    if args.do_train:
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            args.teacher_model_path,
            padding_side="right",
        )
        if teacher_tokenizer.eos_token_id is None:
            raise ValueError("Teacher tokenizer must define eos_token_id")
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        if (
            args.logit_weight > 0
            and tokenizer.get_vocab() != teacher_tokenizer.get_vocab()
        ):
            raise ValueError(
                "Top-k KL requires identical vocabulary/token-ID mappings; "
                "set --logit-weight 0 for different tokenizers"
            )

    datasets = prepare_dataset(args, tokenizer, teacher_tokenizer)
    if args.do_train:
        configure_training_steps(
            args, len(make_loader(args, datasets["train"], training=True))
        )
        print_rank("Train iters per epoch", args.train_iters_per_epoch)
        print_rank("total_iters", args.total_iters)

    if get_rank() == 0:
        print_args(args)
        for name, payload in (("args", vars(args)), ("ds_config", ds_config)):
            destination = Path(args.save) / f"{name}.json"
            with destination.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)

    # config_params supplies the DeepSpeed config directly; clearing the CLI field
    # avoids passing the same configuration through two code paths.
    args.deepspeed_config = None
    model, optimizer, lr_scheduler = setup_model_and_optimizer(
        args,
        ds_config,
        device,
    )
    teacher_model = get_teacher_model(args, device) if args.do_train else None

    global_step = 0
    if args.do_train:
        global_step = finetune(
            args,
            tokenizer,
            model,
            optimizer,
            lr_scheduler,
            datasets,
            device,
            teacher_model=teacher_model,
        )
    if args.do_eval:
        evaluate(
            args,
            tokenizer,
            model,
            datasets["test"],
            "test",
            global_step,
            device,
        )


if __name__ == "__main__":
    main()
