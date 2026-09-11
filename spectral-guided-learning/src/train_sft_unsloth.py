"""Spectral masked SFT via Unsloth full fine-tuning, single GPU.

A leaner sibling of `train_sft.py` for the spectral arm only: full-FT (no LoRA), no DeepSpeed,
one GPU. The loss mask is still applied the framework-agnostic way -- the collator sets
`labels=-100` at masked positions -- so the ONLY thing that changes versus `train_sft.py` is who
computes the cross-entropy: here it is the model's own (Unsloth-patched, fused) loss instead of
the custom `masked_cross_entropy`. With a plain `Trainer` on transformers 4.57 that fused loss is
normalized by `num_items_in_batch` summed across the grad-accum step, which equals Eq. 9's Z.
`--force-eq9-denominator` makes that normalization explicit if the equivalence gate shows the
native path does not do it on its own.

    modal run modal_train.py --mode check-unsloth      # verify the stack loads
    modal run modal_train.py --mode eq9-gate           # all-ones equivalence gate
    modal run --detach modal_train.py --mode train-eval-unsloth

Output dir is a normal merged full checkpoint (weights + tokenizer + run-summary.json), loaded by
evaluate.py exactly like a `train_sft.py` checkpoint.
"""

# Unsloth MUST be imported before transformers/torch so its kernel patches take effect.
import unsloth  # noqa: F401  (import-for-side-effects; keep first)
from unsloth import FastLanguageModel

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from transformers import Trainer, TrainingArguments

from data_collator import MaskedSFTCollator
from masked_dataset import MaskedSFTDataset, VolumeCommitCallback


class _Eq9DenominatorTrainer(Trainer):
    """Fallback for `--force-eq9-denominator`.

    Used only if the Phase-3 all-ones gate shows the native path does NOT normalize by
    `num_items_in_batch`. Passes the whole-step supervised-token count straight into the model's
    fused loss so the denominator is Eq. 9's Z, while keeping Unsloth's fused-CE memory benefit
    (the logits are never materialized as a full (B,T,V) tensor -- unlike the old
    `masked_cross_entropy` path). Per-microbatch normalization is deliberately not used: it
    over-weights sparsely-supervised sequences, exactly what spectral masks create.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        outputs = model(**inputs, num_items_in_batch=num_items_in_batch)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def build_training_arguments(config: dict) -> TrainingArguments:
    on_gpu = torch.cuda.is_available()
    return TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["per_device_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": config["min_learning_rate"]},
        warmup_ratio=config["warmup_ratio"],
        bf16=on_gpu,
        use_cpu=not on_gpu,
        # Full-FT has no frozen base params, so reentrant checkpointing is safe here; single GPU
        # means no DDP "mark variable ready once" constraint either. Unsloth still applies its own
        # optimized checkpointing under the hood where it can.
        # LoRA uses Unsloth's own "unsloth" checkpointing (set in get_peft_model); enabling it
        # here as well would wrap the model twice.
        gradient_checkpointing=on_gpu and not config.get("use_lora"),
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if on_gpu and not config.get("use_lora") else None
        ),
        # adamw_torch for 1.5B (trivially fits); adamw_8bit for 7B to keep optimizer state ~14GB
        # (fp32 Adam would be ~56GB) so full-FT fits an 80GB GPU. Unsloth cuts activation memory,
        # not optimizer state, so the optimizer choice is what makes 7B full-FT fit.
        optim=config.get("optim", "adamw_torch"),
        logging_steps=config.get("logging_steps", 5),
        save_strategy=config.get("save_strategy", "epoch"),
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 6),
        report_to=[],
        seed=config.get("seed", 42),
        remove_unused_columns=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--smoke", action="store_true", help="tiny run to validate the setup")
    parser.add_argument(
        "--use-lora", action=argparse.BooleanOptionalAction,
        help="LoRA instead of full fine-tuning (P-ALIGN uses LoRA)",
    )
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--lora-target-modules", help="comma-separated projection names")
    parser.add_argument(
        "--float32-mixed-precision", action=argparse.BooleanOptionalAction,
        help="fp32 master weights with bf16 compute; Unsloth defaults to pure bf16",
    )
    parser.add_argument("--model-name")
    parser.add_argument("--data-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--per-device-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument(
        "--optim",
        help="transformers optimizer name (default adamw_torch); pass adamw_8bit for the 7B runs "
        "so full-FT optimizer state fits an 80GB GPU (needs bitsandbytes in the image)",
    )
    parser.add_argument(
        "--force-eq9-denominator", action="store_true",
        help="explicitly divide the fused loss by num_items_in_batch (Eq. 9 Z). Only needed if the "
        "all-ones equivalence gate shows the native Trainer path does not already do it.",
    )
    parser.add_argument(
        "--metrics-log",
        help="write the full training metric history (trainer.state.log_history) to this JSON file",
    )
    parser.add_argument(
        "--max-seq-len", type=int,
        help="drop samples with more than this many input_ids before training (single-GPU OOM guard)",
    )
    parser.add_argument("--save-strategy", choices=["epoch", "steps", "no"])
    parser.add_argument("--save-steps", type=int, help="interval when --save-strategy=steps")
    parser.add_argument("--save-total-limit", type=int)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction,
        help="resume from the last checkpoint in --output-dir if one exists (else start fresh)",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
    overrides = {
        "model_name": args.model_name,
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
        "logging_steps": args.logging_steps,
        "optim": args.optim,
        "metrics_log": args.metrics_log,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
        "float32_mixed_precision": args.float32_mixed_precision,
        "max_seq_len": args.max_seq_len,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "resume": args.resume,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})

    missing = [key for key in ("model_name", "data_path", "output_dir") if key not in config]
    if missing:
        parser.error(f"missing required settings (pass via --config or CLI flags): {missing}")

    max_seq_len = config.get("max_seq_len") or 32768

    dataset = MaskedSFTDataset(config["data_path"], max_seq_len=config.get("max_seq_len"))
    if args.smoke:
        dataset.records = dataset.records[:8]
        config["epochs"] = 1
    print(f"{len(dataset)} samples, {dataset.supervised_token_count():,} supervised tokens")

    # load_in_4bit=False keeps the base in bf16 (dtype=None lets Unsloth pick bf16 on Ampere+);
    # full_finetuning is the inverse of --use-lora. Returns the tokenizer too -- do not re-load.
    use_lora = bool(config.get("use_lora"))
    load_kwargs = {
        "model_name": config["model_name"],
        "max_seq_length": max_seq_len,
        "dtype": None,
        "load_in_4bit": False,
        "full_finetuning": not use_lora,
    }
    if config.get("float32_mixed_precision"):
        load_kwargs["float32_mixed_precision"] = True
    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)

    if use_lora:
        targets = (config.get("lora_target_modules")
                   or "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
        # "unsloth" checkpointing is Unsloth's own offloading variant; it replaces the
        # TrainingArguments-level gradient_checkpointing, which is disabled below for LoRA.
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.get("lora_r") or 16,
            lora_alpha=config.get("lora_alpha") or 32,
            lora_dropout=config.get("lora_dropout") if config.get("lora_dropout") is not None else 0.05,
            target_modules=[t.strip() for t in targets.split(",") if t.strip()],
            use_gradient_checkpointing="unsloth",
            random_state=config.get("seed", 42),
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"LoRA r={config.get('lora_r') or 16}: {trainable:,} trainable / {total:,} total "
              f"({100*trainable/total:.3f}%)")
    model.config.use_cache = False

    callbacks = []
    commit_volume = os.environ.get("MODAL_COMMIT_VOLUME")
    if commit_volume:
        callbacks.append(VolumeCommitCallback(commit_volume))
        print(f"per-epoch volume commits enabled (volume: {commit_volume})")

    trainer_cls = _Eq9DenominatorTrainer if args.force_eq9_denominator else Trainer
    if args.force_eq9_denominator:
        print("using explicit Eq. 9 denominator (num_items_in_batch) in compute_loss")
    trainer = trainer_cls(
        model=model,
        args=build_training_arguments(config),
        train_dataset=dataset,
        data_collator=MaskedSFTCollator(pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id),
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    resume_from = None
    if config.get("resume"):
        from transformers.trainer_utils import get_last_checkpoint

        out_dir = config["output_dir"]
        last = get_last_checkpoint(out_dir) if os.path.isdir(out_dir) else None
        if last:
            print(f"--resume: continuing from {last}")
            resume_from = last
        else:
            print("--resume: no existing checkpoint found, starting fresh")

    result = trainer.train(resume_from_checkpoint=resume_from)

    if config.get("metrics_log"):
        metrics_path = Path(config["metrics_log"])
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "config": {
                        "model_name": config["model_name"],
                        "data_path": config["data_path"],
                        "epochs": config["epochs"],
                        "learning_rate": config["learning_rate"],
                        "min_learning_rate": config.get("min_learning_rate"),
                        "per_device_batch_size": config["per_device_batch_size"],
                        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
                        "effective_batch_size": config["per_device_batch_size"]
                        * config["gradient_accumulation_steps"],
                        "warmup_ratio": config.get("warmup_ratio"),
                        "seed": config.get("seed", 42),
                        "optim": config.get("optim", "adamw_torch"),
                        "trainer": "unsloth-lora" if config.get("use_lora") else "unsloth-full-ft",
                        "max_seq_len": max_seq_len,
                        "lora": {
                            "r": config.get("lora_r") or 16,
                            "alpha": config.get("lora_alpha") or 32,
                            "dropout": config.get("lora_dropout"),
                            "target_modules": config.get("lora_target_modules"),
                        } if config.get("use_lora") else None,
                        "float32_mixed_precision": bool(config.get("float32_mixed_precision")),
                    },
                    "final_metrics": result.metrics,
                    "log_history": trainer.state.log_history,
                },
                indent=2,
            )
        )
        print(f"metric history ({len(trainer.state.log_history)} entries) -> {metrics_path}")

    # LoRA: save_model writes the adapter only -- evaluate.py loads it with --lora-adapter
    # against --base-model. Full-FT: the same call writes the whole model.
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    (Path(config["output_dir"]) / "run-summary.json").write_text(
        json.dumps(
            {
                "data_path": config["data_path"],
                "samples": len(dataset),
                "supervised_tokens": dataset.supervised_token_count(),
                "epochs": config["epochs"],
                "seed": config.get("seed", 42),
                "train_runtime_s": result.metrics.get("train_runtime"),
                "final_train_loss": result.metrics.get("train_loss"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
