"""Phase 5: masked supervised fine-tuning (paper Eq. 9).

Shared by both the vanilla and spectral SFT launchers (scripts/qwen3/{sft,spectral}/) --
only the loss mask in the dataset differs. Settings come from CLI flags or --config.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from data_collator import MaskedSFTCollator
from masked_dataset import MaskedSFTDataset, VolumeCommitCallback
from masked_loss import masked_cross_entropy
from training_utils import compensate_global_token_mean, set_training_seed


class MaskedSFTTrainer(Trainer):
    """Trainer applying the selective masked objective instead of the default LM loss.

    `model_accepts_loss_kwargs=True` makes Trainer sum supervised-token counts across a whole
    gradient-accumulation step and pass it as `num_items_in_batch`, used as Z in Eq. 9 --
    averaging per-microbatch losses instead would over-weight sparsely-supervised sequences
    (exactly what spectral masks create).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True
        self._loss_shifts_labels = True  # our loss shifts labels, so count over labels[..., 1:]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        token_weights = inputs.pop("loss_weights", None)
        outputs = model(**inputs)
        loss = masked_cross_entropy(
            outputs.logits, labels, denominator=num_items_in_batch, token_weights=token_weights
        )
        # ``num_items_in_batch`` is a token count gathered across DDP ranks.  Gradients are
        # averaged by DDP, so mirror Trainer.compute_loss's compensation and recover the
        # global sum/Z objective instead of shrinking it by world size.
        loss = compensate_global_token_mean(
            loss,
            average_tokens_across_devices=self.args.average_tokens_across_devices,
            num_items_in_batch=num_items_in_batch,
            num_processes=self.accelerator.num_processes,
        )
        return (loss, outputs) if return_outputs else loss


def build_training_arguments(config: dict) -> TrainingArguments:
    on_gpu = torch.cuda.is_available()  # CPU falls back to fp32 for local smoke runs
    return TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["per_device_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": config["min_learning_rate"]},
        warmup_ratio=config["warmup_ratio"],  # deprecated in transformers>=5 (still functions correctly)
        bf16=on_gpu,
        use_cpu=not on_gpu,
        gradient_checkpointing=on_gpu,
        # Reentrant checkpointing trips DDP's "mark variable ready only once" assertion with
        # LoRA's frozen base params (multi-GPU only); non-reentrant avoids the double-backward.
        gradient_checkpointing_kwargs={"use_reentrant": False} if on_gpu else None,
        logging_steps=config.get("logging_steps", 5),
        # save_strategy: "epoch"/"steps" as usual; "no" skips intermediate checkpoints (final save still runs).
        save_strategy=config.get("save_strategy", "epoch"),
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 6),
        report_to=[],
        seed=config.get("seed", 42),
        remove_unused_columns=False,
        deepspeed=config.get("deepspeed_config"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    # --config is an optional yaml base; CLI flags below override it when both are given.
    parser.add_argument("--config")
    parser.add_argument("--smoke", action="store_true", help="tiny run to validate the setup")
    parser.add_argument("--model-name")
    parser.add_argument("--data-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--per-device-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--attn-implementation")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument(
        "--metrics-log",
        help="write the full training metric history (trainer.state.log_history: per-step loss, "
        "grad_norm, learning_rate, epoch + final summary) to this JSON file after training",
    )
    parser.add_argument(
        "--deepspeed-config",
        help="path to a DeepSpeed json config (e.g. configs/deepspeed/ds_config_zero2_offload.json); "
        "batch size/grad accum/bf16 fields there are \"auto\", filled in from the flags above. "
        "ZeRO-2 + optimizer CPU offload for long-sequence runs that OOM at --per-device-batch-size 1.",
    )
    parser.add_argument(
        "--max-seq-len", type=int,
        help="drop samples with more than this many input_ids before training. Optimizer-state "
        "offload (--deepspeed-config) doesn't reduce per-sample activation memory -- a single very "
        "long sequence at --per-device-batch-size 1 can still OOM regardless. Empirically measured "
        "on a 40GB GPU (Qwen3-1.7B, this LoRA config, gradient checkpointing on): ~31GB peak at "
        "12k tokens, ~40GB (no margin) at 16k, OOM at 20k -- pick a value with headroom for your GPU.",
    )
    # checkpointing
    parser.add_argument("--save-strategy", choices=["epoch", "steps", "no"])
    parser.add_argument("--save-steps", type=int, help="interval when --save-strategy=steps")
    parser.add_argument("--save-total-limit", type=int)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction,
        help="resume from the last checkpoint in --output-dir if one exists (else start fresh); "
        "pair with per-epoch volume commits so an interrupted run continues instead of restarting",
    )
    # LoRA (optional): trains low-rank adapters instead of full weights. Merged back into the
    # base model before saving, so output_dir is a normal full checkpoint either way.
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--lora-merge", action=argparse.BooleanOptionalAction,
        help="merge adapters into the base weights before saving (default true); "
        "--no-lora-merge keeps output_dir as an adapter-only checkpoint (much smaller, "
        "loadable by vLLM's native LoRA support without a second full-weight copy)",
    )
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--lora-target-modules", help="comma-separated module names")
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
        "attn_implementation": args.attn_implementation,
        "seed": args.seed,
        "logging_steps": args.logging_steps,
        "metrics_log": args.metrics_log,
        "deepspeed_config": args.deepspeed_config,
        "max_seq_len": args.max_seq_len,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "resume": args.resume,
        "use_lora": args.use_lora,
        "lora_merge": args.lora_merge,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})

    missing = [key for key in ("model_name", "data_path", "output_dir") if key not in config]
    if missing:
        parser.error(f"missing required settings (pass via --config or CLI flags): {missing}")

    # Trainer sets this later in its constructor, which is too late for LoRA's random
    # adapter initialization.  Do it before loading or wrapping the model.
    set_training_seed(config.get("seed", 42))
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    dataset = MaskedSFTDataset(config["data_path"], max_seq_len=config.get("max_seq_len"))

    if args.smoke:
        dataset.records = dataset.records[:8]
        config["epochs"] = 1

    print(f"{len(dataset)} samples, {dataset.supervised_token_count():,} supervised tokens")

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=config.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = False

    if config.get("use_lora"):
        target_modules = config.get(
            "lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        ).split(",")
        model = get_peft_model(
            model,
            LoraConfig(
                r=config.get("lora_r", 16),
                lora_alpha=config.get("lora_alpha", 32),
                lora_dropout=config.get("lora_dropout", 0.05),
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        # gradient checkpointing backprops through a frozen base model here; without this the
        # graph has no leaf requiring grad and checkpointed activations produce no gradient.
        model.enable_input_require_grads()
        model.print_trainable_parameters()

    # On Modal, commit the volume after each epoch's save so an interrupted run keeps finished epochs.
    callbacks = []
    commit_volume = os.environ.get("MODAL_COMMIT_VOLUME")
    if commit_volume:
        callbacks.append(VolumeCommitCallback(commit_volume))
        print(f"per-epoch volume commits enabled (volume: {commit_volume})")

    trainer = MaskedSFTTrainer(
        model=model,
        args=build_training_arguments(config),
        train_dataset=dataset,
        data_collator=MaskedSFTCollator(pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id),
        # lets Trainer write tokenizer files into every checkpoint-N/, not just output_dir at the
        # end — otherwise intermediate checkpoints aren't loadable by vLLM (evaluate.py) on their own.
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # --resume: pick up the last checkpoint in output_dir if one is present (a re-run after a crash),
    # otherwise start fresh. Guarded so a first run with an empty/absent output_dir doesn't error.
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

    # Persist every logged metric (per-step loss/grad_norm/lr/epoch + the final train_* summary)
    # to a standalone JSON so the full training curve survives independently of the stdout log.
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
                        "use_lora": bool(config.get("use_lora")),
                    },
                    "final_metrics": result.metrics,
                    "log_history": trainer.state.log_history,
                },
                indent=2,
            )
        )
        print(f"metric history ({len(trainer.state.log_history)} entries) -> {metrics_path}")

    if config.get("use_lora") and config.get("lora_merge", True):
        # merge adapters into the base weights so output_dir is a normal full checkpoint,
        # loadable by evaluate.py/vLLM exactly like a non-LoRA run.
        trainer.model.merge_and_unload().save_pretrained(config["output_dir"])
    else:
        # --no-lora-merge: trainer.save_model() on a PEFT model saves adapter weights only
        # (tens-hundreds of MB) instead of a full ~16GB checkpoint; evaluate.py loads it as a
        # LoRA adapter on top of the base model.
        trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    # evidence for the paper's "fewer supervised tokens" claim, recorded next to the checkpoint
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
