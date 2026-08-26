import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import HfArgumentParser, set_seed
from transformers.trainer_utils import get_last_checkpoint

from src.arguments import DataArguments, ModelArguments, TrainingArguments
from src.criterions import build_criterion
from src.data.dataset import make_vlm_distill_data_module
from src.distiller import Distiller
from src.model.model import VLMModel
from src.model.processor import load_processor
from src.trainer import DistillTrainer
from src.utils import print_master


logger = logging.getLogger(__name__)

def disable_verbose_logs():
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_VERBOSITY"] = "error"

    for name in [
        "httpcore",
        "httpcore.http11",
        "httpcore.connection",
        "httpx",
        "urllib3",
        "huggingface_hub",
        "transformers",
    ]:
        logger = logging.getLogger(name)
        logger.setLevel(logging.WARNING)
        logger.propagate = False


disable_verbose_logs()


def _has_teacher(model_args: ModelArguments) -> bool:
    return bool(model_args.teacher_model_name)


def _maybe_set_deepspeed_alias(training_args: TrainingArguments) -> None:
    ds_path = training_args.deepspeed_config or training_args.ds_config
    if ds_path and getattr(training_args, "deepspeed", None) in (None, "", False):
        training_args.deepspeed = ds_path


def _resolve_resume_checkpoint(training_args: TrainingArguments) -> Optional[str]:
    resume_from = (training_args.resume_from or "none").lower()
    if resume_from == "none":
        return None

    if resume_from == "auto":
        if not training_args.output_dir:
            return None
        checkpoint = get_last_checkpoint(training_args.output_dir)
        if checkpoint is not None:
            print_master(f"Resuming from latest checkpoint: {checkpoint}")
        return checkpoint

    checkpoint_path = Path(training_args.resume_from)
    if checkpoint_path.exists():
        return str(checkpoint_path)

    if training_args.output_dir:
        nested = Path(training_args.output_dir) / f"checkpoint-{training_args.resume_from}"
        if nested.exists():
            return str(nested)

    raise ValueError(f"Could not resolve resume checkpoint: {training_args.resume_from}")


def _trainer_processor_kwargs(processor: Any) -> Dict[str, Any]:
    signature = inspect.signature(DistillTrainer.__mro__[1].__init__)
    if "processing_class" in signature.parameters:
        return {"processing_class": processor}
    return {"tokenizer": processor}


def build_model_and_processors(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
    if _has_teacher(model_args):
        model = Distiller(model_args, training_args)
        student_processor = model.get_student_processor()
        teacher_processor = model.get_teacher_processor()
        criterion = build_criterion(training_args)
        print_master("Training mode: teacher distillation")
    else:
        model = VLMModel.build(model_args)
        student_processor = load_processor(model_args, data_args)
        teacher_processor = None
        criterion = None
        print_master("Training mode: student SFT without teacher")

    return model, student_processor, teacher_processor, criterion


def log_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    ratio = 100.0 * trainable_params / total_params if total_params else 0.0
    print_master(
        "Trainable parameters: "
        f"{trainable_params:,} / {total_params:,} ({ratio:.4f}%)"
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        for action in parser._actions:
            if action.help:
                action.help = action.help.replace("%", "%%")
        parser.print_help()
        return

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    _maybe_set_deepspeed_alias(training_args)
    if training_args.output_dir is None:
        raise ValueError("--output_dir is required.")
    if not data_args.data_path and not data_args.dataset_name and not data_args.dataset_config:
        raise ValueError("Pass --data_path, --dataset_name, or --dataset_config for training data.")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO if training_args.local_rank in (-1, 0) else logging.WARNING,
    )
    set_seed(training_args.seed)

    model, student_processor, teacher_processor, criterion = build_model_and_processors(
        model_args, data_args, training_args
    )
    log_trainable_parameters(model)
    setattr(data_args, "kd_loss_type", training_args.kd_loss_type)
    data_module = make_vlm_distill_data_module(
        student_processor=student_processor,
        teacher_processor=teacher_processor,
        data_args=data_args,
        model_args=model_args,
    )

    train_dataset = data_module["train_dataset"]
    if len(train_dataset) == 0:
        raise ValueError("The training dataset is empty after filtering missing images/conversations.")
    print_master(f"Train samples: {len(train_dataset)}")

    trainer = DistillTrainer(
        model_args=model_args,
        criterion=criterion,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_module["data_collator"],
        **_trainer_processor_kwargs(student_processor),
    )

    resume_checkpoint = _resolve_resume_checkpoint(training_args)
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(training_args.output_dir)
    trainer.save_state()

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)


if __name__ == "__main__":
    main()
