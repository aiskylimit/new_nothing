import logging
import os
from numbers import Number
from typing import Any, Callable, Dict, Optional

import torch
import torch.distributed as dist
from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_pt_utils import LengthGroupedSampler

from src.arguments import ModelArguments, TrainingArguments
from src.distiller import Distiller
from src.model.model import VLMModel
from src.model.processor import save_processor
from src.utils import print_master


logger = logging.getLogger(__name__)

# Metadata keys that should never be forwarded to the model.
_BATCH_METADATA_KEYS = frozenset(
    {"ids", "image_paths", "texts", "images", "global_dataset_name", "dataset_name"}
)


class DistillTrainer(Trainer):
    """
    HF Trainer subclass that handles two training modes:

    SFT mode (no teacher):
        Pass a ``VLMModel`` as ``model``.  The trainer uses the cross-entropy
        loss computed against the ``labels`` tensor supplied by the collator.
        Labels are built in ``VlmDistillDataCollator`` and mask all non-assistant
        tokens (system prompt, user turns, padding) with -100 so only assistant
        responses contribute to the loss.

    Distillation mode (student + teacher):
        Pass a ``Distiller`` as ``model`` **and** supply a ``criterion``
        callable with signature ``criterion(distiller, batch) -> loss``.
        The distiller owns both student and teacher.  ``student_inputs`` always
        carries a ``labels`` tensor (same masking as SFT) which the criterion
        can use to gate token-level KD loss to assistant positions only.
    """

    def __init__(
        self,
        model_args: ModelArguments,
        criterion: Optional[Callable] = None,
        **kwargs,
    ):
        self.model_args = model_args
        self.criterion = criterion

        model = kwargs.get("model")
        self._is_distillation = isinstance(model, Distiller)

        if self._is_distillation and criterion is None:
            raise ValueError("A criterion callable is required when model is a Distiller.")

        # The collator returns nested dicts; Trainer's column-removal
        # introspects forward() signatures and would silently drop everything.
        args = kwargs.get("args")
        if args is not None:
            args.remove_unused_columns = False

        super().__init__(**kwargs)

        self._is_ddp = dist.is_initialized()
        self._loss_metric_sums: Dict[str, Any] = {}
        self._loss_metric_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Column removal — disabled; our batches are nested dicts
    # ------------------------------------------------------------------

    def _remove_unused_columns(self, dataset, description=None):
        return dataset

    def _get_train_sampler(self, train_dataset=None):
        """Use the dataset's cheap text-length estimates for VLM bucketing."""
        train_dataset = train_dataset if train_dataset is not None else self.train_dataset
        if (
            train_dataset is not None
            and self.args.train_sampling_strategy == "group_by_length"
            and hasattr(train_dataset, "lengths")
        ):
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                lengths=train_dataset.lengths,
            )
        return super()._get_train_sampler(train_dataset)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self._is_distillation:
            # The criterion receives the full batch, which includes
            # student_inputs["labels"] for assistant-position gating.
            loss_output = model(self.criterion, inputs)
            if isinstance(loss_output, dict):
                self._record_loss_metrics(loss_output)
                loss = loss_output["loss"]
            else:
                loss = loss_output

            return (loss, None) if return_outputs else loss

        # SFT mode: forward through student model only.
        student_inputs = inputs.get("student_inputs", inputs)
        model_inputs = self._build_sft_model_inputs(student_inputs)
        outputs = model(**model_inputs)

        loss = outputs.loss
        if loss is None:
            raise RuntimeError(
                "VLMModel did not return a loss. Make sure labels are present in the "
                "batch, or that the backbone computes LM loss internally."
            )
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _build_sft_model_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strip metadata keys and forward the rest to the model.

        ``labels`` is now expected to be present in ``inputs`` — it is built
        by ``VlmDistillDataCollator._build_labels`` at collation time, with
        all non-assistant positions already masked to -100.  We therefore no
        longer synthesise labels here; if they are somehow absent we raise an
        explicit error rather than silently using a wrong label tensor.
        """
        cleaned = {k: v for k, v in inputs.items() if k not in _BATCH_METADATA_KEYS}

        if "labels" not in cleaned:
            raise RuntimeError(
                "No 'labels' key found in student_inputs. "
                "Ensure VlmDistillDataCollator is used and that "
                "_build_labels ran successfully during collation."
            )

        return cleaned

    # ------------------------------------------------------------------
    # Logging — include criterion loss components
    # ------------------------------------------------------------------

    def _record_loss_metrics(self, loss_output: Dict[str, Any]) -> None:
        for name, value in loss_output.items():
            if name == "loss":
                continue
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                # Keep detached scalar accumulators on their original device.
                # Calling .item() here would synchronize CUDA once per metric
                # and micro-batch; conversion is deferred to periodic log().
                scalar = value.detach().float()
            elif isinstance(value, Number):
                scalar = float(value)
            else:
                continue

            current = self._loss_metric_sums.get(name)
            self._loss_metric_sums[name] = scalar if current is None else current + scalar
            self._loss_metric_counts[name] = self._loss_metric_counts.get(name, 0) + 1

    @staticmethod
    def _to_log_scalar(value: Any) -> Optional[float]:
        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            value = value.detach().float()
            if not torch.isfinite(value):
                return None
            return float(value.item())

        if isinstance(value, Number):
            value = float(value)
            if value != value or value in (float("inf"), float("-inf")):
                return None
            return value

        return None

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        if self._loss_metric_sums:
            for name, total in self._loss_metric_sums.items():
                count = self._loss_metric_counts.get(name, 0)
                if count > 0 and name not in logs:
                    scalar = self._to_log_scalar(total / count)
                    if scalar is not None:
                        logs[name] = scalar
            self._loss_metric_sums.clear()
            self._loss_metric_counts.clear()

        return super().log(logs, *args, **kwargs)

    # ------------------------------------------------------------------
    # Optimizer — separate LR for projectors when configured
    # ------------------------------------------------------------------

    def create_optimizer(self):
        """
        Build the optimizer.  When in distillation mode, projector parameters
        are placed in a dedicated param group if ``projector_lr`` differs from
        the base learning rate.
        """
        super().create_optimizer()

        if not (self._is_distillation and self._projector_needs_own_group()):
            return self.optimizer

        projector_ids = {
            id(p)
            for p in self.model.projectors.parameters()
            if p.requires_grad
        }

        for group in self.optimizer.param_groups:
            group["params"] = [p for p in group["params"] if id(p) not in projector_ids]

        proj_lr = self.model_args.projector_lr or self.args.learning_rate
        self.optimizer.add_param_group(
            {
                "params": [p for p in self.model.projectors.parameters() if p.requires_grad],
                "lr": proj_lr,
                # The SRA-v2 reference explicitly excludes its learned hidden
                # projector from weight decay.
                "weight_decay": 0.0,
            }
        )
        print_master(f"Projector param group added to optimizer (lr={proj_lr})")
        return self.optimizer

    def _projector_needs_own_group(self) -> bool:
        if not (hasattr(self.model, "projectors") and self.model.projectors is not None):
            return False
        if not any(p.requires_grad for p in self.model.projectors.parameters()):
            return False
        proj_lr = getattr(self.model_args, "projector_lr", None)
        return proj_lr is not None and proj_lr != self.args.learning_rate

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        os.makedirs(output_dir, exist_ok=True)

        raw_model = self.model
        if hasattr(raw_model, "module"):
            raw_model = raw_model.module

        if self._is_distillation:
            raw_model.student.save(output_dir)
            raw_model.save_projectors(os.path.join(output_dir, "projectors"))
        else:
            raw_model.save(output_dir)

        processing_class = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if processing_class is not None:
            save_processor(processing_class, output_dir, self.model_args.model_backbone)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        print_master(f"Checkpoint saved to {output_dir}")
