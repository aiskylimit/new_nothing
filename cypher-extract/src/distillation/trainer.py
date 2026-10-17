from __future__ import annotations

import json
import os
import platform
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset as HFDataset
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from torch.utils.data import DataLoader, IterableDataset
from transformers import TrainerCallback
from transformers.trainer_utils import seed_worker

from schema_grounding.selector_labels import parse_selector_label

from .arguments import DistillationArguments
from .distillm import AdaptiveRolloutScheduler, ReplayBuffer, RolloutSource, StudentRolloutGenerator
from .fdd import causal_response_mask, fdd_loss
from .generation import resolve_eos_token_ids, selector_generation_kwargs
from .losses import causal_lm_loss, compute_distillation_loss, compute_hpd_loss
from .metrics import decode_task_outputs
from .prepare_data import LAYOUT_FILE
from .resume import (
    full_model_checkpoint_identity,
    validate_fresh_output_dir,
    validate_resume_checkpoint,
    write_resume_manifest,
)
from .sampling import TemplateDistributedBatchSampler, selector_droppable_batch_range
from .task_balancing import task_balanced_loss
from .utils import print_rank

DISTILLM_STATE_NAME = "distillm_state"
_RUNTIME_PACKAGES = ("torch", "transformers", "accelerate", "datasets", "peft", "llamafactory", "deepspeed")


def _unwrap_model(model: torch.nn.Module, accelerator: Any) -> torch.nn.Module:
    try:
        return accelerator.unwrap_model(model)
    except Exception:
        return getattr(model, "module", model)


def _lm_head(model: torch.nn.Module, accelerator: Any) -> torch.nn.Module:
    unwrapped = _unwrap_model(model, accelerator)
    output_embeddings = unwrapped.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError(f"{type(unwrapped).__name__} does not expose output embeddings required by FDD.")
    return output_embeddings


def _dataset_identity(dataset: Any) -> Any:
    if isinstance(dataset, Mapping):
        return {str(key): _dataset_identity(value) for key, value in sorted(dataset.items())}
    identity = {"type": type(dataset).__name__}
    fingerprint = getattr(dataset, "_fingerprint", None)
    if fingerprint is not None:
        identity["fingerprint"] = str(fingerprint)
    try:
        identity["rows"] = len(dataset)
    except TypeError:
        identity["rows"] = None
    return identity


def _model_identity(model: torch.nn.Module | None, accelerator: Any) -> dict[str, Any] | None:
    if model is None:
        return None
    config = getattr(_unwrap_model(model, accelerator), "config", None)
    if config is None:
        return {"type": type(model).__name__}
    return {
        "type": type(config).__name__,
        "name_or_path": getattr(config, "_name_or_path", None),
        "commit_hash": getattr(config, "_commit_hash", None),
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_identity(value: str | None) -> list[dict[str, Any]]:
    if not value:
        return []
    identities = []
    for item in (part.strip() for part in value.split(",")):
        path = Path(item).expanduser().resolve()
        identity: dict[str, Any] = {"source": item}
        if path.is_dir():
            files = []
            for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
                candidate = path / name
                if candidate.is_file():
                    files.append({"name": name, "size": candidate.stat().st_size, "sha256": _file_sha256(candidate)})
            identity.update(resolved_path=str(path), files=files)
        identities.append(identity)
    return identities


def _tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    chat_template = getattr(tokenizer, "chat_template", None)
    return {
        "type": type(tokenizer).__name__,
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "commit_hash": init_kwargs.get("_commit_hash") if isinstance(init_kwargs, Mapping) else None,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "chat_template_sha256": sha256(str(chat_template).encode("utf-8")).hexdigest(),
    }


def _config_file_identity(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        return value
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        return {"source": value}
    return {"source": value, "resolved_path": str(path), "sha256": _file_sha256(path)}


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for package in _RUNTIME_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


class _CheckpointStateCallback(TrainerCallback):
    def __init__(self, trainer: KDTrainer) -> None:
        self.trainer = trainer

    def on_save(self, args, state, control, **kwargs):
        del control, kwargs
        checkpoint = Path(args.output_dir, f"checkpoint-{state.global_step}")
        self.trainer.save_resume_manifest(checkpoint, state.global_step)
        self.trainer.save_distillm_state(checkpoint)


class KDTrainer(CustomSeq2SeqTrainer):
    """Shared LlamaFactory trainer for SFT and teacher-based KD baselines."""

    def __init__(
        self,
        *args,
        distillation_args: DistillationArguments,
        data_args: Any,
        generating_args: Any,
        **kwargs,
    ) -> None:
        self.distillation_args = distillation_args
        self.data_args = data_args
        self.generating_args = generating_args
        super().__init__(*args, **kwargs)
        # This trainer consumes and reduces every custom loss itself and does
        # not use Transformers' num_items_in_batch normalization contract.
        # Qwen/Llama forward methods expose **kwargs, which otherwise makes
        # Trainer assume that contract is implemented and skip division by
        # the active gradient-accumulation window.
        self.model_accepts_loss_kwargs = False
        if self.distillation_args.uses_kd and self.ref_model is None:
            raise ValueError("Knowledge distillation requires LlamaFactory ref_model.")

        self.replay_buffer: ReplayBuffer | None = None
        self.rollout_scheduler: AdaptiveRolloutScheduler | None = None
        self.rollout_generator: StudentRolloutGenerator | None = None
        self._stored_hpd_metrics: defaultdict[str, list[float]] = defaultdict(list)
        self._stored_loss_metrics: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
        self._rollout_counts = {source.value: 0 for source in RolloutSource}
        self._last_eval_artifact_dir: Path | None = None
        if self.distillation_args.is_adaptive:
            rank_seed = int(self.args.seed) + int(self.args.process_index)
            self.replay_buffer = ReplayBuffer(self.distillation_args.capacity, seed=rank_seed)
            self.rollout_scheduler = AdaptiveRolloutScheduler(
                threshold=self.distillation_args.init_threshold,
                loss_eps=self.distillation_args.loss_eps,
                replay_ratio=self.distillation_args.replay_ratio,
                seed=rank_seed,
            )
            self.rollout_generator = StudentRolloutGenerator(
                tokenizer=self.processing_class,
                cutoff_len=self.data_args.cutoff_len,
                rollout_context_length=self.distillation_args.rollout_context_length,
                do_sample=self.generating_args.do_sample,
                top_p=self.distillation_args.gen_top_p,
                top_k=self.generating_args.top_k,
                temperature=self.generating_args.temperature,
                repetition_penalty=self.generating_args.repetition_penalty,
                eos_token_ids=resolve_eos_token_ids(
                    self.processing_class,
                    getattr(self.model, "generation_config", None),
                ),
            )
        self._resume_runtime_identity = {
            "packages": _package_versions(),
            "python": platform.python_version(),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            "train_dataset": _dataset_identity(self.train_dataset),
            "eval_dataset": _dataset_identity(self.eval_dataset),
            "student_model": _model_identity(self.model, self.accelerator),
            "teacher_model": _model_identity(self.ref_model, self.accelerator),
            "teacher_checkpoint": full_model_checkpoint_identity(
                getattr(self.finetuning_args, "ref_model", None)
            ),
            "teacher_adapters": _adapter_identity(getattr(self.finetuning_args, "ref_model_adapters", None)),
            "tokenizer": _tokenizer_identity(self.processing_class),
            "deepspeed_config": _config_file_identity(self.args.deepspeed),
        }
        self.add_callback(_CheckpointStateCallback(self))

    def get_train_dataloader(self) -> DataLoader:
        """Build distributed batches while preserving prepared task mixtures."""

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if isinstance(self.train_dataset, IterableDataset):
            raise ValueError("Template-parity sampling requires a sized training dataset.")
        if self.args.dataloader_drop_last:
            raise ValueError("Set dataloader_drop_last=false; the template drops only at the distributed sampler.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if isinstance(train_dataset, HFDataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="Training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="Training")

        droppable_batch_start = 0
        droppable_batch_count = 0
        layout_path = Path(self.data_args.dataset_dir, LAYOUT_FILE)
        if layout_path.is_file():
            layout = json.loads(layout_path.read_text(encoding="utf-8"))
            prepared_batch_size = int(layout["batch_size"])
            if prepared_batch_size != self._train_batch_size:
                raise ValueError(
                    f"Prepared batch size is {prepared_batch_size}, but training uses "
                    f"{self._train_batch_size}. Regenerate the prepared data."
                )
            contrast_pairs = int(layout["train_selector_contrast_pairs"])
            droppable_batch_start, droppable_batch_count = selector_droppable_batch_range(
                batch_size=prepared_batch_size,
                contrast_pairs=contrast_pairs,
                unpaired_negatives=int(layout["train_selector_unpaired_negatives"]),
                total_rows=int(layout["train_rows"]),
            )

        batch_sampler = TemplateDistributedBatchSampler(
            train_dataset,
            batch_size=self._train_batch_size,
            num_replicas=self.accelerator.num_processes,
            # Match the template's fixed sampler seed while shuffling whole batches.
            seed=0,
            droppable_batch_start=droppable_batch_start,
            droppable_batch_count=droppable_batch_count,
        )
        should_fork = torch.backends.mps.is_available() and self.args.dataloader_num_workers > 1
        dataloader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor,
            multiprocessing_context="fork" if should_fork else None,
            worker_init_fn=partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            ),
        )

        previous_even_batches = self.accelerator.even_batches
        self.accelerator.even_batches = False
        try:
            return self.accelerator.prepare_data_loader(dataloader)
        finally:
            self.accelerator.even_batches = previous_even_batches

    def _state_file(self, directory: str | os.PathLike[str]) -> Path:
        return Path(directory, f"{DISTILLM_STATE_NAME}_rank{self.args.process_index}.pt")

    def save_resume_manifest(self, directory: str | os.PathLike[str], global_step: int) -> None:
        resume_config = getattr(self.args, "_cypher_resume_config", None)
        if not self.args.should_save or resume_config is None:
            return
        write_resume_manifest(
            directory,
            config=resume_config,
            runtime_identity=self._resume_runtime_identity,
            world_size=int(self.args.world_size),
            global_step=global_step,
        )

    def save_distillm_state(self, directory: str | os.PathLike[str]) -> None:
        if self.rollout_scheduler is None or self.replay_buffer is None:
            return
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "scheduler": self.rollout_scheduler.state_dict(),
                "replay_buffer": self.replay_buffer.state_dict(),
                "rollout_counts": self._rollout_counts,
            },
            self._state_file(directory),
        )

    def load_distillm_state(self, directory: str | os.PathLike[str]) -> bool:
        if self.rollout_scheduler is None or self.replay_buffer is None:
            return False
        state_file = self._state_file(directory)
        if not state_file.exists():
            return False
        state = torch.load(state_file, map_location="cpu", weights_only=False)
        self.rollout_scheduler.load_state_dict(state["scheduler"])
        self.replay_buffer.load_state_dict(state["replay_buffer"])
        self._rollout_counts.update(state.get("rollout_counts", {}))
        return True

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        checkpoint_path = validate_resume_checkpoint(
            resume_from_checkpoint,
            output_dir=self.args.output_dir,
            world_size=int(self.args.world_size),
            deepspeed_enabled=self.is_deepspeed_enabled,
            require_distillm_state=self.distillation_args.is_adaptive,
            train_batch_size=int(self._train_batch_size),
            expected_config=getattr(self.args, "_cypher_resume_config", None),
            expected_runtime=self._resume_runtime_identity,
        )
        if checkpoint_path is not None:
            if self.distillation_args.is_adaptive and not self.load_distillm_state(checkpoint_path):
                raise RuntimeError(f"Failed to restore adaptive DistiLLM state from {checkpoint_path}.")
            resume_from_checkpoint = str(checkpoint_path)
        else:
            validate_fresh_output_dir(self.args.output_dir)
        return super().train(resume_from_checkpoint=resume_from_checkpoint, *args, **kwargs)

    def save_state(self) -> None:
        super().save_state()
        self.save_distillm_state(self.args.output_dir)

    def _training_progress(self) -> float:
        max_steps = max(int(getattr(self.state, "max_steps", 0)), 1)
        # The template numbers the first optimizer step as one.
        return min(max(float(self.state.global_step + 1) / max_steps, 0.0), 1.0)

    def _collate_rollouts(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        batch = self.data_collator(features)
        return self._prepare_inputs(batch)

    def _store_hpd_metrics(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self._stored_hpd_metrics[key].append(float(value))

    def _store_loss_metrics(self, metrics: dict[str, torch.Tensor]) -> None:
        for key, value in metrics.items():
            self._stored_loss_metrics[key].append(value.detach().float())

    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        # Trainer calls log() for train, eval, and summary records. Only attach
        # micro-batch loss components to the matching periodic train record.
        if "loss" in logs and (self._stored_loss_metrics or self._stored_hpd_metrics):
            logs = dict(logs)
            metric_values = []
            metric_keys = []
            for key, values in self._stored_loss_metrics.items():
                metric_keys.append(key)
                metric_values.append(torch.stack(values).mean().to(self.accelerator.device))
            for key, values in self._stored_hpd_metrics.items():
                metric_keys.append(key)
                metric_values.append(torch.tensor(values, dtype=torch.float32, device=self.accelerator.device).mean())
            reduced_values = self.accelerator.reduce(torch.stack(metric_values), "mean").tolist()
            logs.update(dict(zip(metric_keys, reduced_values, strict=True)))
            self._stored_loss_metrics.clear()
            self._stored_hpd_metrics.clear()
        super().log(logs, *args, **kwargs)

    def _maybe_replace_with_student_data(
        self, model: torch.nn.Module, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if (
            not self.distillation_args.is_adaptive
            or not model.training
            or self.rollout_scheduler is None
            or self.replay_buffer is None
            or self.rollout_generator is None
        ):
            return inputs

        batch_size = int(inputs["input_ids"].shape[0])
        source = self.rollout_scheduler.choose(
            progress=self._training_progress(),
            replay_size=len(self.replay_buffer),
            capacity=self.replay_buffer.capacity,
            batch_size=batch_size,
        )
        self._rollout_counts[source.value] += 1

        if source is RolloutSource.FRESH:
            generation_model = _unwrap_model(model, self.accelerator)
            features = self.rollout_generator.generate(generation_model, inputs)
            # A student can emit EOS immediately for every selected span.
            # Keep the original supervised batch in that case rather than
            # collating an empty/all-masked rollout batch.
            if not features:
                return inputs
            self.replay_buffer.extend(features)
            return self._collate_rollouts(features)
        if source is RolloutSource.REPLAY:
            return self._collate_rollouts(self.replay_buffer.sample(batch_size))
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        inputs = self._maybe_replace_with_student_data(model, inputs)
        task_ids = inputs.pop("task_ids", None)
        use_task_normalization = self.distillation_args.uses_task_normalized_loss
        selector_weight = self.distillation_args.selector_loss_weight
        if use_task_normalization and task_ids is None:
            raise ValueError("Task-normalized loss requires task_ids from TaskAwareDataCollator.")
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("KDTrainer requires labels from the LlamaFactory SFT collator.")

        output_hidden_states = self.distillation_args.uses_fdd and model.training
        student_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        if use_task_normalization:
            student_outputs = model(**student_inputs, output_hidden_states=output_hidden_states, use_cache=False)
            lm_loss, _ = task_balanced_loss(
                task_ids,
                selector_weight,
                lambda mask: causal_lm_loss(student_outputs.logits[mask], labels[mask]),
            )
        else:
            student_outputs = model(**inputs, output_hidden_states=output_hidden_states, use_cache=False)
            if student_outputs.loss is None:
                raise ValueError("The student model did not return an LM loss.")
            lm_loss = student_outputs.loss

        if model.training:
            self._store_loss_metrics({"train_lm_loss": lm_loss})

        # SFT is deliberately teacher-free.  This also makes every existing
        # KD method with kd_ratio=0 a true SFT run: no teacher forward, no KD
        # loss dispatch, and no KD-specific data selection.
        if not self.distillation_args.uses_kd:
            return (lm_loss, student_outputs) if return_outputs else lm_loss

        # The reference evaluation loop measures student LM loss only. This
        # value drives DistiLLM's adaptive threshold and does not require a
        # teacher forward pass.
        if not model.training:
            return (lm_loss, student_outputs) if return_outputs else lm_loss

        with torch.no_grad():
            self.ref_model.eval()
            teacher_outputs = self.ref_model(
                **student_inputs,
                output_hidden_states=output_hidden_states,
                use_cache=False,
            )

        hpd_task_metrics: dict[str, dict[str, float]] = {}

        def compute_task_kd(mask: torch.Tensor) -> torch.Tensor:
            if self.distillation_args.base_method == "hpd":
                task_loss, task_metrics = compute_hpd_loss(
                    student_outputs.logits[mask],
                    teacher_outputs.logits[mask],
                    labels[mask],
                    sample_in_fp32=self.distillation_args.hpd_sample_in_fp32,
                )
                task_name = "selector" if torch.equal(mask, task_ids.eq(1)) else "generator"
                hpd_task_metrics[task_name] = task_metrics
                return task_loss
            return compute_distillation_loss(
                self.distillation_args.base_method,
                student_outputs.logits[mask],
                teacher_outputs.logits[mask],
                labels[mask],
                skew_alpha=self.distillation_args.skew_alpha,
                amid_div_name=self.distillation_args.amid_div_name,
                amid_div_order=self.distillation_args.amid_div_order,
                amid_alpha=self.distillation_args.amid_alpha,
                amid_lam=self.distillation_args.amid_lam,
                bdl_lambda=self.distillation_args.bdl_lambda,
            )

        if use_task_normalization:
            kd_loss, _ = task_balanced_loss(task_ids, selector_weight, compute_task_kd)
            for task_name, task_metrics in hpd_task_metrics.items():
                self._store_hpd_metrics({f"{task_name}_{key}": value for key, value in task_metrics.items()})
        elif self.distillation_args.base_method == "hpd":
            kd_loss, hpd_metrics = compute_hpd_loss(
                student_outputs.logits,
                teacher_outputs.logits,
                labels,
                sample_in_fp32=self.distillation_args.hpd_sample_in_fp32,
            )
            self._store_hpd_metrics(hpd_metrics)
        else:
            kd_loss = compute_distillation_loss(
                self.distillation_args.base_method,
                student_outputs.logits,
                teacher_outputs.logits,
                labels,
                skew_alpha=self.distillation_args.skew_alpha,
                amid_div_name=self.distillation_args.amid_div_name,
                amid_div_order=self.distillation_args.amid_div_order,
                amid_alpha=self.distillation_args.amid_alpha,
                amid_lam=self.distillation_args.amid_lam,
                bdl_lambda=self.distillation_args.bdl_lambda,
            )

        if self.distillation_args.uses_fdd:
            if student_outputs.hidden_states is None or teacher_outputs.hidden_states is None:
                raise ValueError("FDD requires output_hidden_states from student and teacher.")
            feature_mask = causal_response_mask(labels, inputs["attention_mask"])
            if use_task_normalization:
                feature_loss, _ = task_balanced_loss(
                    task_ids,
                    selector_weight,
                    lambda mask: fdd_loss(
                        tuple(hidden[mask] for hidden in student_outputs.hidden_states),
                        tuple(hidden[mask] for hidden in teacher_outputs.hidden_states),
                        feature_mask[mask],
                        _lm_head(model, self.accelerator),
                        _lm_head(self.ref_model, self.accelerator),
                        self.distillation_args.student_layer_mapping,
                        self.distillation_args.teacher_layer_mapping,
                    ),
                )
            else:
                feature_loss = fdd_loss(
                    student_outputs.hidden_states,
                    teacher_outputs.hidden_states,
                    feature_mask,
                    _lm_head(model, self.accelerator),
                    _lm_head(self.ref_model, self.accelerator),
                    self.distillation_args.student_layer_mapping,
                    self.distillation_args.teacher_layer_mapping,
                )
            kd_component = 2.0 * (
                (1.0 - self.distillation_args.fdd_weight) * kd_loss + self.distillation_args.fdd_weight * feature_loss
            )
        else:
            kd_component = kd_loss

        loss_metrics = {"train_distill_loss": kd_component}
        if self.distillation_args.uses_fdd:
            loss_metrics.update(
                train_token_kd_loss=kd_loss,
                train_feature_loss=feature_loss,
            )
        self._store_loss_metrics(loss_metrics)

        loss = (1.0 - self.distillation_args.kd_ratio) * lm_loss + self.distillation_args.kd_ratio * kd_component
        return (loss, student_outputs) if return_outputs else loss

    def _eval_artifact_directory(self) -> Path:
        step = int(self.state.global_step)
        if self.is_in_train:
            epoch = float(self.state.epoch or 0.0)
            epoch_label = f"{epoch:.4f}".rstrip("0").rstrip(".")
            run_name = f"epoch-{epoch_label}-step-{step}"
        else:
            run_name = f"final-step-{step}"
        return Path(self.args.output_dir, "eval", run_name)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ):
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        self._last_eval_artifact_dir = None
        if (
            self.distillation_args.save_eval_predictions
            and self.args.should_save
            and output.predictions is not None
            and output.label_ids is not None
        ):
            predictions, references = decode_task_outputs(
                self.processing_class,
                output.predictions,
                output.label_ids,
            )
            artifact_dir = self._eval_artifact_directory()
            artifact_dir.mkdir(parents=True, exist_ok=True)
            with (artifact_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
                for index, (prediction, reference) in enumerate(zip(predictions, references, strict=True)):
                    handle.write(
                        json.dumps(
                            {"index": index, "prediction": prediction, "reference": reference},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            self._last_eval_artifact_dir = artifact_dir
            print_rank(f"Saved eval predictions to {artifact_dir}")
        return output

    def evaluate(self, *args, **kwargs):
        self._last_eval_artifact_dir = None
        original_padding_side = self.processing_class.padding_side
        if self.args.predict_with_generate:
            self.processing_class.padding_side = "left"
        try:
            metrics = super().evaluate(*args, **kwargs)
        finally:
            self.processing_class.padding_side = original_padding_side
        if self.rollout_scheduler is not None and "eval_loss" in metrics:
            changed = False
            if self.is_in_train:
                changed = self.rollout_scheduler.update(float(metrics["eval_loss"]))
            metrics["eval_distillm_threshold"] = self.rollout_scheduler.threshold
            metrics["eval_distillm_threshold_changed"] = float(changed)
            metrics.update({f"eval_rollout_{key}_steps": value for key, value in self._rollout_counts.items()})
            print_rank(
                f"DistiLLM threshold={self.rollout_scheduler.threshold:.2f} after eval_loss={metrics['eval_loss']:.6f}"
            )
        if self._last_eval_artifact_dir is not None:
            (self._last_eval_artifact_dir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return metrics

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
        **gen_kwargs,
    ):
        """Preserve student LM loss while LlamaFactory generates metric predictions."""

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                **gen_kwargs,
            )

        prepared_inputs = self._prepare_inputs(inputs)
        labels = prepared_inputs["labels"]
        attention_mask = prepared_inputs.get("attention_mask")
        prompt_rows = []
        for index in range(labels.size(0)):
            if attention_mask is not None and attention_mask.ndim == 2:
                valid = attention_mask[index].bool()
                input_ids = prepared_inputs["input_ids"][index][valid]
                row_labels = labels[index][valid]
            else:
                input_ids = prepared_inputs["input_ids"][index]
                row_labels = labels[index]
            response_positions = torch.nonzero(row_labels.ne(-100), as_tuple=False).flatten()
            if response_positions.numel() == 0:
                raise ValueError("Generative evaluation requires at least one unmasked response token per row.")
            prompt_rows.append(input_ids[: int(response_positions[0])])

        prompt_length = max(row.numel() for row in prompt_rows)
        pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processing_class.eos_token_id
        prompt_input_ids = prepared_inputs["input_ids"].new_full(
            (len(prompt_rows), prompt_length), pad_token_id
        )
        prompt_attention_mask = prepared_inputs["input_ids"].new_zeros(
            (len(prompt_rows), prompt_length)
        )
        for index, row in enumerate(prompt_rows):
            prompt_input_ids[index, -row.numel() :] = row
            prompt_attention_mask[index, -row.numel() :] = 1

        generation_inputs = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
        }
        selector_rows = []
        generator_rows = []
        for index, row_labels in enumerate(labels):
            response_ids = row_labels[row_labels.ne(-100)].tolist()
            response = self.processing_class.decode(response_ids, skip_special_tokens=True).strip()
            target = selector_rows if parse_selector_label(response) is not None else generator_rows
            target.append(index)

        generated_parts: list[tuple[torch.Tensor, torch.Tensor]] = []
        for row_indices, task_gen_kwargs in (
            (generator_rows, gen_kwargs),
            (selector_rows, {**gen_kwargs, **selector_generation_kwargs()}),
        ):
            if not row_indices:
                continue
            indices = torch.tensor(row_indices, device=prompt_input_ids.device)
            task_inputs = {
                key: value.index_select(0, indices)
                for key, value in generation_inputs.items()
            }
            _, task_tokens, _ = super().prediction_step(
                model,
                task_inputs,
                prediction_loss_only=False,
                ignore_keys=ignore_keys,
                **task_gen_kwargs,
            )
            if task_tokens is None:
                raise RuntimeError("Generative evaluation returned no generated tokens.")
            generated_parts.append((indices, task_tokens))

        generated_width = max(tokens.size(1) for _, tokens in generated_parts)
        generated_tokens = labels.new_full(
            (labels.size(0), generated_width),
            pad_token_id,
        )
        for indices, task_tokens in generated_parts:
            generated_tokens[indices, : task_tokens.size(1)] = task_tokens
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, prepared_inputs).detach().mean()
        return loss, generated_tokens, labels
