from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from distillation.generation import generation_eos_value, resolve_eos_token_ids
from distillation.utils import capture_rng_state, restore_rng_state
from schema_grounding.inference.prompting import Message, render_llama3, render_qwen3_nothink

_DTYPES = {
    "auto": "auto",
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _pinned_base_revision(checkpoint_path: Path, base_name: str, adapter_revision: str | None) -> str | None:
    """Prefer the immutable base commit recorded when this checkpoint was saved."""

    manifest_path = checkpoint_path / "resume_manifest.json"
    if not manifest_path.is_file():
        return adapter_revision
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid training manifest {manifest_path}: {exc}") from exc
    runtime = manifest.get("runtime") if isinstance(manifest, dict) else None
    student_model = runtime.get("student_model") if isinstance(runtime, dict) else None
    if not isinstance(student_model, dict):
        return adapter_revision
    recorded_name = student_model.get("name_or_path")
    if recorded_name and recorded_name != base_name:
        raise ValueError(
            f"Training manifest base model {recorded_name!r} does not match adapter base model {base_name!r}"
        )
    commit_hash = student_model.get("commit_hash")
    return commit_hash if isinstance(commit_hash, str) and commit_hash else adapter_revision
_TOKENIZER_ASSETS = (
    "tokenizer.json",
    "tokenizer.model",
    "sentencepiece.bpe.model",
    "spiece.model",
    "vocab.json",
)


@dataclass
class ModelRunner:
    model: Any
    tokenizer: Any
    device: torch.device
    model_family: str | None = None
    safe_batch_sizes: dict[int, int] = field(default_factory=dict, repr=False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        dtype: str = "bfloat16",
        device: str = "cuda",
        merge_adapter: bool = True,
        model_family: str | None = None,
    ) -> ModelRunner:
        if dtype not in _DTYPES:
            raise ValueError(f"Unsupported dtype {dtype!r}; choose from {', '.join(_DTYPES)}")
        checkpoint_directory = Path(checkpoint_path)
        checkpoint_source = str(checkpoint_directory)
        if not (checkpoint_directory / "adapter_config.json").is_file():
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_source, use_fast=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            target_device = torch.device(device)
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_source,
                dtype=_DTYPES[dtype],
                low_cpu_mem_usage=True,
            )
            model.to(target_device)
            model.eval()
            return cls(
                model=model,
                tokenizer=tokenizer,
                device=target_device,
                model_family=model_family,
            )

        peft_config = PeftConfig.from_pretrained(checkpoint_source)
        base_name = peft_config.base_model_name_or_path
        if not base_name:
            raise ValueError(f"Adapter {checkpoint_directory} does not declare a base model")
        base_revision = _pinned_base_revision(checkpoint_directory, base_name, peft_config.revision)
        has_local_tokenizer = (checkpoint_directory / "tokenizer_config.json").is_file() and any(
            (checkpoint_directory / filename).is_file() for filename in _TOKENIZER_ASSETS
        )
        tokenizer_source = checkpoint_source if has_local_tokenizer else base_name
        tokenizer_revision = None if has_local_tokenizer else base_revision
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, revision=tokenizer_revision, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        target_device = torch.device(device)
        model = AutoModelForCausalLM.from_pretrained(
            base_name,
            revision=base_revision,
            dtype=_DTYPES[dtype],
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(model, checkpoint_source)
        if merge_adapter:
            model = model.merge_and_unload()
        model.to(target_device)
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=target_device,
            model_family=model_family,
        )

    @classmethod
    def from_adapter(
        cls,
        adapter_path: str | Path,
        *,
        dtype: str = "bfloat16",
        device: str = "cuda",
        merge_adapter: bool = True,
        model_family: str | None = None,
    ) -> ModelRunner:
        """Backward-compatible adapter entry point."""

        return cls.from_checkpoint(
            adapter_path,
            dtype=dtype,
            device=device,
            merge_adapter=merge_adapter,
            model_family=model_family,
        )

    def prompt_length(self, messages: Sequence[Message]) -> int:
        return len(self._tokenize_chat(messages))

    def _tokenize_chat(self, messages: Sequence[Message]) -> list[int]:
        if self.model_family in {"qwen3", "qwen2.5_coder"}:
            return self.tokenizer.encode(
                render_qwen3_nothink(list(messages)),
                add_special_tokens=False,
            )
        if self.model_family == "llama3":
            bos_token = self.tokenizer.bos_token
            if not isinstance(bos_token, str) or not bos_token:
                raise ValueError("Llama 3 inference requires a tokenizer BOS token.")
            return self.tokenizer.encode(
                render_llama3(list(messages), bos_token=bos_token),
                add_special_tokens=False,
            )
        token_ids = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if hasattr(token_ids, "keys") and "input_ids" in token_ids:
            token_ids = token_ids["input_ids"]
        if not isinstance(token_ids, list):
            token_ids = token_ids.tolist()
        return token_ids

    @torch.inference_mode()
    def generate(
        self,
        conversations: Sequence[Sequence[Message]],
        *,
        max_new_tokens: int,
        do_sample: bool = True,
        temperature: float = 0.5,
        top_p: float = 0.95,
        top_k: int = 0,
        num_beams: int = 1,
    ) -> list[str]:
        if not conversations:
            return []
        safe_batch_size = self.safe_batch_sizes.get(max_new_tokens)
        if safe_batch_size is not None and len(conversations) > safe_batch_size:
            outputs: list[str] = []
            for start in range(0, len(conversations), safe_batch_size):
                outputs.extend(
                    self.generate(
                        conversations[start : start + safe_batch_size],
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        num_beams=num_beams,
                    )
                )
            return outputs
        rng_state = capture_rng_state()
        try:
            return self._generate_batch(
                conversations,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
            )
        except torch.cuda.OutOfMemoryError:
            if len(conversations) == 1:
                raise
            restore_rng_state(rng_state)
            torch.cuda.empty_cache()
            middle = len(conversations) // 2
            previous_safe_size = self.safe_batch_sizes.get(max_new_tokens, len(conversations))
            self.safe_batch_sizes[max_new_tokens] = min(previous_safe_size, middle)
            return [
                *self.generate(
                    conversations[:middle],
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_beams=num_beams,
                ),
                *self.generate(
                    conversations[middle:],
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_beams=num_beams,
                ),
            ]

    def _generate_batch(
        self,
        conversations: Sequence[Sequence[Message]],
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        num_beams: int,
    ) -> list[str]:
        features = []
        for messages in conversations:
            input_ids = self._tokenize_chat(messages)
            features.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in batch.items()}
        generation_kwargs = {
            "do_sample": do_sample,
            "num_beams": num_beams,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": generation_eos_value(
                resolve_eos_token_ids(self.tokenizer, getattr(self.model, "generation_config", None))
            ),
        }
        if do_sample:
            generation_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
        generated = self.model.generate(**batch, **generation_kwargs)
        prompt_width = batch["input_ids"].shape[1]
        return self.tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
