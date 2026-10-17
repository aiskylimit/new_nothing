from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import GenerationConfig

from ..generation import generation_eos_value, resolve_eos_token_ids

IGNORE_INDEX = -100


@dataclass(slots=True)
class StudentRolloutGenerator:
    tokenizer: Any
    cutoff_len: int
    rollout_context_length: int
    do_sample: bool = True
    top_p: float = 1.0
    top_k: int = 0
    temperature: float = 0.5
    repetition_penalty: float = 1.0
    eos_token_ids: list[int] | None = None
    generation_config: GenerationConfig = field(init=False, repr=False)
    _stop_token_ids: frozenset[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 < self.rollout_context_length < self.cutoff_len:
            raise ValueError("rollout_context_length must be positive and smaller than cutoff_len.")
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Student rollout generation requires tokenizer pad and EOS tokens.")
        eos_token_ids = self.eos_token_ids or resolve_eos_token_ids(self.tokenizer)
        self.eos_token_ids = list(dict.fromkeys(eos_token_ids))
        self._stop_token_ids = frozenset((*self.eos_token_ids, self.tokenizer.pad_token_id))
        self.generation_config = GenerationConfig(
            do_sample=self.do_sample,
            top_p=self.top_p,
            top_k=self.top_k,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            eos_token_id=generation_eos_value(self.eos_token_ids),
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )

    def _extract_prompts_with_rows(self, inputs: dict[str, torch.Tensor]) -> list[tuple[torch.Tensor, int]]:
        """Extract a rollout prompt before every contiguous assistant target.

        LlamaFactory labels all assistant and function-call turns while
        masking user/system/tool observations. A tool trajectory can therefore
        contain multiple target spans. Each one becomes an independent
        on-policy continuation: the first may be a function call and a later
        one may be the final natural-language answer after a tool result.
        """

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"].bool()
        labels = inputs["labels"]
        prompts: list[tuple[torch.Tensor, int]] = []

        for row_index, (row_ids, row_mask, row_labels) in enumerate(
            zip(input_ids, attention_mask, labels, strict=True)
        ):
            valid_ids = row_ids[row_mask]
            valid_labels = row_labels[row_mask]
            response_positions = torch.nonzero(valid_labels.ne(IGNORE_INDEX), as_tuple=False).flatten()
            if response_positions.numel() == 0:
                raise ValueError("Cannot create a student rollout from an example without response labels.")
            span_starts = response_positions[
                torch.cat(
                    (
                        torch.ones(1, dtype=torch.bool, device=response_positions.device),
                        response_positions[1:].sub(response_positions[:-1]).ne(1),
                    )
                )
            ]
            for span_start in span_starts.tolist():
                if span_start == 0:
                    raise ValueError("Cannot create a student rollout from an empty prompt.")
                context_start = max(0, span_start - self.rollout_context_length)
                prompts.append((valid_ids[context_start:span_start], row_index))
        return prompts

    def extract_prompts(self, inputs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        return [prompt for prompt, _ in self._extract_prompts_with_rows(inputs)]

    def _left_pad_prompts(self, prompts: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(prompts)
        # The template always builds a fixed-width generation batch.
        padded_length = self.rollout_context_length
        device = prompts[0].device
        input_ids = torch.full(
            (batch_size, padded_length), self.tokenizer.pad_token_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros((batch_size, padded_length), dtype=torch.long, device=device)
        for index, prompt in enumerate(prompts):
            input_ids[index, -prompt.numel() :] = prompt
            attention_mask[index, -prompt.numel() :] = 1
        return input_ids, attention_mask

    def _trim_response(self, response_ids: torch.Tensor) -> torch.Tensor:
        kept: list[torch.Tensor] = []
        for token in response_ids:
            token_id = int(token.item())
            if token_id in self._stop_token_ids:
                break
            kept.append(token)
        if not kept:
            return response_ids[:0]
        return torch.stack(kept)

    @torch.no_grad()
    def generate(self, model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        prompt_records = self._extract_prompts_with_rows(inputs)
        prompts = [prompt for prompt, _ in prompt_records]
        generation_input_ids, generation_attention_mask = self._left_pad_prompts(prompts)
        input_width = generation_input_ids.shape[1]
        max_new_tokens = self.cutoff_len - self.rollout_context_length

        was_training = model.training
        model.eval()
        try:
            generated = model.generate(
                input_ids=generation_input_ids,
                attention_mask=generation_attention_mask,
                generation_config=self.generation_config,
                max_new_tokens=max_new_tokens,
            )
        finally:
            if was_training:
                model.train()

        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        features: list[dict[str, torch.Tensor]] = []
        task_ids = inputs.get("task_ids")
        if task_ids is not None and (task_ids.ndim != 1 or task_ids.shape[0] != inputs["input_ids"].shape[0]):
            raise ValueError("task_ids must contain one value per rollout input row.")
        for (prompt, row_index), sequence in zip(prompt_records, sequences, strict=True):
            # `prompt` is unpadded. Keep all of its tokens: some chat
            # templates deliberately use the EOS token between messages, so
            # removing values equal to pad_token_id would corrupt context.
            stored_prompt = prompt
            response = self._trim_response(sequence[input_width:])
            # An immediate EOS produces no token-level learning signal. Do
            # not add an all-IGNORE replay item because model CE loss is
            # undefined for a batch with no supervised tokens.
            if response.numel() == 0:
                continue
            available_response_length = self.cutoff_len - stored_prompt.numel()
            response = response[:available_response_length]
            full_ids = torch.cat((stored_prompt, response), dim=0)
            labels = torch.cat((torch.full_like(stored_prompt, IGNORE_INDEX), response), dim=0)
            feature = {
                "input_ids": full_ids.detach().cpu(),
                "attention_mask": torch.ones_like(full_ids, device="cpu"),
                "labels": labels.detach().cpu(),
            }
            if task_ids is not None:
                feature["task_id"] = task_ids[row_index].detach().cpu()
            features.append(feature)
        return features
