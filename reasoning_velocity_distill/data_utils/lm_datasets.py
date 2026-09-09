"""Causal training and benchmark datasets for JSONL and indexed token files."""

import json
import os
import re
from bisect import bisect_left
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import print_rank

def _count_indexed_steps(response_ids, marker_ids):
    marker_count = 0
    cursor = 0
    last_marker_end = 0
    marker_length = len(marker_ids)
    while cursor <= len(response_ids) - marker_length:
        if response_ids[cursor:cursor + marker_length] == marker_ids:
            marker_count += 1
            cursor += marker_length
            last_marker_end = cursor
        else:
            cursor += 1
    if last_marker_end < len(response_ids):
        marker_count += 1
    return marker_count


def _add_step_spans(sample, tokenizer, separator, response=None):
    """Map text boundaries onto the response tokens actually visible to the model.

    Offsets handle BPE tokens that merge punctuation with a newline. Matching
    tokenizer.encode('\n') directly in the input would miss those boundaries.
    """
    prompt_length = len(sample["prompt_ids"])
    visible_ids = sample["input_ids"][prompt_length:]
    if visible_ids and visible_ids[-1] == tokenizer.eos_token_id:
        visible_ids = visible_ids[:-1]
    sample["step_spans"] = []
    sample["marker_count"] = 0
    if not visible_ids:
        return sample
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Trajectory step boundaries require a fast tokenizer with offset mappings")
    text = sample["response"] if response is None else response
    if not text:
        text = tokenizer.decode(visible_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"][:len(visible_ids)] != visible_ids:
        raise ValueError("Response text does not align with indexed tokens; preprocess with the training tokenizer")

    boundaries = [0] + [match.end() for match in re.finditer(re.escape(separator), text)]
    boundaries.append(len(text))
    offsets = encoded["offset_mapping"][:len(visible_ids)]
    starts = [start for start, _ in offsets]
    for start, end in zip(boundaries, boundaries[1:]):
        if not text[start:end].strip():
            continue
        # A token crossing a separator (e.g. '.\n\n') belongs to the
        # preceding step; do not discard it along with the empty next line.
        token_start = bisect_left(starts, start)
        token_end = bisect_left(starts, end)
        if token_end > token_start:
            sample["step_spans"].append((prompt_length + token_start, prompt_length + token_end))
    sample["marker_count"] = len(sample["step_spans"])
    return sample


def _references(record):
    response = record.get("output", record.get("response"))
    answers = response if isinstance(response, list) else [response]
    if not answers or any(not isinstance(answer, str) or not answer.strip() for answer in answers):
        raise ValueError("'output' or 'response' must contain nonempty text references")
    return [answer.replace("\r\n", "\n").replace("\r", "\n") for answer in answers]


def _pack_example(prompt_ids, response_ids, max_length, max_prompt_length, response, marker_count):
    if not 0 < max_prompt_length < max_length:
        raise ValueError("Require 0 < max_prompt_length < max_length")
    prompt_ids = list(prompt_ids)[-max_prompt_length:]
    if not prompt_ids:
        raise ValueError("Prompt must contain at least one token")
    tokens = (prompt_ids + list(response_ids))[:max_length]
    if len(tokens) < 2:
        raise ValueError("A causal example must contain at least two tokens")
    labels = tokens[1:]
    labels[:len(prompt_ids) - 1] = [-100] * (len(prompt_ids) - 1)
    return dict(input_ids=tokens[:-1], label=labels, prompt_ids=prompt_ids,
                response=response, marker_count=marker_count)


def encode_causal_example(record, tokenizer, max_length, max_prompt_length, separator="\n"):
    if not isinstance(record, dict):
        raise TypeError("Each JSONL record must be an object")
    response = record.get("output", record.get("response"))
    if not isinstance(response, str) or not response.strip():
        raise ValueError("'output' or 'response' must be a nonempty string")
    response = response.replace("\r\n", "\n").replace("\r", "\n")
    if not isinstance(separator, str) or not separator:
        raise ValueError("step separator must be a nonempty string")
    prompt = record.get("prompt")
    if prompt is None:
        user_prompt = record.get("user_prompt")
        if not isinstance(user_prompt, str):
            raise ValueError("Provide a rendered 'prompt' or a 'user_prompt' string")
        messages = []
        if record.get("system_prompt"):
            messages.append({"role": "system", "content": record["system_prompt"]})
        messages.append({"role": "user", "content": user_prompt})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("'prompt' must be a nonempty string")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    marker_ids = tokenizer.encode(separator, add_special_tokens=False)
    if not marker_ids:
        raise ValueError("step separator must contain at least one token")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    if not response_ids:
        raise ValueError("Response must contain at least one token")
    sample = _pack_example(prompt_ids, response_ids + [tokenizer.eos_token_id],
                           max_length, max_prompt_length, response, marker_count=0)
    # The fallback matches token IDs in the packed input, so reserve one slot
    # per visible marker-delimited span, including a nonempty unmarked tail.
    # Count after truncation and causal shifting; EOS is a label, not an input.
    visible_ids = sample["input_ids"][len(sample["prompt_ids"]):]
    sample["marker_count"] = _count_indexed_steps(visible_ids, marker_ids)
    return sample


def _indexed_example(data, tokenizer, max_length, max_prompt_length, marker_ids,
                     response="", marker_count=None, prompt_only=False):
    # The preprocessors encode -1 in the file's integer dtype. Never treat a
    # valid token 65535 in a uint32 vocabulary as a prompt separator.
    sentinel = np.iinfo(data.dtype).max if np.issubdtype(data.dtype, np.unsignedinteger) else -1
    positions = np.flatnonzero(data == sentinel)
    if len(positions) != 1:
        if prompt_only and len(positions) == 0:
            prompt = data.astype(np.int64).tolist()
            response_ids = [tokenizer.eos_token_id]
        else:
            raise ValueError("Indexed causal data requires one -1 prompt/response separator")
    else:
        boundary = int(positions[0])
        prompt = data[:boundary].astype(np.int64).tolist()
        response_ids = data[boundary + 1:].astype(np.int64).tolist()
        if not response_ids:
            raise ValueError("Indexed example contains no response tokens")
    if marker_count is None:
        visible_response_ids = response_ids
        if visible_response_ids and visible_response_ids[-1] == tokenizer.eos_token_id:
            visible_response_ids = visible_response_ids[:-1]
        marker_count = _count_indexed_steps(visible_response_ids, marker_ids)
    return _pack_example(prompt, response_ids, max_length, max_prompt_length,
                         response, marker_count)


class LMTrainDataset(Dataset):
    def __init__(self, args, tokenizer, path, split, num=-1, ratio=1.0,
                 rng_sample=None, teacher_tokenizer=None, with_teacher=True,
                 *, prefer_indexed=False):
        del rng_sample
        self.args = args
        self.tokenizer = tokenizer
        self.teacher_tokenizer = teacher_tokenizer or tokenizer
        self.with_teacher = with_teacher
        self.split = split
        self.separator = getattr(args, "step_separator", "\n\n")
        if not isinstance(self.separator, str) or not self.separator:
            raise ValueError("step separator must be a nonempty string")
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id")
        self.student_marker_ids = tokenizer.encode(self.separator, add_special_tokens=False)
        self.teacher_marker_ids = self.teacher_tokenizer.encode(self.separator, add_special_tokens=False) if with_teacher else self.student_marker_ids
        if not self.student_marker_ids or not self.teacher_marker_ids:
            raise ValueError("step separator must contain at least one token")
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if not 0 < self.max_prompt_length < self.max_length:
            raise ValueError("Require 0 < max_prompt_length < max_length")
        if not 0 < ratio <= 1 or (num != -1 and num <= 0):
            raise ValueError("Dataset ratio must be in (0, 1], num must be -1 or positive")

        path = Path(path)
        source = path if path.is_file() else path / f"{split}.jsonl"
        if split == "valid" and not source.exists() and (path / "dev.jsonl").exists():
            source = path / "dev.jsonl"
        directory = source.parent
        indexed_split = source.stem
        has_indexed = (directory / f"{indexed_split}_0.idx").exists() and (directory / f"{indexed_split}_0.bin").exists()
        self.lm_ctx = self.t_lm_ctx = None
        if has_indexed and (prefer_indexed or getattr(args, "bin_data", False) or not source.exists()):
            from .distributed_indexed import DistributedMMapIndexedDataset

            # DistributedSampler handles rank assignment; the reader exposes the
            # complete dataset. Its legacy path API requires a trailing slash.
            self.lm_ctx = DistributedMMapIndexedDataset(str(directory) + os.sep, indexed_split, 0, 1)
            if with_teacher and (directory / f"teacher_{indexed_split}_0.idx").exists():
                self.t_lm_ctx = DistributedMMapIndexedDataset(str(directory) + os.sep, f"teacher_{indexed_split}", 0, 1)
                if len(self.t_lm_ctx) != len(self.lm_ctx):
                    raise ValueError("Teacher/student indexed datasets must contain the same number of examples")
        if self.lm_ctx is None and not source.exists():
            raise FileNotFoundError(f"No JSONL or indexed dataset found for {source}")
        self.raw = []
        if source.exists():
            with source.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise TypeError(f"{source}:{line_number}: Each JSONL record must be an object")
                    self.raw.append(record)
        total = len(self.lm_ctx) if self.lm_ctx is not None else len(self.raw)
        if self.lm_ctx is not None and self.raw and len(self.raw) != total:
            raise ValueError("JSONL references and indexed data must contain the same number of examples")
        self.num = min(int(total * ratio), num if num != -1 else total)
        if self.num == 0:
            raise ValueError(f"No examples selected from {source}")
        self.raw = self.raw[:self.num]
        self.answers = [_references(record) for record in self.raw]
        self.samples = None
        if self.lm_ctx is None:
            self.samples = []
            for index, record in enumerate(self.raw):
                canonical = dict(record, output=self.answers[index][0])
                student = encode_causal_example(canonical, tokenizer, self.max_length, self.max_prompt_length, self.separator)
                teacher = None
                if with_teacher:
                    teacher = encode_causal_example(canonical, self.teacher_tokenizer, args.t_max_length, args.t_max_prompt_length, self.separator)
                    _add_step_spans(student, tokenizer, self.separator)
                    _add_step_spans(teacher, self.teacher_tokenizer, self.separator)
                self.samples.append((student, teacher))
        print_rank(f"Num LM instances: {self.num}")

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        if index < 0:
            index += self.num
        if not 0 <= index < self.num:
            raise IndexError(index)
        if self.samples is not None:
            return self.samples[index]
        response = self.answers[index][0] if self.answers else ""
        marker_count = None
        student = _indexed_example(
            self.lm_ctx[index], self.tokenizer, self.max_length, self.max_prompt_length,
            self.student_marker_ids, response, marker_count,
            prompt_only=getattr(self.args, "only_prompt", False),
        )
        teacher = None
        if self.with_teacher:
            if self.t_lm_ctx is not None:
                teacher = _indexed_example(self.t_lm_ctx[index], self.teacher_tokenizer,
                    self.args.t_max_length, self.args.t_max_prompt_length,
                    self.teacher_marker_ids, response, marker_count)
            elif self.teacher_tokenizer is self.tokenizer:
                teacher = _indexed_example(self.lm_ctx[index], self.teacher_tokenizer,
                    self.args.t_max_length, self.args.t_max_prompt_length,
                    self.teacher_marker_ids, response, marker_count)
            elif self.raw:
                canonical = dict(self.raw[index], output=response)
                teacher = encode_causal_example(canonical, self.teacher_tokenizer,
                    self.args.t_max_length, self.args.t_max_prompt_length, self.separator)
            else:
                raise ValueError("A separate teacher tokenizer requires teacher indexed data or JSONL text")
            # Indexed text is kept verbatim: normalizing CRLF here would change
            # offsets relative to the tokens already written by preprocessing.
            indexed_response = None
            if self.raw:
                indexed_response = self.raw[index].get("output", self.raw[index].get("response"))
                if isinstance(indexed_response, list):
                    indexed_response = indexed_response[0]
            _add_step_spans(student, self.tokenizer, self.separator, indexed_response)
            teacher_response = indexed_response if self.t_lm_ctx is not None or self.teacher_tokenizer is self.tokenizer else None
            _add_step_spans(teacher, self.teacher_tokenizer, self.separator, teacher_response)
        return student, teacher

    def _collate_model(self, samples, tokenizer, model_type, marker_ids, step_count):
        length = max(len(sample["input_ids"]) for sample in samples)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        model_data = {
            "input_ids": torch.full((len(samples), length), pad_id, dtype=torch.long),
            "attention_mask": torch.zeros(len(samples), length, dtype=torch.long),
        }
        no_model_data = {
            "label": torch.full((len(samples), length), -100, dtype=torch.long),
            "step_marker_ids": torch.tensor(marker_ids, dtype=torch.long),
            "max_step_markers": step_count,
        }
        if self.with_teacher:
            no_model_data["step_spans"] = torch.full((len(samples), step_count, 2), -1, dtype=torch.long)
        if model_type == "gpt2":
            model_data["position_ids"] = torch.zeros(len(samples), length, dtype=torch.long)
        for index, sample in enumerate(samples):
            size = len(sample["input_ids"])
            model_data["input_ids"][index, :size] = torch.tensor(sample["input_ids"])
            model_data["attention_mask"][index, :size] = 1
            no_model_data["label"][index, :size] = torch.tensor(sample["label"])
            if self.with_teacher and sample["step_spans"]:
                no_model_data["step_spans"][index, :len(sample["step_spans"])] = torch.tensor(sample["step_spans"])
            if "position_ids" in model_data:
                model_data["position_ids"][index, :size] = torch.arange(size)
        no_model_data["loss_mask"] = (no_model_data["label"] != -100).float()
        return model_data, no_model_data

    def collate(self, samples):
        students, teachers = zip(*samples)
        step_count = max(sample["marker_count"] for sample in students)
        if self.with_teacher:
            step_count = max(step_count, max(sample["marker_count"] for sample in teachers))
        model_data, no_model_data = self._collate_model(
            students, self.tokenizer, self.args.model_type, self.student_marker_ids, step_count)
        teacher_data = teacher_metadata = None
        if self.with_teacher:
            teacher_data, teacher_metadata = self._collate_model(
                teachers, self.teacher_tokenizer,
                getattr(self.args, "teacher_model_type", None) or self.args.model_type,
                self.teacher_marker_ids, step_count)
        gen_data = {
            "input_ids": torch.full((len(samples), self.max_prompt_length), self.pad_id, dtype=torch.long),
            "attention_mask": torch.zeros(len(samples), self.max_prompt_length, dtype=torch.long),
        }
        for index, sample in enumerate(students):
            prompt = sample["prompt_ids"]
            gen_data["input_ids"][index, -len(prompt):] = torch.tensor(prompt)
            gen_data["attention_mask"][index, -len(prompt):] = 1
        return model_data, no_model_data, gen_data, teacher_data, teacher_metadata

    @staticmethod
    def move_to_device(model_data, no_model_data, gen_data, device):
        for batch in (model_data, no_model_data, gen_data):
            if batch is not None:
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device)
        return model_data, no_model_data, gen_data


class LMEvalDataset(LMTrainDataset):
    def __init__(self, args, tokenizer, path, split="test", rng_sample=None):
        super().__init__(args, tokenizer, path, split, rng_sample=rng_sample,
                         with_teacher=False, prefer_indexed=True)
        if not self.answers:
            raise ValueError("Benchmark evaluation requires JSONL references alongside indexed data")

    def collate(self, samples):
        model_data, no_model_data, gen_data, _, _ = super().collate(samples)
        return model_data, no_model_data, gen_data
