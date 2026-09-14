import random
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call

import torch

from data_utils.lm_datasets import encode_causal_example
from src.adaptive import AdaptiveScheduler
from src.modes import align_response_logits
from src.self_distill import (complete_visible_steps, make_reference_model,
                              prepare_self_distill_batches, refresh_reference_model,
                              render_context_prompt, sample_context)


class CharacterTokenizer:
    eos_token_id = 999
    pad_token_id = 0
    is_fast = True
    chat_template = None
    all_special_tokens = []

    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 1 for char in text]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"input_ids": self.encode(text),
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class RenderedTokenizer(CharacterTokenizer):
    chat_template = "rendered"
    all_special_tokens = ["<u>", "<a>"]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False):
        return "<u>" + messages[-1]["content"] + "</u><a>"


class SelfDistillTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()
        self.response = "A\n\nB\n\nC\n\nD"

    def test_context_uses_complete_visible_steps_only(self):
        sample = encode_causal_example(
            {"prompt": "Q", "output": self.response}, self.tokenizer,
            max_length=8, max_prompt_length=4, separator="\n\n")
        self.assertEqual(complete_visible_steps(sample, self.tokenizer, "\n\n"),
                         ["A", "B", "C"])

    def test_random_context_is_ordered_proper_subset(self):
        steps = ["A", "B", "C", "D"]
        for ratio in (0.0, 0.5, 1.0):
            rng = random.Random(13)
            for _ in range(20):
                context = sample_context(steps, "\n\n", ratio, rng).split("\n\n")
                self.assertGreaterEqual(len(context), 1)
                self.assertLess(len(context), len(steps))
                self.assertEqual(context, [step for step in steps if step in context])
        self.assertEqual(sample_context(["A"], "\n\n", 0.5, random.Random(1)), "")

    def test_drop_probability_is_resampled_for_each_context(self):
        rng = Mock()
        rng.uniform.side_effect = [0.1, 0.4]
        rng.random.side_effect = [0.05, 0.2, 0.6, 0.8] * 2
        steps = ["A", "B", "C", "D"]
        self.assertEqual(sample_context(steps, "\n\n", 0.5, rng), "B\n\nC\n\nD")
        self.assertEqual(sample_context(steps, "\n\n", 0.5, rng), "C\n\nD")
        self.assertEqual(rng.uniform.call_args_list, [call(0.0, 0.5)] * 2)

    def test_adaptive_exposes_renamed_mode_without_changing_probability_rule(self):
        scheduler = AdaptiveScheduler(seed=1)
        self.assertEqual(scheduler.MODES, ("off_policy", "self_distill", "on_policy"))
        scheduler.on_evaluation(1.0, 1.0)
        logged = scheduler.on_evaluation(1.1, 1.0)
        self.assertAlmostEqual(logged["scheduler/rho_self"],
                               scheduler.config.rho_self_init + scheduler.config.rho_self_increment)
        self.assertAlmostEqual(logged["scheduler/rho_on"], 0.05)
        self.assertAlmostEqual(logged["scheduler/rho_off"],
                               1 - logged["scheduler/rho_self"] - logged["scheduler/rho_on"])

    def test_context_stays_before_rendered_assistant_header(self):
        tokenizer = RenderedTokenizer()
        rendered = render_context_prompt(
            {"prompt": "<u>Q</u><a>"}, tokenizer, "A",
            "\nContext: {context}")
        self.assertEqual(rendered, "<u>Q\nContext: A</u><a>")

    def test_empty_context_reuses_the_canonical_prompt(self):
        record = {"instruction": "raw question", "prompt": "custom rendered prompt"}
        self.assertEqual(render_context_prompt(
            record, RenderedTokenizer(), "", "\nContext: {context}"),
            record["prompt"])

    def test_context_uses_the_canonical_prompt_when_raw_fields_also_exist(self):
        record = {"instruction": "different question", "prompt": "<u>Canonical problem</u><a>"}
        self.assertEqual(render_context_prompt(
            record, RenderedTokenizer(), "A", "\nContext: {context}"),
            "<u>Canonical problem\nContext: A</u><a>")

    def test_reference_stays_frozen_until_explicit_refresh(self):
        student = torch.nn.Linear(2, 2)
        reference = make_reference_model(student)
        initial = reference.weight.detach().clone()
        self.assertFalse(reference.training)
        self.assertFalse(any(parameter.requires_grad for parameter in reference.parameters()))
        with torch.no_grad():
            student.weight.add_(1)
        self.assertTrue(torch.equal(reference.weight, initial))
        refresh_reference_model(reference, student)
        self.assertTrue(torch.equal(reference.weight, student.weight))
        self.assertFalse(reference.training)
        self.assertFalse(any(parameter.requires_grad for parameter in reference.parameters()))

    def test_full_response_labels_align_with_context_reference(self):
        sample = encode_causal_example(
            {"prompt": "Q", "output": self.response}, self.tokenizer,
            max_length=20, max_prompt_length=4, separator="\n\n")
        args = SimpleNamespace(max_length=20, max_prompt_length=4,
                               t_max_length=50, t_max_prompt_length=35,
                               model_type="qwen", step_separator="\n\n",
                               self_distill_context_drop_ratio=0.5,
                               self_distill_context_max_tokens=31,
                               self_distill_context_template="\nContext: {context}\n")
        batch = {"input_ids": torch.tensor([sample["input_ids"]])}
        metadata = {"label": torch.tensor([sample["label"]]),
                    "self_distill_steps": [["A", "B", "C", "D"]],
                    "self_distill_records": [{"prompt": "Q"}]}
        student, smeta, reference, rmeta = prepare_self_distill_batches(
            args, self.tokenizer, batch, metadata, random.Random(2))
        student_labels = smeta["label"]
        reference_labels = rmeta["label"]
        self.assertEqual(student_labels[student_labels != -100].tolist(),
                         reference_labels[reference_labels != -100].tolist())
        self.assertEqual(reference_labels[reference_labels != -100].tolist(),
                         sample["label"][-len(self.tokenizer.encode(self.response))-1:])
        aligned = align_response_logits(
            torch.zeros((*student_labels.shape, 2)), student_labels,
            torch.zeros((*reference_labels.shape, 2)), reference_labels)
        self.assertEqual(aligned[2].tolist(), reference_labels[reference_labels != -100].tolist())

    def test_generated_text_supplies_context_and_the_shared_completion(self):
        record = {"prompt": "Question?", "generated_text": "Step A\n\nStep B\n\nStep C"}
        sample = encode_causal_example(record, self.tokenizer, 100, 30, "\n\n")
        steps = complete_visible_steps(sample, self.tokenizer, "\n\n")
        self.assertEqual(steps, ["Step A", "Step B", "Step C"])
        args = SimpleNamespace(max_length=100, max_prompt_length=30,
                               t_max_length=150, t_max_prompt_length=80,
                               model_type="qwen", step_separator="\n\n",
                               self_distill_context_drop_ratio=0.5,
                               self_distill_context_max_tokens=50,
                               self_distill_context_template="\nContext:\n{context}\n")
        batch = {"input_ids": torch.tensor([sample["input_ids"]])}
        metadata = {"label": torch.tensor([sample["label"]]),
                    "self_distill_steps": [steps],
                    "self_distill_records": [record]}
        rng = Mock()
        rng.uniform.return_value = 0.5
        rng.random.side_effect = [0.9, 0.1, 0.9]
        student, student_meta, reference, reference_meta = prepare_self_distill_batches(
            args, self.tokenizer, batch, metadata, rng)

        def prompt_text(batch_data, batch_meta):
            start = (batch_meta["label"][0] != -100).nonzero()[0].item()
            ids = batch_data["input_ids"][0, :start + 1].tolist()
            return "".join(chr(token - 1) for token in ids)

        self.assertEqual(prompt_text(student, student_meta), "Question?")
        self.assertEqual(prompt_text(reference, reference_meta),
                         "Question?\nContext:\nStep A\n\nStep C\n")
        expected = self.tokenizer.encode(record["generated_text"]) + [self.tokenizer.eos_token_id]
        self.assertEqual(student_meta["label"][student_meta["label"] != -100].tolist(), expected)
        self.assertEqual(reference_meta["label"][reference_meta["label"] != -100].tolist(), expected)

    def test_short_prompt_cannot_use_more_than_its_context_token_cap(self):
        record = {"prompt": "Q", "generated_text": "AA\n\nBB\n\nCC\n\nDD"}
        sample = encode_causal_example(record, self.tokenizer, 32, 16, "\n\n")
        args = SimpleNamespace(max_length=32, max_prompt_length=16,
                               t_max_length=36, t_max_prompt_length=20,
                               model_type="qwen", step_separator="\n\n",
                               self_distill_context_drop_ratio=0.0,
                               self_distill_context_max_tokens=4,
                               self_distill_context_template="\n{context}")
        batch = {"input_ids": torch.tensor([sample["input_ids"]])}
        metadata = {"label": torch.tensor([sample["label"]]),
                    "self_distill_steps": [complete_visible_steps(sample, self.tokenizer, "\n\n")],
                    "self_distill_records": [record]}
        rng = Mock()
        rng.uniform.return_value = 0.0
        rng.random.side_effect = [0.9] * 4
        rng.randrange.return_value = 3
        _, student_meta, reference, reference_meta = prepare_self_distill_batches(
            args, self.tokenizer, batch, metadata, rng)
        start = (reference_meta["label"][0] != -100).nonzero()[0].item()
        prompt_ids = reference["input_ids"][0, :start + 1].tolist()
        self.assertEqual("".join(chr(token - 1) for token in prompt_ids), "Q\nAA")
        self.assertEqual(student_meta["self_distill_context_tokens"].item(), 2)

    def test_mixed_length_batch_keeps_each_response_aligned(self):
        records = [{"prompt": "Q", "output": "A\n\nB"},
                   {"prompt": "Long question", "output": "C\n\nD\n\nE"}]
        samples = [encode_causal_example(record, self.tokenizer, 32, 16, "\n\n")
                   for record in records]
        width = max(len(sample["label"]) for sample in samples)
        inputs = torch.full((2, width), self.tokenizer.pad_token_id)
        labels = torch.full((2, width), -100)
        for index, sample in enumerate(samples):
            size = len(sample["label"])
            inputs[index, :size] = torch.tensor(sample["input_ids"])
            labels[index, :size] = torch.tensor(sample["label"])
        args = SimpleNamespace(max_length=32, max_prompt_length=16,
                               t_max_length=64, t_max_prompt_length=40,
                               model_type="qwen", step_separator="\n\n",
                               self_distill_context_drop_ratio=0.5,
                               self_distill_context_max_tokens=24,
                               self_distill_context_template="\nContext: {context}\n")
        metadata = {"label": labels,
                    "self_distill_steps": [["A", "B"], ["C", "D", "E"]],
                    "self_distill_records": records}
        _, student_meta, _, reference_meta = prepare_self_distill_batches(
            args, self.tokenizer, {"input_ids": inputs}, metadata,
            [random.Random(1), random.Random(2)])
        for index, sample in enumerate(samples):
            expected = [token for token in sample["label"] if token != -100]
            self.assertEqual(student_meta["label"][index][
                student_meta["label"][index] != -100].tolist(), expected)
            self.assertEqual(reference_meta["label"][index][
                reference_meta["label"][index] != -100].tolist(), expected)

    def test_reference_context_cannot_displace_the_problem_statement(self):
        tokenizer = RenderedTokenizer()
        record = {"prompt": "<u>Problem: 2+2?</u><a>", "output": "Z"}
        sample = encode_causal_example(record, tokenizer, 40, 32, "\n\n")
        batch = {"input_ids": torch.tensor([sample["input_ids"]])}
        metadata = {"label": torch.tensor([sample["label"]]),
                    "self_distill_steps": [["A", "B", "C"]],
                    "self_distill_records": [record]}
        args = SimpleNamespace(max_length=40, max_prompt_length=32,
                               t_max_length=50, t_max_prompt_length=35,
                               model_type="qwen", step_separator="\n\n",
                               self_distill_context_drop_ratio=0.0,
                               self_distill_context_max_tokens=3,
                               self_distill_context_template="\nContext: {context}")

        def reference_prompt():
            student, student_meta, reference, reference_meta = prepare_self_distill_batches(
                args, tokenizer, batch, metadata, random.Random(2))
            self.assertEqual(student_meta["label"][student_meta["label"] != -100].tolist(),
                             reference_meta["label"][reference_meta["label"] != -100].tolist())
            start = (reference_meta["label"][0] != -100).nonzero()[0].item()
            prompt_ids = reference["input_ids"][0, :start + 1].tolist()
            return "".join(chr(token - 1) for token in prompt_ids), student_meta

        prompt, student_meta = reference_prompt()
        self.assertTrue(prompt.startswith("<u>Problem: 2+2?\nContext: "))
        self.assertTrue(prompt.endswith("</u><a>"))
        self.assertEqual(student_meta["self_distill_context_tokens"].item(), 1)

        args.t_max_prompt_length = len(record["prompt"])
        prompt, student_meta = reference_prompt()
        self.assertEqual(prompt, record["prompt"])
        self.assertEqual(student_meta["self_distill_context_tokens"].item(), 0)


if __name__ == "__main__":
    unittest.main()
