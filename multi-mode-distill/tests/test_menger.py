import unittest

import torch

from src.menger import _step_token_ranges, menger_loss_per_response


class NonRoundTripTokenizer:
    is_fast = True
    eos_token_id = None

    def __init__(self):
        self.pieces = {
            1: "alpha\n", 2: "\nbeta", 3: "\n\n", 4: "gamma",
            5: "\n\n", 6: "delta",
        }

    def decode(self, ids, **_kwargs):
        return "".join(self.pieces[token_id] for token_id in ids)

    def __call__(self, text, **_kwargs):
        # Tokenizing the isolated response chooses different IDs.
        return {"input_ids": [999], "offset_mapping": [(0, len(text))]}


class MengerNonRoundTripTest(unittest.TestCase):
    def test_step_ranges_keep_original_token_indices(self):
        tokenizer = NonRoundTripTokenizer()
        ranges = _step_token_ranges(tokenizer, torch.tensor([1, 2, 3, 4, 5, 6]), "\n\n")
        self.assertEqual(ranges, [(0, 1), (1, 2), (3, 4), (5, 6)])

    def test_partial_unicode_token_is_included(self):
        class UnicodeTokenizer(NonRoundTripTokenizer):
            def decode(self, ids, **_kwargs):
                if not ids:
                    return ""
                if len(ids) == 1:
                    return "\ufffd"
                return "你" + "".join(self.pieces[token_id] for token_id in ids[2:])

        tokenizer = UnicodeTokenizer()
        ranges = _step_token_ranges(tokenizer, torch.tensor([1, 2, 3, 4, 5, 6]), "\n\n")
        self.assertEqual(ranges, [(0, 2), (3, 4), (5, 6)])

    def test_full_loss_backpropagates_when_response_does_not_round_trip(self):
        tokenizer = NonRoundTripTokenizer()
        token_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        batch = {"input_ids": token_ids, "attention_mask": torch.ones_like(token_ids)}
        student = torch.tensor([[
            [0., 0.], [1., 0.], [0., 0.], [1., 1.], [0., 0.], [2., 1.],
        ]], requires_grad=True)
        teacher = torch.tensor([[
            [0., 0.], [1., 0.], [0., 0.], [2., 1.], [0., 0.], [3., 1.],
        ]])
        loss = menger_loss_per_response(
            student, teacher, batch, token_ids, batch, token_ids,
            tokenizer, separator="\n\n",
        ).mean()
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all().item())


if __name__ == "__main__":
    unittest.main()
