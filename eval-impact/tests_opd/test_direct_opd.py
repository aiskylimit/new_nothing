import unittest
import torch
from opd.losses import direct_opd_loss, build_direct_batch, require_shared_tokenizer
from opd.alignment import response_alignment_loss
from tests_opd.test_pipeline import tokenizer, tiny


class DirectOPDTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_reverse_kl_matches_distribution_definition_and_masks_prompt(self):
        s = torch.tensor([[[8., -8.], [0.4, -0.2], [-0.7, 0.3]]], requires_grad=True)
        t = torch.tensor([[[-8., 8.], [-0.1, 0.5], [0.2, 0.8]]], requires_grad=True)
        labels = torch.tensor([[-100, 0, 1]])
        expected = torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=s[0, 1:]),
            torch.distributions.Categorical(logits=t[0, 1:])).mean()
        actual = direct_opd_loss(s, t, labels)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNone(t.grad)
        self.assertEqual(float(s.grad[0, 0].abs().sum()), 0.)
        self.assertGreater(float(s.grad[0, 1:].abs().sum()), 0.)

    def test_equal_distributions_and_shape_validation(self):
        logits = torch.randn(1, 3, 5)
        labels = torch.tensor([[-100, 2, 3]])
        for objective in ["forward_kl", "reverse_kl", "js_divergence"]:
            self.assertAlmostEqual(float(direct_opd_loss(logits, logits, labels, objective)), 0., places=6)
        with self.assertRaises(ValueError):
            direct_opd_loss(logits, logits[:, :, :4], labels)
        with self.assertRaises(ValueError):
            direct_opd_loss(logits, logits, torch.full_like(labels, -100))

    def test_mismatched_tokenizer_rejected(self):
        require_shared_tokenizer(tokenizer(), tokenizer())
        with self.assertRaisesRegex(ValueError, "same token-to-ID"):
            require_shared_tokenizer(tokenizer(), tokenizer(True))

    def test_unassigned_head_padding_is_excluded_with_shared_token_ids(self):
        s = torch.tensor([[[0.4, -.2, 99.]]])
        t = torch.tensor([[[-.1, .5, -99., 99.]]])
        labels = torch.tensor([[1]])
        ids = torch.tensor([0, 1])
        expected = direct_opd_loss(s[:, :, :2], t[:, :, :2], labels)
        actual = direct_opd_loss(s, t, labels, token_ids=ids)
        torch.testing.assert_close(actual, expected)

    def test_exact_sampled_ids_one_shift_and_eos_as_pad(self):
        tok = tokenizer()
        tok.pad_token = tok.eos_token
        ids = tok.encode("hello world") + [tok.eos_token_id]
        batch = build_direct_batch(tok, ["answer"], [ids], "cpu", 64)
        self.assertIs(batch["op_input_batch"], batch["op_teacher_input_batch"])
        labels = batch["op_label_batch"]["label"]
        self.assertEqual(labels[labels.ne(-100)].tolist(), ids)
        self.assertEqual(int(labels.eq(tok.eos_token_id).sum()), 1)

    def test_opd_plus_impact_gradients_without_dual_space_modules(self):
        tok = tokenizer()
        student, teacher = tiny(len(tok), 16, 2), tiny(len(tok), 24, 3).eval().requires_grad_(False)
        batch = build_direct_batch(tok, ["answer"], [tok.encode("hello world") + [tok.eos_token_id]], "cpu", 64)
        so = student(**batch["op_input_batch"], output_hidden_states=True)
        with torch.no_grad():
            to = teacher(**batch["op_teacher_input_batch"], output_hidden_states=True)
        projection = torch.nn.Parameter(torch.randn(24, 16) * .01)
        opd = direct_opd_loss(so.logits, to.logits, batch["op_label_batch"]["label"])
        impact = response_alignment_loss(so.hidden_states, to.hidden_states, batch, tok, tok,
                                        projection, torch.tensor([.2, .3, .1]), 2)
        (opd + .25 * impact).backward()
        self.assertGreater(float(projection.grad.abs().sum()), 0.)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters()))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_eos_only_rollout_keeps_opd_and_zero_auxiliary(self):
        tok = tokenizer()
        student, teacher = tiny(len(tok), 16, 2), tiny(len(tok), 24, 3).eval().requires_grad_(False)
        batch = build_direct_batch(tok, ["answer"], [[tok.eos_token_id]], "cpu", 64)
        so = student(**batch["op_input_batch"], output_hidden_states=True)
        with torch.no_grad():
            to = teacher(**batch["op_teacher_input_batch"], output_hidden_states=True)
        loss = direct_opd_loss(so.logits, to.logits, batch["op_label_batch"]["label"])
        aux = response_alignment_loss(so.hidden_states, to.hidden_states, batch, tok, tok,
               torch.randn(24, 16), torch.ones(3), 2, allow_empty=True)
        self.assertEqual(float(aux), 0.)
        self.assertTrue(torch.isfinite(loss))
        (loss + aux).backward()


if __name__ == "__main__":
    unittest.main()
