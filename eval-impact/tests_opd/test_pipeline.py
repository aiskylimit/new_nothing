"""Offline tests: actual differently segmented text, gradients, and saved BI."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from opd.alignment import decoded_offsets, response_alignment_loss
from opd.bi import masked_bi, summarize, compare, collect
from opd.data import normalize, split_samples
from tests_opd.legacy_helpers import build_batch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "DSKDv2" / "code"))


def LegacyDualSpaceFixture(student, teacher, st, tt, args):
    # Only the archived legacy criterion regression tests need this fixture.
    return SimpleNamespace(student_model=student, teacher_model=teacher,
        student_tokenizer=st, teacher_tokenizer=tt, args=args,
        t2s_projector=torch.nn.Linear(teacher.config.hidden_size, student.config.hidden_size, bias=False),
        s2t_projector=torch.nn.Linear(student.config.hidden_size, teacher.config.hidden_size, bias=False))


def tokenizer(split=False):
    words = ["<pad>", "<eos>", "<unk>", "hello", "world", "a", "b", "c", "d", "answer", "1", "2", "3", "."]
    if split:
        # WordPiece teacher splits 'world' into 'wor' + '##ld'.
        words.remove("world")
        words += ["wor", "##ld"]
        tok = Tokenizer(models.WordPiece({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
        tok.decoder = decoders.WordPiece(prefix="##", cleanup=False)
    else:
        tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    result = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    result.chat_template = "{% for m in messages %}{{m['content'] + '\\n'}}{% endfor %}"
    return result


def tiny(vocab, width=16, depth=2):
    return GPT2LMHeadModel(GPT2Config(vocab_size=vocab, n_embd=width, n_layer=depth,
                          n_head=2, n_positions=128, eos_token_id=1, pad_token_id=0,
                          resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_bi_mask_excludes_padding_and_is_batch_invariant(self):
        h0 = torch.tensor([[[1., 0.], [1., 0.], [1., 0.]], [[1., 0.], [1., 0.], [1., 0.]]])
        h1 = torch.tensor([[[1., 0.], [0., 1.], [-1., 0.]], [[0., 1.], [-1., 0.], [-1., 0.]]])
        scores = masked_bi([h0, h1], torch.tensor([[1, 1, 0], [1, 0, 0]]))
        torch.testing.assert_close(scores[:, 0], torch.tensor([0.5, 1.]))
        with self.assertRaises(ValueError):
            masked_bi([h0, h1], torch.zeros(2, 3))

    def test_deterministic_disjoint_sampling(self):
        rows = [normalize({"question": f"question {i}"}, "gsm", "local/train", i) for i in range(30)]
        train, calib = split_samples(rows + rows[:2], 20, 8, 42)
        self.assertEqual((train, calib), split_samples(list(reversed(rows)), 20, 8, 42))
        self.assertFalse(set(r["id"] for r in train) & set(r["id"] for r in calib))

    def test_scores_rank_and_change_report(self):
        samples = [{"scores": [0.2, 0.8, 0.4]}, {"scores": [0.3, 0.7, 0.5]}]
        metadata = {"model": "tiny", "sample_ids": ["a", "b"]}
        a = summarize(samples, metadata, top_k=2, bootstrap=20)
        b = summarize([{"scores": [0.9, 0.1, 0.4]}, {"scores": [0.8, 0.2, 0.5]}], metadata,
                      top_k=2, bootstrap=20)
        self.assertEqual(a["top_layers_ranked"], [2, 3])
        r = compare(a, b)
        self.assertAlmostEqual(r["topk_overlap_fraction"], 0.5)
        self.assertAlmostEqual(r["spearman"], -1.)
        self.assertAlmostEqual(sum(x["weight"] for x in a["layers"]), 1.)
        self.assertEqual(r["per_layer"][0]["rank_change"], -2)

    def test_same_text_different_tokenization_actual_gradients(self):
        st, tt = tokenizer(), tokenizer(True)
        student, teacher = tiny(len(st)), tiny(len(tt), 24, 3).eval().requires_grad_(False)
        args = SimpleNamespace(projector_vocab=20, init_s2t_projector=False, topk_vocab=-1,
                               only_stu_kd=False, only_tea_kd=False, t2s_agreement=1.,
                               kd_rate=1., kd_temperature=2., teacher_temperature=1.,
                               kd_objective="forward_kl", label_smoothing=0.)
        model = LegacyDualSpaceFixture(student, teacher, st, tt, args)
        ids = st.encode("hello world", add_special_tokens=False) + [st.eos_token_id]
        batch = build_batch(st, tt, ["answer"], ["hello world"], [ids], "cpu", 64)
        self.assertNotEqual(int(batch["op_input_batch"]["attention_mask"].sum()),
                            int(batch["op_teacher_input_batch"]["attention_mask"].sum()))
        so = student(**batch["op_input_batch"], output_hidden_states=True)
        with torch.no_grad():
            to = teacher(**batch["op_teacher_input_batch"], output_hidden_states=True)
        from criterions.dual_space_kd_v2_with_exact_token_alignment import DualSpaceKDV2WithETA
        criterion = DualSpaceKDV2WithETA(args)
        criterion.distiller = model
        kd, logs = criterion.compute_on_policy_dual_space_kd_loss_with_eta(so, to, batch, model, {})
        projection = torch.nn.Parameter(torch.randn(24, 16) * .01)
        impact = response_alignment_loss(so.hidden_states, to.hidden_states, batch, st, tt,
                                        projection, torch.tensor([.2, .3, .1]), 2)
        (kd + .25 * impact).backward()
        self.assertTrue(torch.isfinite(kd) and torch.isfinite(impact))
        self.assertGreater(float(projection.grad.abs().sum()), 0.)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters()))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertIn("align_ratio", logs)

    def test_actual_bi_collect_persists_and_restores_mode(self):
        tok = tokenizer()
        model = tiny(len(tok)).train()
        rows = [normalize({"prompt": text}, "tiny", "test", i) for i, text in enumerate(["hello", "hello world", "a b c"])]
        with tempfile.TemporaryDirectory() as temp:
            a = collect(model, tok, rows, Path(temp) / "a", "tiny", batch_size=1, top_k=1)
            b = collect(model, tok, rows, Path(temp) / "b", "tiny", batch_size=2, top_k=1)
            torch.testing.assert_close(torch.tensor([r["bi_mean"] for r in a["layers"]]),
                                       torch.tensor([r["bi_mean"] for r in b["layers"]]), atol=1e-6, rtol=1e-5)
            self.assertEqual(len((Path(temp) / "b" / "samples.jsonl").read_text().splitlines()), 3)
            self.assertTrue((Path(temp) / "b" / "layers.csv").is_file())
            self.assertTrue(model.training)

    def test_mask_preserves_eos_when_it_is_pad(self):
        st, tt = tokenizer(), tokenizer(True)
        st.pad_token, tt.pad_token = st.eos_token, tt.eos_token
        ids = st.encode("hello world") + [st.eos_token_id]
        batch = build_batch(st, tt, ["answer"], ["hello world"], [ids], "cpu", 64)
        self.assertEqual(int(batch["op_label_batch"]["label"].eq(st.eos_token_id).sum()), 1)

    def test_rejects_different_responses_instead_of_stale_offsets(self):
        st, tt = tokenizer(), tokenizer(True)
        batch = build_batch(st, tt, ["answer"], ["a b"], [st.encode("hello world") + [1]], "cpu", 64)
        so = tiny(len(st))(**batch["op_input_batch"], output_hidden_states=True)
        to = tiny(len(tt))(**batch["op_teacher_input_batch"], output_hidden_states=True)
        with self.assertRaisesRegex(ValueError, "different response"):
            response_alignment_loss(so.hidden_states, to.hidden_states, batch, st, tt,
                                    torch.eye(16), torch.ones(2), 1)

    def test_byte_fallback_prefix_offsets_preserve_sampled_ids(self):
        class Decoder:
            def decode(self, ids, **kwargs):
                return {(): "", (1,): "\ufffd", (1, 2): "é", (1, 2, 3): "é!"}[tuple(ids)]
        text, offsets = decoded_offsets(Decoder(), [1, 2, 3])
        self.assertEqual(text, "é!")
        self.assertEqual(offsets, [(0, 0), (0, 1), (1, 2)])

    def test_legacy_opd_criterion_ignores_stale_raw_text(self):
        from criterions.dskdv2_impact import DualSpaceKDV2WithIMPACT
        st, tt = tokenizer(), tokenizer(True)
        args = SimpleNamespace(projector_vocab=20, init_s2t_projector=False, topk_vocab=-1,
                               only_stu_kd=False, only_tea_kd=False, t2s_agreement=1.,
                               kd_rate=1., kd_temperature=2., teacher_temperature=1.,
                               kd_objective="forward_kl", label_smoothing=0.)
        student, teacher = tiny(len(st)), tiny(len(tt), 24, 3).eval().requires_grad_(False)
        model = LegacyDualSpaceFixture(student, teacher, st, tt, args)
        batch = build_batch(st, tt, ["answer"], ["hello world"], [st.encode("hello world") + [1]], "cpu", 64)
        batch.update(input_batch=batch["op_input_batch"], label_batch=batch["op_label_batch"],
                     raw_texts=["stale off-policy response must never be aligned"])
        class Prepared(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._W_proj = torch.randn(24, 16)
                self._frozen_layer_scores = torch.tensor([.1, .4, .2])
                self.choose_align, self.choose_layer = "BI", "BI"
                self.top_k, self.bi_tau, self.random_seed = 2, 2., 42
            def prepare(self, distiller):
                pass
            def _ensure_bi_calibrated(self, *args):
                pass
        criterion = DualSpaceKDV2WithIMPACT(args)
        criterion.impact = Prepared()
        criterion.record_logging_output = lambda logging, log: log
        loss, logs = criterion(model, batch, {})
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(logs["impact_loss"]), 0)


if __name__ == "__main__":
    unittest.main()
