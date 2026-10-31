"""Numerical equivalence, padding/EOS and real checkpoint regrouping regressions."""
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from opd.alignment import response_alignment_loss, response_view, boundary_state
from opd.batching import rollout_inputs, trim_generated, supervised_batch_loss
from opd.b200 import speed_up_plan, main as pipeline_main
from opd.data import normalize, write_jsonl, save_json, read_jsonl
from opd.losses import build_direct_batch, direct_opd_loss
from opd.resume import rewind_log
from opd.train import check_resume_batch, main
from tests_opd.test_pipeline import tokenizer, tiny


def scalar_alignment(student_hs, teacher_hs, batch, st, tt, projection, scores, top_k):
    """Original full-sequence projection + scalar interpolation reference."""
    chosen = torch.argsort(scores, descending=True, stable=True)[:top_k]
    weights = (scores[chosen] / 2).softmax(0)
    total = student_hs[-1].sum() * 0
    for index, weight in zip(chosen.tolist(), weights):
        layer = index + 1
        sl = math.ceil(layer * (len(student_hs) - 1) / (len(teacher_hs) - 1))
        losses = []
        for row in range(batch["op_input_batch"]["input_ids"].shape[0]):
            _, so, sp, _ = response_view(batch["op_input_batch"], batch["op_label_batch"], st, row)
            text, to, tp, base = response_view(batch["op_teacher_input_batch"], batch["op_teacher_label_batch"], tt, row)
            if not sp or not tp or base is None:
                continue
            th = teacher_hs[layer][row].float() @ projection.float()
            ts = torch.cat([th[base:base + 1], th[tp]], 0)
            for (a, b), pos in zip(so, sp):
                if b > a and b <= len(text):
                    dt = boundary_state(b, to, ts) - boundary_state(a, to, ts)
                    ds = student_hs[sl][row, pos] - student_hs[sl][row, pos - 1]
                    losses.append(1 - torch.nn.functional.cosine_similarity(ds, dt, dim=0, eps=1e-6))
        if losses:
            total = total + weight * torch.stack(losses).mean()
    return total


class FastTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(1)

    def test_rollout_left_padding_and_eos_as_pad(self):
        tok = tokenizer()
        tok.pad_token = tok.eos_token
        rows = [{"id": "a", "prompt": "hello", "prompt_format": "raw"},
                {"id": "b", "prompt": "hello world a", "prompt_format": "raw"}]
        enc = rollout_inputs(tok, rows, "cpu", 2)
        self.assertEqual(enc["attention_mask"].tolist(), [[0, 1], [1, 1]])
        self.assertEqual(enc["input_ids"][0].tolist(), [tok.eos_token_id, tok.convert_tokens_to_ids("hello")])
        eos = tok.eos_token_id
        actual = trim_generated(torch.tensor([[3, eos, eos], [eos, eos, eos], [3, 4, 5]]), eos)
        self.assertEqual(actual, [[3, eos], [eos], [3, 4, 5]])

    def test_ce_batched_matches_per_sample_losses_and_gradients(self):
        tok = tokenizer()
        rows = [{"id": "a", "prompt": "hello", "response": "world", "prompt_format": "raw"},
                {"id": "b", "prompt": "hello world", "response": "a b c d", "prompt_format": "raw"}]
        student = tiny(len(tok), 16, 2).eval()
        batched = supervised_batch_loss(student, tok, rows, "cpu", 32)
        expected = torch.stack([supervised_batch_loss(student, tok, [row], "cpu", 32) for row in rows]).mean()
        torch.testing.assert_close(batched, expected, atol=1e-6, rtol=1e-5)
        actual_grad = torch.autograd.grad(batched, student.get_input_embeddings().weight, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, student.get_input_embeddings().weight)[0]
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-4)

    def test_llama_batched_greedy_rollout_matches_individual_prompts(self):
        tok = tokenizer()
        tok.pad_token = tok.eos_token
        model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            bos_token_id=None, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)).eval()
        rows = [{"id": str(i), "prompt": p, "prompt_format": "raw"}
                for i, p in enumerate(("hello", "hello world a b"))]

        def generate(selected):
            enc = rollout_inputs(tok, selected, "cpu")
            with torch.no_grad():
                ids = model.generate(**enc, max_new_tokens=4, do_sample=False, use_cache=True)
            return trim_generated(ids[:, enc["input_ids"].shape[1]:], tok.eos_token_id)

        self.assertEqual(generate(rows), [generate([row])[0] for row in rows])

    def test_kd_sample_weighting_and_padding_gradients(self):
        for objective in ("forward_kl", "reverse_kl", "js_divergence"):
            s = torch.randn(3, 6, 9, requires_grad=True)
            t = torch.randn_like(s, requires_grad=True)
            labels = torch.tensor([[-100, 1, -100, -100, -100, -100],
                                   [-100, -100, 1, 1, 1, 1], [1, 1, -100, -100, -100, -100]])
            actual = direct_opd_loss(s, t, labels, objective, 2., reduction="sample_mean")
            expected = torch.stack([direct_opd_loss(s[i:i+1], t[i:i+1], labels[i:i+1], objective, 2.) for i in range(3)]).mean()
            torch.testing.assert_close(actual, expected)
            ag = torch.autograd.grad(actual, s, retain_graph=True)[0]
            eg = torch.autograd.grad(expected, s)[0]
            torch.testing.assert_close(ag, eg)
            self.assertEqual(ag[labels == -100].abs().sum(), 0)
            self.assertIsNone(t.grad)

    def test_vectorized_alignment_matches_original_value_and_gradients(self):
        from tests_opd.legacy_helpers import build_batch
        for different_tokenizer in (False, True):
            st, tt = tokenizer(), tokenizer(different_tokenizer)
            responses = ["hello world a b", "world"]
            sampled = [st.encode(r) + [st.eos_token_id] for r in responses]
            batch = build_batch(st, tt, ["answer", "hello answer"], responses, sampled, "cpu", 64)
            bs, sl = batch["op_input_batch"]["input_ids"].shape
            tl = batch["op_teacher_input_batch"]["input_ids"].shape[1]
            sh = [torch.randn(bs, sl, 16, requires_grad=True) for _ in range(3)]
            th = [torch.randn(bs, tl, 24) for _ in range(4)]
            projection = torch.randn(24, 16, requires_grad=True)
            scores = torch.tensor([.2, .4, .1])
            actual = response_alignment_loss(sh, th, batch, st, tt, projection, scores, 2)
            expected = scalar_alignment(sh, th, batch, st, tt, projection, scores, 2)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            ag = torch.autograd.grad(actual, [projection, sh[-1]], retain_graph=True)
            eg = torch.autograd.grad(expected, [projection, sh[-1]])
            for a, e in zip(ag, eg):
                torch.testing.assert_close(a, e, atol=1e-6, rtol=1e-4)

    def test_impact_sample_mean_preserves_empty_sample_and_padding(self):
        tok = tokenizer()
        sampled = [tok.encode("hello world a") + [tok.eos_token_id],
                   [tok.eos_token_id], tok.encode("b") + [tok.eos_token_id]]
        batch = build_direct_batch(tok, ["answer", "hello answer", "world"], sampled, "cpu", 32)
        shape = batch["op_input_batch"]["input_ids"].shape
        sh = [torch.randn(*shape, 16, requires_grad=True) for _ in range(3)]
        th = [torch.randn(*shape, 24) for _ in range(4)]
        proj = torch.randn(24, 16, requires_grad=True)
        scores = torch.ones(3)
        actual = response_alignment_loss(sh, th, batch, tok, tok, proj, scores, 2,
                                         allow_empty=True, reduction="sample_mean")
        losses = []
        for i in range(3):
            one = {key: {k: v[i:i+1] if v.ndim else v for k, v in value.items()} for key, value in batch.items()}
            losses.append(response_alignment_loss([h[i:i+1] for h in sh], [h[i:i+1] for h in th],
                          one, tok, tok, proj, scores, 2, allow_empty=True))
        expected = torch.stack(losses).mean()
        torch.testing.assert_close(actual, expected)
        ag = torch.autograd.grad(actual, proj, retain_graph=True)[0]
        eg = torch.autograd.grad(expected, proj)[0]
        torch.testing.assert_close(ag, eg, atol=1e-6, rtol=1e-4)

    def test_fast_plan_keeps_completed_jobs_and_effective_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save_json(root / "completed.json", {"state": "completed"})
            jobs = [{"stage": "train", "name": name, "artifact": str(root / f"{name}.json"),
                     "command": ["python", "-m", "opd.train", "--method", name, "--grad-accum", "32"]}
                    for name in ("completed", "opd_impact")]
            original = {"jobs": jobs, "cases": [], "arguments": {"steps": 0}}
            result = speed_up_plan(original)
            self.assertEqual(result["jobs"][0], original["jobs"][0])
            command = result["jobs"][1]["command"]
            self.assertEqual(command[command.index("--grad-accum") + 1], "4")
            self.assertIn("--no-gradient-checkpointing", command)
            self.assertEqual(speed_up_plan(result), result)
            self.assertNotIn("throughput_history", original)
            with self.assertRaises(ValueError):
                speed_up_plan(original, 3)
        check_resume_batch({"grad_accum": 32}, {"grad_accum": 4, "micro_batch_size": 8})
        with self.assertRaisesRegex(ValueError, "effective batch"):
            check_resume_batch({"grad_accum": 32}, {"grad_accum": 32, "micro_batch_size": 8})

    def test_unsaved_log_tail_is_archived_on_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_jsonl(root / "metrics.jsonl", [{"step": i} for i in (1, 2, 3)])
            rewind_log(root / "metrics.jsonl", 2, root / "archive")
            self.assertEqual(read_jsonl(root / "metrics.jsonl"), [{"step": 1}, {"step": 2}])
            self.assertEqual(read_jsonl(root / "archive/metrics.jsonl"), [{"step": 3}])

    def test_fast_cli_migrates_saved_plan_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan_file = root / "preview/plan.json"
            argv = ["b200", "plan", "--project-root", str(root / "local-assets"),
                    "--run-name", "fixture", "--plan-out", str(plan_file)]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                pipeline_main()
            before = json.loads(plan_file.read_text())
            with patch("sys.argv", argv + ["--fast"]), contextlib.redirect_stdout(io.StringIO()):
                pipeline_main()
            after = json.loads(plan_file.read_text())
            self.assertEqual(before["cases"], after["cases"])
            self.assertEqual([j for j in before["jobs"] if j["stage"] != "train"],
                             [j for j in after["jobs"] if j["stage"] != "train"])
            self.assertEqual(len(list((plan_file.parent / "plan_history").glob("*.json"))), 1)
            for job in after["jobs"]:
                if job["stage"] == "train":
                    command = job["command"]
                    self.assertEqual(command[command.index("--micro-batch-size") + 1], "8")
            with patch("sys.argv", argv + ["--fast"]), contextlib.redirect_stdout(io.StringIO()):
                pipeline_main()
            self.assertEqual(json.loads(plan_file.read_text()), after)
            self.assertFalse((root / "local-assets").exists())

    def test_real_llama_impact_training_resume_regroups_old_checkpoint(self):
        self._real_resume_smoke()

    def test_real_llama_deadline_saves_incomplete_checkpoint_and_resumes(self):
        self._real_resume_smoke(stop_at_deadline=True)

    def _real_resume_smoke(self, stop_at_deadline=False):
        with tempfile.TemporaryDirectory() as temp:
            root, tok = Path(temp), tokenizer()
            for name, width, layers in (("student", 16, 2), ("teacher", 24, 3)):
                model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=width,
                    intermediate_size=width * 2, num_hidden_layers=layers, num_attention_heads=2,
                    num_key_value_heads=2, max_position_embeddings=64, bos_token_id=None,
                    eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id))
                model.save_pretrained(root / name)
                tok.save_pretrained(root / name)
            rows = [normalize({"prompt": p, "response": "hello world"}, "fixture", "test", i)
                    for i, p in enumerate(("hello", "hello world", "a", "a b"))]
            for row in rows:
                row["prompt_format"] = "raw"
            write_jsonl(root / "train.jsonl", rows)
            write_jsonl(root / "calib.jsonl", [normalize({"prompt": "answer c d"}, "fixture", "test", 5)])
            common = ["train", "--student", str(root / "student"), "--teacher", str(root / "teacher"),
                      "--method", "opd_impact", "--data", str(root / "train.jsonl"),
                      "--calibration", str(root / "calib.jsonl"), "--out", str(root / "run"),
                      "--device", "cpu", "--dtype", "float32", "--epochs", "1", "--steps", "2",
                      "--lora-rank", "2", "--max-length", "16", "--max-prompt-length", "8",
                      "--max-new-tokens", "4", "--bi-max-length", "8", "--top-k", "2", "--save-every", "1"]
            # Simulated wall clock advances past the deadline after one metric
            # is committed. This exercises actual checkpointing, not a mock save.
            def virtual_clock():
                metrics = root / "run/metrics.jsonl"
                return 5_000_000_000. if metrics.exists() and metrics.stat().st_size else 0.

            stop_args = (["--stop-at", "2099-11-01T11:50:00+07:00"] if stop_at_deadline else
                         ["--max-minutes", "0.0000001"])
            with patch("sys.argv", common + ["--grad-accum", "2", *stop_args]), contextlib.redirect_stdout(io.StringIO()):
                with patch("opd.train.time.time", side_effect=virtual_clock) if stop_at_deadline else contextlib.nullcontext():
                    main()
            status = json.loads((root / "run/status.json").read_text())
            self.assertEqual(status["state"], "deadline_reached" if stop_at_deadline else "time_budget_reached")
            self.assertEqual(status["step"], 1)
            # Simulate schema-3 config (batch=1, no checkpoint-local config).
            checkpoint = root / "run/checkpoint-1"
            (checkpoint / "run_config.json").unlink()
            old = json.loads((root / "run/config.json").read_text())
            old.pop("micro_batch_size")
            save_json(root / "run/config.json", old)
            with patch("sys.argv", common + ["--grad-accum", "1", "--micro-batch-size", "2",
                       "--no-gradient-checkpointing", "--attention-backend", "sdpa", "--resume", str(checkpoint)]), contextlib.redirect_stdout(io.StringIO()):
                main()
            status = json.loads((root / "run/status.json").read_text())
            self.assertEqual(status["state"], "completed")
            metrics = read_jsonl(root / "run/metrics.jsonl")
            self.assertEqual([r["step"] for r in metrics], [1, 2])
            self.assertTrue(math.isfinite(metrics[-1]["impact_loss"]))
            self.assertGreater(metrics[-1]["samples_per_second"], 0)
            rollouts = read_jsonl(root / "run/rollouts.jsonl")
            self.assertEqual(len(rollouts), 4)
            self.assertEqual({r["sample_id"] for r in rollouts}, {r["id"] for r in rows})
            state = torch.load(root / "run/checkpoint-2/training_state.pt", weights_only=False)
            self.assertEqual(state["pointer"], 4)
            self.assertEqual(state["optimizer"]["param_groups"][0]["lr"], 0.)


if __name__ == "__main__":
    unittest.main()
