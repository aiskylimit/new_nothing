import argparse
import json
import tempfile
import unittest
from pathlib import Path
import torch
from opd.b200 import make_plan, execute_job
from opd.bi import render_prompt
from opd.data import read_jsonl, save_json
from opd.local_assets import prepare_reference_dolly
from opd.losses import build_direct_batch, direct_opd_loss
from opd.recipes import apply_recipe
from opd.train import make_parser
from opd.final_results import aggregate
from opd.evaluate import command as eval_command
from test_pipeline import tokenizer


class LlamaRecipeTests(unittest.TestCase):
    def test_matched_plan_has_teacher_dependencies_and_no_math_jobs(self):
        args = argparse.Namespace(project_root="/mnt/local/test/eval-impact", run_name="llama",
            profile="llama2-tinyllama", domains=["dolly-dskd"], seeds=[10], bi_seeds=[42, 43, 44],
            python_bin="python", steps=0, grad_accum=0, eval_max_new_tokens=1024, max_minutes=0,
            methods=["opd", "opd_impact"], math_eval=False)
        plan = make_plan(args)
        self.assertIn("Llama-2-7b-hf", plan["teacher"])
        self.assertIn("intermediate-step-1431k-3T", plan["student"])
        self.assertEqual(plan["jobs"][0]["stage"], "teacher")
        self.assertFalse(any(j["stage"] == "eval" for j in plan["jobs"]))
        train = [j for j in plan["jobs"] if j["stage"] == "train"]
        self.assertEqual(len(train), 2)
        for job in train:
            self.assertEqual(job["command"][job["command"].index("--grad-accum") + 1], "32")
            self.assertIn("--teacher-adapter", job["adapter_dependencies"])
        parsed = apply_recipe(make_parser().parse_args(["--method", "opd", "--data", "d", "--calibration", "c", "--out", "o"]))
        self.assertEqual((parsed.lr, parsed.kd_rate, parsed.kd_temperature), (.001, .5, 2.))
        self.assertEqual((parsed.lora_rank, parsed.lora_alpha, parsed.lora_dropout), (256, 8, .1))
        self.assertEqual(parsed.scheduler, "cosine")

    def test_reference_split_keeps_prompts_and_excludes_train_test_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            prepare_reference_dolly(temp)
            root = Path(temp) / "data/opd/dolly-dskd"
            train, test = read_jsonl(root / "train.jsonl"), read_jsonl(root / "test.jsonl")
            train_hashes = {r["prompt_sha256"] for r in train}
            test_hashes = {r["prompt_sha256"] for r in test}
            self.assertEqual((len(train), len(test)), (11421, 500))
            self.assertFalse(train_hashes & test_hashes)
            source = Path(__file__).resolve().parents[1] / "data/dolly/train.jsonl"
            original = read_jsonl(source)
            for row in train:
                self.assertEqual(row["prompt"], original[row["source_index"]]["prompt"])
                self.assertEqual(row["response"], original[row["source_index"]]["output"])
            subsets = []
            for seed in (42, 43, 44):
                calibration = read_jsonl(root / f"calibration-seed{seed}.jsonl")
                self.assertEqual(len(calibration), 256)
                self.assertFalse({r["prompt_sha256"] for r in calibration} & (train_hashes | test_hashes))
                subsets.append([r["id"] for r in calibration])
            self.assertNotEqual(subsets[0], subsets[1])

    def test_raw_prompt_truncation_is_identical_for_rollout_scoring(self):
        tok = tokenizer()
        prompt = "hello world answer question"
        self.assertEqual(render_prompt(tok, prompt, "raw"), prompt)
        prefix = tok.encode(prompt, add_special_tokens=False)[:2]
        batch = build_direct_batch(tok, [prompt], [[3, 4, tok.eos_token_id]], "cpu", 8,
                                   prompt_format="raw", max_prompt_length=2)
        ids = batch["op_input_batch"]["input_ids"][0].tolist()
        self.assertEqual(ids, prefix + [3, 4])
        self.assertEqual(batch["op_label_batch"]["label"][0].tolist(), [-100, 3, 4, tok.eos_token_id])

    def test_reference_temperature_scaling_is_explicit(self):
        torch.manual_seed(1)
        s, t, labels = torch.randn(1, 2, 5), torch.randn(1, 2, 5), torch.ones(1, 2).long()
        scaled = direct_opd_loss(s, t, labels, "forward_kl", 2.)
        reference = direct_opd_loss(s, t, labels, "forward_kl", 2., temperature_squared=False)
        torch.testing.assert_close(scaled, reference * 4)

    def test_incomplete_teacher_cannot_feed_student(self):
        with tempfile.TemporaryDirectory() as temp:
            pointer = Path(temp) / "latest.json"
            save_json(pointer, {"checkpoint": "teacher-checkpoint"})
            save_json(Path(temp) / "status.json", {"state": "failed"})
            job = {"command": ["python", "--teacher-adapter", "pending"],
                   "adapter_dependencies": {"--teacher-adapter": str(pointer)}}
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                execute_job(job, {}, Path(temp) / "logs")

    def test_no_math_report_tracks_both_training_budgets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cases, jobs = [], []
            for method in ("opd", "opd_impact"):
                folder = root / method
                save_json(folder / "status.json", {"state": "completed"})
                (folder / "metrics.jsonl").write_text(json.dumps({"step": 2, "opd_loss": .1,
                    "cumulative_loop_seconds": 3., "cumulative_generated_tokens": 6}) + "\n")
                cases.append({"name": method, "domain": "dolly-dskd", "method": method, "seed": 10,
                              "train_dir": str(folder), "eval_dir": None})
                jobs.append({"name": method, "artifact": str(folder / "status.json")})
            plan = {"run_root": str(root), "profile": "llama2-tinyllama", "math_eval_enabled": False,
                    "comparison": ["OPD", "OPD + IMPACT"], "cases": cases, "jobs": jobs}
            save_json(root / "plan.json", plan)
            report = aggregate(root / "plan.json", root / "report")
            self.assertEqual(report["state"], "complete")
            self.assertEqual(report["missing_evaluations"], [])
            self.assertEqual(report["accuracy"], [])
            self.assertEqual(len(report["training_cost"]), 2)
            self.assertFalse((root / "report/accuracy_by_run.csv").exists())
            save_json(root / "opd/status.json", {"state": "time_budget_reached"})
            self.assertEqual(aggregate(root / "plan.json", root / "report")["state"], "incomplete")

    def test_base_model_eval_does_not_require_chat_template(self):
        args = argparse.Namespace(model="base-llama", dtype="bfloat16", adapter=None, revision=None,
            batch_size=1, device="cpu", out="eval", seed=10, max_new_tokens=4, num_fewshot=None,
            limit=None, use_chat_template=False)
        command = eval_command(args)
        self.assertNotIn("--apply_chat_template", command)
        self.assertNotIn("--fewshot_as_multiturn", command)


if __name__ == "__main__":
    unittest.main()
