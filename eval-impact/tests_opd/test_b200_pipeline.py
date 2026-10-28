import argparse
import json
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
from opd.b200 import make_plan, execute_job
from opd.local_assets import parse_manifest
from opd.final_results import aggregate
from opd.data import save_json


class B200PipelineTests(unittest.TestCase):
    def test_plan_has_local_models_pure_opd_and_native_local_eval(self):
        args = argparse.Namespace(project_root="/mnt/local/test/impact_opd", run_name="test",
            profile="math-small", domains=["gsm8k"], seeds=[42], bi_seeds=[42, 43],
            python_bin="/mnt/local/uvenvs/impact-opd-b200/bin/python", steps=4, grad_accum=2,
            eval_max_new_tokens=1024, max_minutes=0)
        plan = make_plan(args)
        train = [j for j in plan["jobs"] if j["stage"] == "train"]
        self.assertEqual(len(train), 2)
        self.assertEqual({j["command"][j["command"].index("--method") + 1] for j in train}, {"opd", "opd_impact"})
        for job in train:
            cmd = job["command"]
            self.assertEqual(cmd[cmd.index("--kd-rate") + 1], "1")
            self.assertTrue(cmd[cmd.index("--teacher") + 1].startswith("/mnt/local/"))
        for job in plan["jobs"]:
            if job["stage"] == "eval":
                self.assertIn("--local-data-root", job["command"])

    def test_download_manifest_remaps_to_root_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "download.txt"
            path.write_text("--hf Qwen/test /mnt/local/@PROJECT@/impact_opd/models/test\n")
            self.assertEqual(parse_manifest(path, temp)[0]["destination"], str((Path(temp) / "models/test").resolve()))
            path.write_text("--hf Qwen/test /mnt/local/@PROJECT@/impact_opd/../../escape\n")
            with self.assertRaises(ValueError):
                parse_manifest(path, temp)

    def test_aggregate_local_task_aliases_delta_and_missing_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cases = []
            for method, score in [("opd", .2), ("opd_impact", .35)]:
                folder = root / method / "eval"
                save_json(folder / "evaluation_manifest.json", {
                    "seed": 42, "scope": "debug_subset", "limit": 1,
                    "max_new_tokens": 8, "num_fewshot": 0, "lm_eval_version": "test", "dtype": "float32",
                    "task_map": {"minerva_math500": "opd_math500_local", "gsm8k_cot": "opd_gsm8k_local"}})
                save_json(folder / "model/results_fixture.json", {"results": {
                    "opd_math500_local": {"exact_match,none": score},
                    "opd_gsm8k_local": {"exact_match,flexible-extract": score}}})
                cases.append({"name": method, "domain": "gsm8k", "method": method,
                              "seed": 42, "eval_dir": str(folder), "train_dir": None})
            plan = {"comparison": ["OPD", "OPD + IMPACT"], "profile": "fixture-only",
                    "run_root": str(root), "cases": cases}
            path = root / "plan.json"
            save_json(path, plan)
            result = aggregate(path, root / "report")
            self.assertEqual(result["state"], "complete")
            self.assertEqual(len(result["paired_deltas"]), 2)
            self.assertAlmostEqual(result["paired_deltas"][0]["delta_percentage_points"], 15.)
            cases.append({"name": "missing", "domain": "mixed", "method": "opd", "seed": 42,
                          "eval_dir": str(root / "missing"), "train_dir": None})
            save_json(path, plan)
            result = aggregate(path, root / "report-incomplete")
            self.assertEqual(result["state"], "incomplete")
            self.assertEqual(result["missing_evaluations"], ["missing"])
            self.assertEqual(len(result["accuracy"]), 4)

    def test_eval_cache_tracks_actual_checkpoint_after_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder, latest = root / "train/eval", root / "train/latest.json"
            job = {"name": "eval", "stage": "eval", "eval_dir": str(folder),
                   "artifact": str(folder / "evaluation_manifest.json"), "checkpoint_from": str(latest),
                   "command": ["python", "-m", "opd.evaluate", "--adapter", "placeholder", "--out", str(folder)]}
            calls = []
            def evaluate(command, **kwargs):
                calls.append(command[command.index("--adapter") + 1])
                save_json(folder / "evaluation_manifest.json", {"adapter": calls[-1]})
                save_json(folder / "results_fixture.json", {})
                return SimpleNamespace(returncode=0)
            ledger = {}
            save_json(latest, {"checkpoint": str(root / "checkpoint-1")})
            with patch("opd.b200.subprocess.run", side_effect=evaluate):
                execute_job(job, ledger, root / "logs")
                execute_job(job, ledger, root / "logs")
                self.assertEqual(len(calls), 1)
                save_json(latest, {"checkpoint": str(root / "checkpoint-2")})
                execute_job(job, ledger, root / "logs")
            self.assertEqual(calls, [str(root / "checkpoint-1"), str(root / "checkpoint-2")])
            self.assertEqual(len(list((folder.parent / "eval_history").iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
