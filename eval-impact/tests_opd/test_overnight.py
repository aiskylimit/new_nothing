import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from opd.b200 import main as pipeline_main, make_plan, speed_up_plan
from opd.data import save_json
from opd.timing import deadline_timestamp, estimate_timing


class OvernightTests(unittest.TestCase):
    def test_deadline_requires_timezone_and_eta_reports_budget(self):
        with self.assertRaisesRegex(ValueError, "timezone"):
            deadline_timestamp("2026-11-01T12:00:00")
        self.assertEqual(deadline_timestamp("2026-11-01T12:00:00+07:00"),
                         deadline_timestamp("2026-11-01T05:00:00Z"))
        timing = estimate_timing(100, 200, [10., 14.], 1000., 2300.)
        self.assertEqual(timing["rolling_step_seconds"], 12.)
        self.assertEqual(timing["required_seconds_per_step"], 13.)
        self.assertTrue(timing["estimated_within_deadline"])
        self.assertFalse(estimate_timing(100, 200, [14.], 1000., 2300.)["estimated_within_deadline"])

    def test_only_impact_does_not_rewrite_baseline_command(self):
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(project_root=temp, run_name="fixture", profile="llama2-tinyllama",
                domains=["dolly-dskd"], seeds=[10], bi_seeds=[42], python_bin="python",
                steps=0, grad_accum=0, max_minutes=0, eval_max_new_tokens=1024, math_eval=False)
            original = make_plan(args)
            updated = speed_up_plan(original, only_method="opd_impact")
            for old, new in zip(original["jobs"], updated["jobs"]):
                if old["stage"] == "train" and "opd_impact" in old["command"]:
                    self.assertIn("--micro-batch-size", new["command"])
                else:
                    self.assertEqual(old, new)
            self.assertEqual(original["cases"], updated["cases"])

    def test_saved_plan_selects_only_impact_and_forwards_runtime_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            argv = ["b200", "plan", "--project-root", str(root), "--run-name", "fixture"]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                pipeline_main()
            deadline = "2099-11-01T11:50:00+07:00"
            argv[1] = "train"
            with patch("sys.argv", argv + ["--fast", "--only-method", "opd_impact", "--stop-at", deadline]), \
                    patch("opd.b200.preflight"), patch("opd.b200.verify_model"), \
                    patch("opd.b200.freeze_inputs"), patch("transformers.AutoTokenizer.from_pretrained"), \
                    patch("opd.losses.require_shared_tokenizer"), patch("opd.b200.execute_job") as execute:
                pipeline_main()
            self.assertEqual(execute.call_count, 1)
            args = execute.call_args.args
            self.assertIn("opd_impact", args[0]["command"])
            self.assertEqual(args[3], deadline)
            saved = json.loads((root / "outputs/fixture/plan.json").read_text())
            self.assertEqual(len(saved["cases"]), 2)
            for job in saved["jobs"]:
                self.assertNotIn("--stop-at", job["command"])


if __name__ == "__main__":
    unittest.main()
