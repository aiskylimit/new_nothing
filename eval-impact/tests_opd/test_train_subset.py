import argparse
import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from opd.b200 import make_plan, main as pipeline_main, freeze_inputs
from opd.data import save_json, write_jsonl
from opd.subset import select_train_subset


class TrainSubsetTests(unittest.TestCase):
    def test_fixed_8000_subset_is_shared_and_2500_updates(self):
        rows = [{"id": f"row-{i:05}"} for i in range(11421)]
        selected = select_train_subset(rows, 8000, 42)
        self.assertEqual(selected, select_train_subset(list(reversed(rows)), 8000, 42))
        self.assertEqual(len({r["id"] for r in selected}), 8000)
        self.assertEqual(len(selected) // (8 * 4) * 10, 2500)
        self.assertNotEqual(selected, select_train_subset(rows, 8000, 43))
        with self.assertRaises(ValueError):
            select_train_subset(rows, 12000)

    def test_plan_reuses_completed_teacher_and_matches_both_subsets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "outputs/full/teacher_sft"
            save_json(source / "latest.json", {"checkpoint": str(source / "checkpoint-3560"), "step": 3560})
            save_json(source / "status.json", {"state": "completed", "step": 3560})
            args = argparse.Namespace(project_root=temp, run_name="small", profile="llama2-tinyllama",
                domains=["dolly-dskd"], seeds=[10], bi_seeds=[42], python_bin="python",
                steps=0, grad_accum=32, max_minutes=0, eval_max_new_tokens=256,
                math_eval=False, teacher_from_run="full", train_samples=8000, train_subset_seed=42)
            plan = make_plan(args)
            self.assertFalse(any(j["stage"] == "teacher" for j in plan["jobs"]))
            for job in plan["jobs"]:
                if job["stage"] == "train":
                    command = job["command"]
                    self.assertEqual(command[command.index("--train-samples") + 1], "8000")
                    self.assertEqual(command[command.index("--train-subset-seed") + 1], "42")
                    self.assertEqual(command[command.index("--teacher-adapter") + 1], str(source / "checkpoint-3560"))
            save_json(source / "status.json", {"state": "failed", "step": 3560})
            with self.assertRaisesRegex(ValueError, "completed"):
                make_plan(args)

    def test_existing_full_or_4000_plan_cannot_change_to_8000_on_resume(self):
        for previous in ("0", "4000"):
            with self.subTest(previous_samples=previous), tempfile.TemporaryDirectory() as temp:
                argv = ["b200", "plan", "--project-root", temp, "--run-name", "full", "--train-samples", previous]
                with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                    pipeline_main()
                argv[1] = "train-eval"
                argv[-1] = "8000"
                with patch("sys.argv", argv + ["--fast"]), self.assertRaisesRegex(ValueError, "new RUN_NAME"):
                    pipeline_main()

    def test_new_benchmark_files_do_not_invalidate_frozen_training_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / "data/opd/dolly-dskd/train.jsonl"
            write_jsonl(train, [{"prompt": "hello"}])
            plan = {"project_root": temp, "run_root": str(root / "outputs/full"),
                    "teacher": str(root / "models/t"), "student": str(root / "models/s"), "arguments": {}, "jobs": []}
            freeze_inputs(plan)
            write_jsonl(root / "data/opd/benchmarks/self-inst/test.jsonl", [{"prompt": "test"}])
            freeze_inputs(plan)
            write_jsonl(train, [{"prompt": "changed training data"}])
            with self.assertRaisesRegex(ValueError, "assets changed"):
                freeze_inputs(plan)

    def test_shell_train_automatically_selects_train_eval_and_8000_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            (bin_dir / "activate").write_text(f'export PATH="{bin_dir}:$PATH"\n')
            python = bin_dir / "python"
            python.write_text('#!/usr/bin/env python3\nimport json, os, sys\nfrom pathlib import Path\nPath(os.environ["OPD_TEST_CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n')
            python.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if k not in ("RUN_NAME", "TRAIN_SAMPLES", "AUTO_EVAL", "TEACHER_ADAPTER")}
            env.update(OPD_VENV=temp, OPD_TEST_CAPTURE=str(root / "args.json"), OPD_PROFILE="llama2-tinyllama")
            script = Path(__file__).resolve().parents[1] / "project_commands.sh"
            subprocess.run(["bash", str(script), "train"], env=env, check=True, capture_output=True, text=True)
            command = json.loads((root / "args.json").read_text())
            self.assertEqual(command[2], "train-eval")
            self.assertEqual(command[command.index("--train-samples") + 1], "8000")
            self.assertEqual(command[command.index("--run-name") + 1], "llama2-tinyllama-opd-8k")
            self.assertEqual(command[command.index("--teacher-from-run") + 1], "llama2-tinyllama-opd")

    def test_8k_script_forces_only_impact_and_keeps_auto_eval_setup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = Path(__file__).resolve().parents[1] / "run_8k_b200.sh"
            (root / source.name).write_text(source.read_text())
            (root / "project_commands.sh").write_text('python3 "$OPD_TEST_RECORDER" "$1"\n')
            recorder = root / "record.py"
            recorder.write_text('import json, os, sys\nfrom pathlib import Path\n'
                'row = {"action": sys.argv[1], **{k: os.environ.get(k) for k in '
                '["METHODS", "ONLY_METHOD", "STOP_AT", "TRAIN_SAMPLES", "DOLLY_EVAL", "EVAL_BENCHMARKS"]}}\n'
                'with Path(os.environ["OPD_TEST_CAPTURE"]).open("a") as f: f.write(json.dumps(row) + "\\n")\n')
            env = {**os.environ, "OPD_TEST_RECORDER": str(recorder), "OPD_TEST_CAPTURE": str(root / "calls.jsonl"),
                   "ONLY_METHOD": "opd", "METHODS": "opd", "STOP_AT": "2020-01-01T00:00:00+07:00"}
            subprocess.run(["bash", str(root / source.name)], env=env, check=True, capture_output=True, text=True)
            calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
            self.assertEqual([r["action"] for r in calls], ["setup-eval", "download-eval", "fast"])
            for row in calls:
                self.assertEqual(row["METHODS"], "opd_impact")
                self.assertEqual(row["ONLY_METHOD"], "opd_impact")
                self.assertEqual(row["TRAIN_SAMPLES"], "8000")
                self.assertEqual(row["DOLLY_EVAL"], "1")
                self.assertEqual(row["EVAL_BENCHMARKS"], "dolly self-inst vicuna sinst")
                self.assertIsNone(row["STOP_AT"])


if __name__ == "__main__":
    unittest.main()
