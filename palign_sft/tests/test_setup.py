import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

import yaml

ROOT = Path(__file__).resolve().parents[1]


def write_parquet_snapshot(directory, rows):
    import pyarrow as arrow
    from pyarrow import parquet

    data = directory / "data"
    data.mkdir(parents=True)
    parquet.write_table(
        arrow.Table.from_pylist(rows), data / "train-00000-of-00001.parquet"
    )


class PrepareS1KTests(unittest.TestCase):
    @staticmethod
    def _run_prepare(source, output, expected):
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "src" / "prepare_s1k.py"),
                "--source",
                str(source),
                "--output-dir",
                str(output),
                "--expected-rows",
                str(expected),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_builds_both_targets_without_reordering(self):
        rows = [
            {
                "question": "  Question A?  ",
                "solution": "  Label A  ",
                "deepseek_thinking_trajectory": "  Long reasoning A  ",
            },
            {
                "question": "Question B?",
                "solution": "Label B",
                "deepseek_thinking_trajectory": "Long reasoning B",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "source.jsonl"
            output = directory / "output"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            result = self._run_prepare(source, output, 2)
            self.assertEqual(result.returncode, 0, result.stderr)
            label = [
                json.loads(line)
                for line in (output / "s1k_label.jsonl").read_text().splitlines()
            ]
            longcot = [
                json.loads(line)
                for line in (output / "s1k_longcot.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [row["input"] for row in label], ["Question A?", "Question B?"]
            )
            self.assertEqual([row["output"] for row in label], ["Label A", "Label B"])
            self.assertEqual(
                [row["output"] for row in longcot],
                ["Long reasoning A", "Long reasoning B"],
            )
            self.assertTrue(
                all(set(row) == {"instruction", "input", "output"} for row in label)
            )

    def test_loads_downloaded_hugging_face_parquet_layout(self):
        rows = [
            {
                "question": "Parquet question?",
                "solution": "Parquet label",
                "deepseek_thinking_trajectory": "Parquet long reasoning",
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "s1K-1.1"
            output = directory / "output"
            write_parquet_snapshot(source, rows)
            result = self._run_prepare(source, output, 1)
            self.assertEqual(result.returncode, 0, result.stderr)
            converted = json.loads((output / "s1k_longcot.jsonl").read_text())
            self.assertEqual(converted["output"], "Parquet long reasoning")

    def test_rejects_missing_target_instead_of_silently_dropping(self):
        row = {"question": "Q", "solution": "S", "deepseek_thinking_trajectory": ""}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "source.json"
            source.write_text(json.dumps([row]), encoding="utf-8")
            result = self._run_prepare(source, directory / "output", 1)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("deepseek_thinking_trajectory", result.stderr)


class ConfigurationMatrixTests(unittest.TestCase):
    RUNS: ClassVar = {
        "qwen25_7b_label": (
            "Qwen2.5-7B-Instruct",
            "qwen",
            "s1k_label",
            "output/qwen25-7b-label",
        ),
        "qwen25_7b_longcot": (
            "Qwen2.5-7B-Instruct",
            "qwen",
            "s1k_longcot",
            "output/qwen25-7b-longcot",
        ),
        "qwen3_8b_label": ("Qwen3-8B", "qwen3", "s1k_label", "output/qwen3-8b-label"),
        "qwen3_8b_longcot": (
            "Qwen3-8B",
            "qwen3",
            "s1k_longcot",
            "output/qwen3-8b-longcot",
        ),
    }

    COMMON: ClassVar = {
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "lora_rank": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "cutoff_len": 32768,
        "train_on_prompt": False,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 3.0,
        "learning_rate": 5.0e-5,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "optim": "adamw_torch",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1.0e-8,
        "weight_decay": 0.0,
        "bf16": True,
        "gradient_checkpointing": True,
    }

    def test_all_train_and_export_configs_are_consistent(self):
        for stem, (model, template, dataset, run_dir) in self.RUNS.items():
            with self.subTest(stem=stem):
                train = yaml.safe_load(
                    (ROOT / "configs" / f"{stem}_sft.yaml").read_text()
                )
                export = yaml.safe_load(
                    (ROOT / "configs" / f"{stem}_export.yaml").read_text()
                )
                for key, value in self.COMMON.items():
                    self.assertEqual(train[key], value, key)
                self.assertTrue(
                    train["model_name_or_path"].endswith("/models/" + model)
                )
                self.assertEqual(train["template"], template)
                self.assertEqual(train["dataset"], dataset)
                self.assertEqual(train["output_dir"], run_dir + "/lora")
                if model == "Qwen3-8B":
                    self.assertIs(train["enable_thinking"], False)
                else:
                    self.assertNotIn("enable_thinking", train)
                self.assertEqual(
                    export["model_name_or_path"], train["model_name_or_path"]
                )
                self.assertEqual(export["template"], template)
                self.assertEqual(export["adapter_name_or_path"], run_dir + "/lora")
                self.assertEqual(export["export_dir"], run_dir + "/merged")

    def test_dataset_registry_and_download_manifest_cover_inputs(self):
        registry = json.loads((ROOT / "data" / "dataset_info.json").read_text())
        self.assertEqual(set(registry), {"s1k_label", "s1k_longcot"})
        manifest = (ROOT / "download.txt").read_text()
        self.assertEqual(manifest.splitlines()[0], "#d")
        self.assertIn("simplescaling/s1K-1.1", manifest)
        self.assertIn("Qwen/Qwen2.5-7B-Instruct", manifest)
        self.assertIn("Qwen/Qwen3-8B", manifest)
        self.assertEqual(manifest.count("--hf-dataset "), 5)
        self.assertEqual(manifest.count("--hf "), 2)


class FetchEvalTests(unittest.TestCase):
    def test_materializes_and_revalidates_all_eval_sets(self):
        specs = {
            "AIME_2024": 30,
            "aime_2025": 30,
            "aimo-validation-amc": 83,
            "MATH-500": 500,
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            assets = directory / "assets"
            output_data = directory / "normalized"
            for dataset_name, count in specs.items():
                rows = [
                    {
                        "question": f"{dataset_name} question {index}",
                        "answer": str(index),
                    }
                    for index in range(count)
                ]
                write_parquet_snapshot(assets / "datasets" / dataset_name, rows)
            environment = os.environ.copy()
            environment.update(
                {
                    "PALIGN_SFT_ASSET_ROOT": str(assets),
                    "PALIGN_SFT_DATA_DIR": str(output_data),
                }
            )
            command = [sys.executable, str(ROOT / "src" / "fetch_eval.py")]
            first = subprocess.run(
                command, check=False, capture_output=True, text=True, env=environment
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            second = subprocess.run(
                command, check=False, capture_output=True, text=True, env=environment
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("validated 500 existing rows", second.stdout)
            self.assertEqual(
                len((output_data / "raw" / "math500.jsonl").read_text().splitlines()),
                500,
            )


class ReportTests(unittest.TestCase):
    def test_report_reads_a_namespaced_result_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result_dir = directory / "result"
            result_dir.mkdir()
            benchmarks = {"aime25": 30, "aime24": 30, "amc12": 83, "math500": 500}
            for benchmark, count in benchmarks.items():
                rows = [{"label": [1, 0, 0], "passn": 1}] * count
                (result_dir / f"{benchmark}_scored.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
            report = directory / "eval_results.txt"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "src" / "report.py"),
                    "--result-dir",
                    str(result_dir),
                    "--out",
                    str(report),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            text = report.read_text(encoding="utf-8")
            self.assertIn("pass@1=33.33", text)
            self.assertIn("pass@3=100.00", text)
            self.assertIn("Avg", text)


class EvaluationTests(unittest.TestCase):
    def test_empty_candidates_are_scored_as_failures_without_dropping_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fake_modules = directory / "fake_modules"
            fake_modules.mkdir()
            (fake_modules / "math_verify.py").write_text(
                "def parse(value):\n    return value\n\n"
                "def verify(golden, prediction):\n    return bool(prediction)\n",
                encoding="utf-8",
            )
            source = directory / "generated.jsonl"
            destination = directory / "scored.jsonl"
            rows = [
                {"answer": "1", "output": ["correct", "", "also correct"]},
                {"answer": "2", "output": ["", "", ""]},
            ]
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            environment = os.environ.copy()
            existing_pythonpath = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = str(fake_modules) + (
                os.pathsep + existing_pythonpath if existing_pythonpath else ""
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "src" / "evaluation.py"),
                    "--input_path",
                    str(source),
                    "--output_path",
                    str(destination),
                    "--expected_n",
                    "3",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            scored = [json.loads(line) for line in destination.read_text().splitlines()]
            self.assertEqual(len(scored), 2)
            self.assertEqual(scored[0]["label"], [1, 0, 1])
            self.assertEqual(scored[1]["label"], [0, 0, 0])

    def test_candidate_count_mismatch_fails_loudly(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fake_modules = directory / "fake_modules"
            fake_modules.mkdir()
            (fake_modules / "math_verify.py").write_text(
                "def parse(value):\n    return value\n\n"
                "def verify(golden, prediction):\n    return False\n",
                encoding="utf-8",
            )
            source = directory / "generated.jsonl"
            source.write_text(
                json.dumps({"answer": "1", "output": ["only one"]}) + "\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(fake_modules)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "src" / "evaluation.py"),
                    "--input_path",
                    str(source),
                    "--output_path",
                    str(directory / "scored.jsonl"),
                    "--expected_n",
                    "3",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected 3 outputs", result.stderr)


class ProjectCommandsTests(unittest.TestCase):
    def test_train_stage_routes_all_four_runs_with_fake_launchers(self):
        eval_specs = {
            "AIME_2024": 30,
            "aime_2025": 30,
            "aimo-validation-amc": 83,
            "MATH-500": 500,
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            assets = directory / "assets"
            prepared = directory / "prepared"
            train_rows = [
                {
                    "question": f"Train question {index}",
                    "solution": f"Label {index}",
                    "deepseek_thinking_trajectory": f"Long reasoning {index}",
                }
                for index in range(2)
            ]
            write_parquet_snapshot(assets / "datasets" / "s1K-1.1", train_rows)
            for dataset_name, count in eval_specs.items():
                rows = [
                    {"question": f"{dataset_name} {index}", "answer": str(index)}
                    for index in range(count)
                ]
                write_parquet_snapshot(assets / "datasets" / dataset_name, rows)
            for model in ("Qwen2.5-7B-Instruct", "Qwen3-8B"):
                model_dir = assets / "models" / model
                model_dir.mkdir(parents=True)
                (model_dir / "config.json").write_text("{}\n", encoding="utf-8")

            fake_venv = directory / "venv" / "bin"
            fake_venv.mkdir(parents=True)
            (fake_venv / "activate").write_text("", encoding="utf-8")
            fake_bin = directory / "bin"
            fake_bin.mkdir()
            log = directory / "commands.log"
            for executable in ("torchrun", "llamafactory-cli"):
                script = fake_bin / executable
                script.write_text(
                    "#!/bin/sh\n"
                    f'printf \'%s\\n\' "{executable} $*" >> "$COMMAND_LOG"\n',
                    encoding="utf-8",
                )
                script.chmod(0o755)

            environment = os.environ.copy()
            environment.update(
                {
                    "ASSET_ROOT": str(assets),
                    "COMMAND_LOG": str(log),
                    "DATA_DIR": str(prepared),
                    "EXPECTED_TRAIN_ROWS": "2",
                    "PALIGN_VENV": str(directory / "venv"),
                    "PATH": str(fake_bin) + os.pathsep + environment["PATH"],
                    "PYTHON": sys.executable,
                }
            )
            result = subprocess.run(
                ["bash", str(ROOT / "project_commands.sh"), "train", "all"],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            commands = log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(commands), 8)
            for config in (
                "qwen25_7b_label_sft.yaml",
                "qwen25_7b_longcot_sft.yaml",
                "qwen3_8b_label_sft.yaml",
                "qwen3_8b_longcot_sft.yaml",
            ):
                self.assertTrue(any(config in command for command in commands), config)
            self.assertTrue((prepared / "dataset_info.json").exists())
            self.assertEqual(
                len((prepared / "s1k_longcot.jsonl").read_text().splitlines()), 2
            )


if __name__ == "__main__":
    unittest.main()
