import argparse
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/eval"))
from eval_utils import (EvalError, classify_checkpoint, detect_architecture, output_dir,
                        git_commit, parse_overrides, resolve_base, select_checkpoint, suite_config)
from eval_utils import update_summary
from prepare_eval_assets import resolve_registry_classes
import prepare_eval_assets as prepare
from run_all_methods import discover_runs, is_completed_summary, is_mode_complete
from run_eval import effective_judge, mode_is_complete, write_config

CANONICAL = ["GQA_TestDev_Balanced", "MME", "RealWorldQA", "ScienceQA_TEST",
             "AI2D_TEST_NO_MASK", "MMMU_DEV_VAL", "MMStar", "ChartQA_TEST",
             "DocVQA_VAL", "TextVQA_VAL", "OCRBench"]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def full_checkpoint(path, model_type):
    path.mkdir(parents=True)
    dump(path / "config.json", {"model_type": model_type, "architectures": [model_type]})
    (path / "model.safetensors").write_bytes(b"weights")
    dump(path / "processor_config.json", {})
    dump(path / "tokenizer_config.json", {})
    return path


def adapter_checkpoint(path, base):
    path.mkdir(parents=True)
    dump(path / "adapter_config.json", {"base_model_name_or_path": str(base), "auto_mapping": {}})
    (path / "adapter_model.safetensors").write_bytes(b"adapter")
    dump(path / "processor_config.json", {})
    dump(path / "tokenizer_config.json", {})
    return path


class EvalPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_canonical_suite_and_no_paid_judge(self):
        cfg = suite_config(ROOT / "configs/eval/requested_benchmarks.json")
        self.assertEqual(cfg["datasets"], CANONICAL)
        self.assertEqual(cfg["dataset_class_overrides"], {"MME": "MMEPerceptionDataset"})
        self.assertIsNone(cfg["judge_model"])
        self.assertEqual(effective_judge(None), "exact_matching")
        self.assertEqual(effective_judge("opt-in-model"), "opt-in-model")

    def test_unsupported_names_consolidated(self):
        class Fake:
            @classmethod
            def supported_datasets(cls): return ["MME"]
        with self.assertRaises(EvalError) as caught:
            resolve_registry_classes([Fake], ["MME", "wrong-one", "wrong-two"])
        self.assertIn("wrong-one, wrong-two", str(caught.exception))

    def test_vendored_commit_metadata_replaces_nested_git(self):
        kit = self.root / "VLMEvalKit"
        kit.mkdir()
        (kit / ".vendored-commit").write_text("bdb6d429f3f6804eaa6cd899c341486c8a42aed8\n")
        self.assertEqual(git_commit(kit), "bdb6d429f3f6804eaa6cd899c341486c8a42aed8")

    def test_latest_checkpoint_numeric_and_complete(self):
        run = self.root / "run"
        full_checkpoint(run / "checkpoint-9", "qwen2_vl")
        full_checkpoint(run / "checkpoint-10", "qwen2_vl")
        (run / "checkpoint-100").mkdir()
        self.assertEqual(select_checkpoint(run).name, "checkpoint-10")

    def test_classification(self):
        full = full_checkpoint(self.root / "full", "qwen2_vl")
        base = full_checkpoint(self.root / "base", "qwen2_5_vl")
        adapter = adapter_checkpoint(self.root / "adapter", base)
        incomplete = self.root / "incomplete"; incomplete.mkdir()
        dump(incomplete / "config.json", {"model_type": "fast_vlm"})
        self.assertEqual(classify_checkpoint(full)[0], "merged/full")
        self.assertEqual(classify_checkpoint(adapter)[0], "adapter-only")
        self.assertEqual(classify_checkpoint(incomplete)[0], "incomplete")
        self.assertEqual(classify_checkpoint(self.root / "missing")[0], "unsupported")

    def test_architecture_uses_metadata_even_with_misleading_name(self):
        for model_type in ("qwen2_vl", "qwen2_5_vl", "fast_vlm"):
            with self.subTest(model_type=model_type):
                checkpoint = full_checkpoint(self.root / f"misleading-{model_type}", model_type)
                self.assertEqual(detect_architecture(checkpoint)[0], model_type)

    def test_architecture_can_be_read_from_hf_architectures(self):
        expected = {"Qwen2VLForConditionalGeneration": "qwen2_vl",
                    "Qwen2_5_VLForConditionalGeneration": "qwen2_5_vl",
                    "FastVlmForConditionalGeneration": "fast_vlm"}
        for architecture, model_type in expected.items():
            with self.subTest(architecture=architecture):
                checkpoint = self.root / model_type; checkpoint.mkdir()
                dump(checkpoint / "config.json", {"architectures": [architecture]})
                (checkpoint / "model.safetensors").write_bytes(b"weights")
                dump(checkpoint / "processor_config.json", {})
                dump(checkpoint / "tokenizer_config.json", {})
                self.assertEqual(detect_architecture(checkpoint)[0], model_type)

    def test_stale_base_override(self):
        base = full_checkpoint(self.root / "new-base", "qwen2_5_vl")
        adapter = adapter_checkpoint(self.root / "adapter", "/stale/absolute/base")
        overrides = parse_overrides([f"/stale/absolute/base={base}"])
        self.assertEqual(resolve_base(adapter, overrides)[0], str(base.resolve()))
        self.assertEqual(detect_architecture(adapter, overrides)[0], "qwen2_5_vl")

    def test_explicit_base_can_replace_any_embedded_adapter_path(self):
        base = full_checkpoint(self.root / "local-base", "qwen2_vl")
        adapter = adapter_checkpoint(self.root / "adapter", "/server-that-no-longer-exists/base")
        embedded = json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"]
        overrides = {embedded: str(base.resolve())}
        architecture, resolved, _ = detect_architecture(adapter, overrides)
        self.assertEqual(architecture, "qwen2_vl")
        self.assertEqual(resolved, str(base.resolve()))

    def test_adapter_auto_mapping_must_match_base_config(self):
        base = full_checkpoint(self.root / "base", "fast_vlm")
        adapter = adapter_checkpoint(self.root / "adapter", base)
        cfg = json.loads((adapter / "adapter_config.json").read_text())
        cfg["auto_mapping"] = {"base_model_class": "Qwen2VLForConditionalGeneration"}
        dump(adapter / "adapter_config.json", cfg)
        with self.assertRaises(EvalError):
            detect_architecture(adapter)

    def test_output_paths_do_not_collide(self):
        one = output_dir(self.root, "suite", "method-a", Path("checkpoint-10"))
        two = output_dir(self.root, "suite", "method-b", Path("checkpoint-10"))
        self.assertNotEqual(one, two)

    def test_recursive_artifact_discovery(self):
        run = self.root / "outputs/group/method"
        full_checkpoint(run / "checkpoint-2", "qwen2_vl")
        self.assertEqual(discover_runs(self.root / "outputs"), [run])

    def test_partial_result_not_done(self):
        run = self.root / "run"; run.mkdir()
        (run / "result.xlsx").write_bytes(b"partial")
        self.assertFalse(is_completed_summary(run / "summary.json"))
        dump(run / "summary.json", {"overall_status": "partial"})
        self.assertFalse(is_completed_summary(run / "summary.json"))
        dump(run / "summary.json", {"overall_status": "complete"})
        self.assertTrue(is_completed_summary(run / "summary.json"))

    def test_batch_completion_is_mode_specific(self):
        path = self.root / "summary.json"
        dump(path, {"overall_status": "partial", "benchmarks": [
            {"inference_status": "complete", "scoring_status": "skipped"}]})
        self.assertTrue(is_mode_complete(path, "infer"))
        self.assertFalse(is_mode_complete(path, "eval"))
        self.assertFalse(is_mode_complete(path, "all"))
        self.assertTrue(mode_is_complete(path, "infer"))
        self.assertFalse(mode_is_complete(path, "all"))

    def test_canonical_summary_contains_each_benchmark_and_metric(self):
        run = self.root / "run"; native = run / "native/model/eval-id"; native.mkdir(parents=True)
        prediction = native / "model_MME.xlsx"; prediction.write_bytes(b"result")
        dump(native / "status.json", {"datasets": {"MME": {"status": "done",
             "prediction_file": prediction.name, "primary_metric": "Overall",
             "metrics": {"Overall": 123.0}}}})
        context = run / "run_context.json"
        dump(context, {"run_dir": str(run), "native_work_dir": str(run / "native"),
             "datasets": ["MME", "MMStar"], "run_name": "method", "checkpoint": "/checkpoint",
             "checkpoint_kind": "merged/full", "architecture": "qwen2_vl", "model_name": "model",
             "suite": "requested", "created_at": "now", "vlmevalkit_commit": "abc",
             "dataset_manifest": "/manifest", "generated_config": "/config", "commands": {},
             "effective_checkpoint": "/merged", "merge_metadata": {"identity": "123"},
             "runtime": {"attention_backend": "eager"}})
        summary = json.loads(update_summary(context).read_text())
        self.assertEqual([row["dataset"] for row in summary["benchmarks"]], ["MME", "MMStar"])
        self.assertEqual(summary["benchmarks"][0]["primary_metric_value"], 123.0)
        self.assertEqual(summary["benchmarks"][0]["inference_status"], "complete")
        self.assertEqual(summary["overall_status"], "partial")
        self.assertEqual(summary["merge_metadata"]["identity"], "123")
        self.assertEqual(summary["runtime"]["attention_backend"], "eager")

    def test_generated_configs_all_architectures(self):
        expected = {"qwen2_vl": "ProjectQwen2VLChat", "qwen2_5_vl": "ProjectQwen2VLChat",
                    "fast_vlm": "FastVLMChat"}
        args = argparse.Namespace(mode="infer", attention_backend="eager", max_new_tokens=64, device_map="auto")
        for architecture, class_name in expected.items():
            with self.subTest(architecture=architecture):
                checkpoint = full_checkpoint(self.root / architecture, architecture)
                target = self.root / f"{architecture}.json"
                write_config(target, "unique", checkpoint, architecture, ["MME"], {"MME": "MMEDataset"}, args)
                cfg = json.loads(target.read_text())
                self.assertEqual(cfg["model"]["unique"]["class"], class_name)
                self.assertFalse(cfg["model"]["unique"]["do_sample"])
                self.assertEqual(cfg["data"]["MME"], {"class": "MMEDataset", "dataset": "MME"})

    def test_smoke_config_limits_each_dataset(self):
        args = argparse.Namespace(mode="infer", attention_backend="sdpa", max_new_tokens=32,
                                  max_samples=7, device_map="auto")
        checkpoint = full_checkpoint(self.root / "qwen", "qwen2_5_vl")
        target = self.root / "smoke.json"
        datasets = ["AI2D_TEST_NO_MASK", "GQA_TestDev_Balanced"]
        classes = {"AI2D_TEST_NO_MASK": "LimitedImageMCQDataset",
                   "GQA_TestDev_Balanced": "LimitedImageVQADataset"}
        write_config(target, "smoke", checkpoint, "qwen2_5_vl", datasets, classes, args)
        data = json.loads(target.read_text())["data"]
        self.assertEqual(data["AI2D_TEST_NO_MASK"]["max_samples"], 7)
        self.assertEqual(data["GQA_TestDev_Balanced"]["max_samples"], 7)

    def test_eval_mode_never_loads_model(self):
        checkpoint = full_checkpoint(self.root / "qwen", "qwen2_vl")
        args = argparse.Namespace(mode="eval", attention_backend="eager", max_new_tokens=64, device_map="auto")
        target = self.root / "eval.json"
        write_config(target, "model", checkpoint, "qwen2_vl", ["MME"], {"MME": "MMEDataset"}, args)
        self.assertEqual(json.loads(target.read_text())["model"]["model"]["class"], "OfflineScoringModel")

    def test_check_and_dry_run_make_no_writes(self):
        class Fake:
            __name__ = "FakeDataset"
            DATASET_MD5 = {}
            DATASET_URL = {}
        project = self.root / "project"; project.mkdir()
        kit = self.root / "kit"; (kit / ".git").mkdir(parents=True)
        (kit / ".git/HEAD").write_text("a" * 40)
        suite = project / "suite.json"
        dump(suite, {"name": "tiny", "datasets": ["MME"], "judge_model": None})
        cache = project / "cache"
        base = ["prepare_eval_assets.py", "--project-dir", str(project), "--vlmevalkit-dir", str(kit),
                "--lmu-data", str(cache), "--suite-config", str(suite)]
        fake_entry = ({"dataset": "MME", "resolved_class": "FakeDataset", "source": None,
                       "expected_checksum": None, "actual_checksum": None, "row_count": 1,
                       "cache_path": str(cache / "MME.tsv"), "status": "ok", "errors": []}, [])
        with mock.patch.object(prepare, "check_environment", return_value=[]), \
             mock.patch.object(prepare, "registry", return_value={"MME": Fake}), \
             mock.patch.object(prepare, "validate_dataset", return_value=fake_entry):
            for mode in ("--dry-run", "--check"):
                with self.subTest(mode=mode), mock.patch.object(sys, "argv", base + [mode]):
                    self.assertEqual(prepare.main(), 0)
        self.assertFalse(cache.exists())
        self.assertFalse((project / "outputs").exists())

        with mock.patch.object(prepare, "check_environment", return_value=[]), \
             mock.patch.object(prepare, "registry", return_value={"MME": Fake}), \
             mock.patch.object(prepare, "validate_dataset", return_value=fake_entry), \
             mock.patch.object(prepare, "prepare_base_models", return_value=([], [])):
            with mock.patch.object(sys, "argv", base + ["--offline"]):
                self.assertEqual(prepare.main(), 0)
        manifest = project / "outputs/eval/manifests/tiny.json"
        self.assertTrue(manifest.is_file())
        self.assertTrue(json.loads(manifest.read_text())["valid"])


if __name__ == "__main__":
    unittest.main()
