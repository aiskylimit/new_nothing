import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from opd.b200 import make_plan, artifact_complete
from opd.benchmark_plan import add_instruction_eval
from opd.benchmark_results import collect_benchmark_results
from opd.benchmarks import BENCHMARKS, ALL_BENCHMARKS, normalized_rows, prepare
from opd.data import digest, read_jsonl, save_json, write_jsonl
from opd.dolly_eval import main as eval_main, score_response
from opd.dolly_results import read_dolly_result


class InstructionBenchmarkTests(unittest.TestCase):
    def plan(self, root):
        args = argparse.Namespace(project_root=str(root), run_name="fixture", profile="llama2-tinyllama",
            domains=["dolly-dskd"], seeds=[10], bi_seeds=[42], python_bin="python",
            steps=2, grad_accum=32, max_minutes=0, eval_max_new_tokens=256,
            math_eval=False, dolly_eval=True)
        return make_plan(args)

    def test_prepare_preserves_multiple_references_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw/instruction_benchmarks/sinst/11_/valid.jsonl"
            write_jsonl(raw, [{"prompt": "question<n>answer:", "output": ["yes", "correct"]}])
            spec = {**BENCHMARKS["sinst"], "n_samples": 1,
                    "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}
            with patch.dict(BENCHMARKS, {"sinst": spec}):
                prepare(root, ["sinst"])
                rows = read_jsonl(root / "data/opd/benchmarks/sinst/test.jsonl")
                self.assertEqual(rows[0]["response"], ["yes", "correct"])
                self.assertEqual(rows[0]["prompt"], "question\nanswer:")
                self.assertEqual(rows[0]["prompt_format"], "raw")
                raw.write_text(raw.read_text() + "\n")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    normalized_rows(raw, "sinst")

    def test_migrate_suite_keeps_training_and_dolly_jobs_and_is_idempotent(self):
        plan = self.plan("/tmp/benchmark-fixture")
        updated = add_instruction_eval(plan, batch_size=8)
        self.assertEqual([j for j in plan["jobs"] if j["stage"] == "train"],
                         [j for j in updated["jobs"] if j["stage"] == "train"])
        self.assertEqual([j for j in plan["jobs"] if j.get("eval_kind") == "dolly"],
                         [j for j in updated["jobs"] if j.get("eval_kind") == "dolly"])
        self.assertEqual(updated, add_instruction_eval(updated, batch_size=8))
        self.assertEqual(len(updated["cases"]), 2)
        self.assertEqual(len(updated["benchmark_cases"]), 6)
        self.assertEqual({c["benchmark"] for c in updated["benchmark_cases"]}, set(BENCHMARKS))
        self.assertNotIn("uinst", json.dumps(updated))
        for job in updated["jobs"]:
            if job.get("eval_kind") == "instruction":
                self.assertEqual(job["command"][job["command"].index("--batch-size") + 1], "8")

    def test_average_requires_full_suite_and_same_checkpoint(self):
        plan = add_instruction_eval(self.plan("/tmp/benchmark-fixture"))
        protocol = dict(decoding_seeds=[10, 20], batch_size=8, max_new_tokens=256,
                        max_prompt_length=256, do_sample=True, temperature=1., top_p=1., top_k=0, dtype="bfloat16")
        def result(name, score=20., complete=True):
            return {"kind": name, "checkpoint_step": 2, "training_complete": complete,
                    "training_steps_budget": 2, "protocol": {**protocol, "benchmark": name,
                        "scope": f"full_{name}_test", "n_samples": 2},
                    "summary": {m: {"mean": score, "std_decoding_seeds": 0.} for m in ("rougeL", "exact_match")},
                    "per_seed": [{"seed": s, "n_samples": 2, "rougeL": score, "exact_match": score} for s in [10, 20]]}
        dolly = [{"domain": "dolly-dskd", "method": m, "training_seed": 10,
                  "checkpoint_step": 2, "training_complete": True, "training_steps_budget": 2,
                  "rougeL_mean": 40., "exact_match_mean": 40., "scope": "full_dolly_test", "protocol": protocol}
                 for m in ("opd", "opd_impact")]
        with patch("opd.benchmark_results.read_dolly_result", side_effect=lambda c: result(c["benchmark"])):
            suite = collect_benchmark_results(plan, dolly)
        self.assertEqual(len(suite["wide"]), 2)
        self.assertEqual(suite["wide"][0]["avg_rougeL"], 25.)
        self.assertEqual(len(suite["paired_deltas"]), 4)
        with patch("opd.benchmark_results.read_dolly_result", side_effect=lambda c: None if c["benchmark"] == "sinst" else result(c["benchmark"])):
            incomplete = collect_benchmark_results(plan, dolly)
        self.assertTrue(all(r["avg_rougeL"] is None for r in incomplete["wide"]))
        with patch("opd.benchmark_results.read_dolly_result", side_effect=lambda c: {**result(c["benchmark"]), "checkpoint_step": 1}):
            stale = collect_benchmark_results(plan, dolly)
        self.assertTrue(all(r["avg_rougeL"] is None for r in stale["wide"]))
        with patch("opd.benchmark_results.read_dolly_result", side_effect=lambda c: result(c["benchmark"], complete=False)):
            partial = collect_benchmark_results(plan, dolly)
        self.assertTrue(all(r["avg_rougeL"] is None for r in partial["wide"]))

    @unittest.skipUnless(importlib.util.find_spec("rouge_score"), "Install requirements-opd-dolly-eval.txt")
    def test_train_eval_executes_only_impact_then_all_four_evaluations(self):
        from opd.b200 import main as pipeline_main
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in BENCHMARKS:
                data = root / "data/opd/benchmarks" / name
                write_jsonl(data / "test.jsonl", [])
                save_json(data / "manifest.json", {})
            argv = ["b200", "train-eval", "--project-root", temp, "--run-name", "fixture", "--fast", "--only-method", "opd_impact"]
            with patch("sys.argv", argv), patch("opd.b200.preflight"), patch("opd.b200.verify_model"), \
                    patch("opd.b200.freeze_inputs"), patch("transformers.AutoTokenizer.from_pretrained"), \
                    patch("opd.losses.require_shared_tokenizer"), patch("opd.b200.execute_job") as execute, \
                    contextlib.redirect_stdout(io.StringIO()):
                pipeline_main()
            jobs = [call.args[0] for call in execute.call_args_list]
            self.assertEqual(len(jobs), 5)
            self.assertEqual(jobs[0]["stage"], "train")
            self.assertIn("opd_impact", jobs[0]["command"])
            self.assertTrue(all(j["stage"] == "eval" and j["method"] == "opd_impact" for j in jobs[1:]))
            self.assertEqual({j.get("benchmark", "dolly") for j in jobs[1:]}, set(ALL_BENCHMARKS))

    @unittest.skipUnless(importlib.util.find_spec("rouge_score"), "Install requirements-opd-dolly-eval.txt")
    def test_real_cpu_eval_multireference_predictions_and_result_validation(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from tests_opd.test_pipeline import tokenizer
        from rouge_score.rouge_scorer import RougeScorer
        scorer = RougeScorer(["rougeL"], use_stemmer=True)
        self.assertEqual(score_response("world", ["hello", "world"], scorer), {"rougeL": 100., "exact_match": 100.})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            torch.set_num_threads(1)
            tok = tokenizer()
            model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=16,
                intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                max_position_embeddings=64, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id))
            model.save_pretrained(root / "model")
            tok.save_pretrained(root / "model")
            data = root / "test.jsonl"
            write_jsonl(data, [{"id": "a", "prompt": "hello", "prompt_format": "raw", "response": ["hello", "world"]},
                               {"id": "b", "prompt": "hello world", "prompt_format": "raw", "response": "world"}])
            save_json(root / "manifest.json", {**BENCHMARKS["sinst"], "benchmark": "sinst", "prepared_sha256": digest(data.read_text())})
            argv = ["eval", "--benchmark", "sinst", "--model", str(root / "model"), "--data", str(data),
                    "--out", str(root / "eval"), "--expected-samples", "2", "--device", "cpu", "--dtype", "float32",
                    "--batch-size", "2", "--max-new-tokens", "3", "--max-prompt-length", "8", "--seeds", "10", "20"]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                eval_main()
            case = {"eval_dir": str(root / "eval")}
            result = read_dolly_result(case)
            self.assertEqual(result["kind"], "sinst")
            self.assertEqual(result["protocol"]["scope"], "custom_test")
            self.assertEqual([s["n_samples"] for s in result["per_seed"]], [2, 2])
            self.assertTrue(artifact_complete({"stage": "eval", "eval_kind": "instruction",
                "artifact": str(root / "eval/evaluation_manifest.json")}))
            predictions = root / "eval/predictions-seed10.jsonl"
            records = read_jsonl(predictions)
            self.assertEqual(records[0]["reference"], ["hello", "world"])
            write_jsonl(predictions, records[:1])
            with self.assertRaisesRegex(ValueError, "coverage"):
                read_dolly_result(case)


if __name__ == "__main__":
    unittest.main()
