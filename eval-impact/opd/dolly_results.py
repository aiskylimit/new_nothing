"""Validate complete, current-checkpoint Dolly evaluations before reporting them."""
import json
import math
import statistics
from pathlib import Path
from .data import digest, read_jsonl
from .recipes import adapter_signature


def read_dolly_result(case):
    folder = Path(case["eval_dir"])
    if not (folder / "evaluation_manifest.json").exists() or not (folder / "results.json").exists():
        return None
    manifest = json.loads((folder / "evaluation_manifest.json").read_text())
    result = json.loads((folder / "results.json").read_text())
    if manifest.get("state") != "completed" or result.get("state") != "completed":
        return None
    if manifest["protocol"] != result["protocol"]:
        raise ValueError("Dolly result/manifest protocol mismatch")
    for key in ("checkpoint", "checkpoint_step", "adapter_sha256", "training_complete"):
        if manifest.get(key) != result.get(key):
            raise ValueError(f"Dolly result/manifest mismatch: {key}")
    if case.get("train_dir"):
        latest_path = Path(case["train_dir"]) / "latest.json"
        if not latest_path.exists():
            return None
        latest = json.loads(latest_path.read_text())
        if str(Path(latest["checkpoint"]).resolve()) != result["checkpoint"] or latest["step"] != result["checkpoint_step"]:
            return None  # Training has advanced; evaluate the new checkpoint.
        if Path(result["checkpoint"]).exists() and adapter_signature(result["checkpoint"]) != result["adapter_sha256"]:
            return None
    protocol = result["protocol"]
    if [r["seed"] for r in result["per_seed"]] != protocol["decoding_seeds"]:
        raise ValueError("Dolly decoding seed coverage mismatch")
    for seed_result in result["per_seed"]:
        predictions_path = folder / f"predictions-seed{seed_result['seed']}.jsonl"
        if not predictions_path.exists():
            return None
        predictions = read_jsonl(predictions_path)
        if len(predictions) != protocol["n_samples"] or seed_result["n_samples"] != len(predictions):
            raise ValueError("Dolly prediction coverage mismatch")
        ids = [r["id"] for r in predictions]
        if len(set(ids)) != len(ids) or digest(json.dumps(ids)) != protocol["sample_ids_sha256"]:
            raise ValueError("Dolly prediction IDs differ from the test set")
        for metric in ("rougeL", "exact_match"):
            values = [r[metric] for r in predictions]
            if any(not math.isfinite(v) or not 0 <= v <= 100 for v in values):
                raise ValueError("Invalid Dolly metric value")
            if not math.isclose(statistics.mean(values), seed_result[metric], abs_tol=1e-6):
                raise ValueError("Dolly per-seed metric differs from predictions")
    for metric in ("rougeL", "exact_match"):
        values = [r[metric] for r in result["per_seed"]]
        if not math.isclose(result["summary"][metric]["mean"], statistics.mean(values), abs_tol=1e-6):
            raise ValueError("Dolly summary differs from seed results")
        expected_std = statistics.stdev(values) if len(values) > 1 else None
        actual_std = result["summary"][metric]["std_decoding_seeds"]
        if expected_std is None and actual_std is not None or expected_std is not None and (
                actual_std is None or not math.isclose(actual_std, expected_std, abs_tol=1e-6)):
            raise ValueError("Dolly decoding standard deviation mismatch")
    return result


def dolly_rows(case, result):
    row = {"domain": case["domain"], "method": case["method"], "training_seed": case["seed"],
           "checkpoint_step": result["checkpoint_step"], "training_complete": result["training_complete"],
           "training_steps_budget": result.get("training_steps_budget"),
           "n_samples": result["protocol"]["n_samples"], "n_decoding_seeds": len(result["per_seed"]),
           "rougeL_mean": result["summary"]["rougeL"]["mean"],
           "rougeL_std_decoding_seeds": result["summary"]["rougeL"]["std_decoding_seeds"],
           "exact_match_mean": result["summary"]["exact_match"]["mean"],
           "scope": result["protocol"]["scope"], "protocol": result["protocol"],
           "result_file": str(Path(case["eval_dir"]) / "results.json")}
    per_seed = [{"domain": case["domain"], "method": case["method"], "training_seed": case["seed"],
                 "decoding_seed": s["seed"], "n_samples": s["n_samples"],
                 "rougeL": s["rougeL"], "exact_match": s["exact_match"]} for s in result["per_seed"]]
    return row, per_seed


def paired_dolly_deltas(rows):
    by_case = {(r["domain"], r["training_seed"], r["method"]): r for r in rows}
    deltas = []
    for row in rows:
        if row["method"] != "opd_impact":
            continue
        base = by_case.get((row["domain"], row["training_seed"], "opd"))
        if base is None:
            continue
        if base["protocol"] != row["protocol"]:
            raise ValueError("Cannot compare Dolly evaluations with different test/decoding protocols")
        deltas.append({"domain": row["domain"], "training_seed": row["training_seed"],
            "opd_step": base["checkpoint_step"], "impact_step": row["checkpoint_step"],
            "opd_rougeL": base["rougeL_mean"], "impact_rougeL": row["rougeL_mean"],
            "delta_rougeL_points": row["rougeL_mean"] - base["rougeL_mean"],
            "delta_exact_match_points": row["exact_match_mean"] - base["exact_match_mean"],
            "matched_full_training_budget": base["training_complete"] and row["training_complete"] and
                base["training_steps_budget"] == row["training_steps_budget"]})
    return deltas
