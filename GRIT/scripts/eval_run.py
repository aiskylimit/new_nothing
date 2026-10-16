#!/usr/bin/env python3
"""Evaluate the base model and GRIT checkpoints: safety + general capability.

Stages (each resumable, skipped when its output is complete):
  1. prepare  local snapshots -> <work-dir>/prepared/<bench>.jsonl (+ manifest)
  2. generate one vLLM subprocess per model (scripts/eval_worker.py --role generate),
              run in parallel: --slots-per-gpu engines share each GPU in --gpus
  3. guard    allenai/wildguard labels the responses to harmful prompts (harmful response = unsafe) and to
              benign prompts (response refusal = over-refusal) (--role guard), parallel too
  4. score    accuracy / extraction / length / truncation / refusal / unsafe rate
  5. compare  every model vs the first one (base) with paired bootstrap 95% CIs
Large files (generations, guard outputs) stay in --work-dir; the small result
file --results-dir/eval_results.json is what summarize_results.py reads.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_lib as E  # noqa: E402

T0 = time.time()


def tick(label):
    print(f"[{time.time() - T0:8.1f}s] {label}", flush=True)


def safe(label):
    return label.replace(":", "_").replace("/", "_")


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def by_uid(path):
    return {row["uid"]: row for row in read_jsonl(path)} if Path(path).exists() else {}


def run_worker(args, *, gpu, startup_lock, role, model, label, items, out):
    """One vLLM engine. The GPU's lock is held while the engine starts (until it reports ready) and while it
    shuts down, so no engine profiles GPU memory while another one on that GPU allocates or frees memory."""
    from scripts.eval_worker import FINISHED_MARKER, READY_MARKER

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
               GRIT_EVAL_HANDSHAKE="1")
    command = [sys.executable, str(ROOT / "scripts" / "eval_worker.py"), "--role", role, "--model", model,
               "--label", label, "--items", str(items), "--out", str(out),
               "--max-model-len", str(args.max_model_len), "--dtype", args.dtype,
               "--gpu-mem", f"{args.gpu_mem / args.slots_per_gpu:.3f}", "--max-num-seqs", str(args.max_num_seqs)]
    startup_lock.acquire()
    locked, process = True, None
    try:
        tick(f"{role}: {label} on GPU {gpu} ({model})")
        process = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(f"[{label}/{role}] {line}", end="", flush=True)
            if locked and READY_MARKER in line:
                startup_lock.release()
                locked = False
            elif not locked and FINISHED_MARKER in line:
                startup_lock.acquire()  # wait for any engine still starting on this GPU, then let this one exit
                locked = True
                try:
                    process.stdin.write("exit\n")
                    process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
        code = process.wait()
        if locked:
            time.sleep(3)  # let the driver reclaim the exited engine's memory before the next startup profiles
        if code != 0:
            raise RuntimeError(f"{role} worker for {label} failed (exit {code})")
    finally:
        if process is not None and process.stdin:
            try:
                process.stdin.close()
            except OSError:
                pass
        if locked:
            startup_lock.release()


def run_parallel(args, jobs):
    """Run worker jobs over --gpus x --slots-per-gpu engines; the first failure stops new launches."""
    if not jobs:
        return
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    startup_locks = {gpu: threading.Lock() for gpu in gpus}
    pending, guard, errors = list(jobs), threading.Lock(), []

    def slot(gpu):
        while True:
            with guard:
                if not pending or errors:
                    return
                job = pending.pop(0)
            try:
                run_worker(args, gpu=gpu, startup_lock=startup_locks[gpu], **job)
            except Exception as error:  # noqa: BLE001 - re-raised below after all slots stop
                errors.append(error)

    # GPU-major order: with fewer jobs than slots, jobs spread over all GPUs before two share a GPU.
    threads = [threading.Thread(target=slot, args=(gpu,)) for _ in range(args.slots_per_gpu) for gpu in gpus]
    for thread in threads[:len(jobs)]:
        thread.start()
    for thread in threads[:len(jobs)]:
        thread.join()
    if errors:
        raise errors[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, metavar="LABEL=PATH",
                        help="Repeatable; the first one is the reference for paired deltas (base)")
    parser.add_argument("--raw-root", required=True, help="Folder with the benchmark snapshots from download.txt")
    parser.add_argument("--safety-file", required=True, help="GRIT artifacts data/eval/evaluation_prompts.parquet")
    parser.add_argument("--guard-path", required=True, help="Local allenai/wildguard snapshot")
    parser.add_argument("--work-dir", required=True, help="Large outputs (generations, guard labels, scores)")
    parser.add_argument("--results-dir", required=True, help="Small outputs: eval_results.json")
    parser.add_argument("--benchmarks", nargs="+", default=E.ALL_BENCHMARKS, choices=E.ALL_BENCHMARKS)
    parser.add_argument("--max-samples", type=int, default=None, help="Random subset per benchmark (smoke test)")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--gpus", default="0", help="Comma list of GPUs for eval engines")
    parser.add_argument("--slots-per-gpu", type=int, default=3,
                        help="Concurrent vLLM engines per GPU; each gets --gpu-mem / slots of the memory")
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=16384,
                        help="Guard context (fits a prompt + the longest judged response, 4096 tokens); generate "
                             "engines raise it to fit the longest prompt + budget")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    parser.add_argument("--lcb-timeout", type=int, default=6, help="Official LiveCodeBench per-test timeout (s)")
    parser.add_argument("--lcb-workers", type=int, default=max(2, min(32, os.cpu_count() or 4)))
    args = parser.parse_args()

    models = {}
    for spec in args.model:
        label, _, path = spec.partition("=")
        if not label or not path:
            parser.error(f"--model expects LABEL=PATH, got {spec!r}")
        models[label] = path
    reference = next(iter(models))
    work, results = Path(args.work_dir), Path(args.results_dir)
    for directory in (work / "prepared", work / "generations", work / "wildguard", work / "scores", results):
        directory.mkdir(parents=True, exist_ok=True)

    # 1. prepare ---------------------------------------------------------------
    tick(f"[1/5] prepare {args.benchmarks}")
    manifest = E.prepare_all(args.benchmarks, work / "prepared", raw_root=args.raw_root,
                             safety_file=args.safety_file, max_samples=args.max_samples,
                             seed=args.sample_seed, log=lambda message: tick(message))
    data = E.load_prepared(work / "prepared", args.benchmarks)
    items_path = work / "items.jsonl"
    with open(items_path, "w", encoding="utf-8") as handle:
        for bench in args.benchmarks:
            for row in data[bench]:
                handle.write(json.dumps({"uid": row["id"], "mode": row["mode"], "prompt": row["prompt"],
                                         "raw_prompt": bool(row.get("raw_prompt")),
                                         "n_options": row.get("n_options"),
                                         "max_tokens": E.MAX_NEW_TOKENS.get(bench, 1)}, ensure_ascii=False) + "\n")
    n_items = sum(len(rows) for rows in data.values())
    tick(f"{n_items} items: " + ", ".join(f"{bench}={len(rows)}" for bench, rows in data.items()))

    # 2. generate --------------------------------------------------------------
    tick(f"[2/5] generate for {list(models)}")
    jobs = []
    uids = {row["id"] for rows in data.values() for row in rows}
    for label, path in models.items():
        out = work / "generations" / f"{safe(label)}.jsonl"
        # Count only current items: generations of items no longer evaluated (e.g. an earlier protocol's ids)
        # must not make an incomplete file look complete.
        if not uids <= by_uid(out).keys():
            jobs.append({"role": "generate", "model": path, "label": label, "items": items_path, "out": out})
    run_parallel(args, jobs)
    gens = {}
    for label in models:
        gens[label] = by_uid(work / "generations" / f"{safe(label)}.jsonl")
        missing = [uid for bench in data for uid in (row["id"] for row in data[bench]) if uid not in gens[label]]
        if missing:
            raise RuntimeError(f"{label}: {len(missing)} items without generation, e.g. {missing[:3]}")

    # 3. guard -----------------------------------------------------------------
    guarded = [bench for bench in E.GUARDED if bench in data]
    guards = {label: {} for label in models}
    if guarded:
        tick(f"[3/5] WildGuard on {guarded}")
        jobs = []
        for label in models:
            pairs = [{"uid": row["id"], "prompt": row["prompt"], "response": gens[label][row["id"]]["response"]}
                     for bench in guarded for row in data[bench]]
            pairs_path = work / "wildguard" / f"{safe(label)}.pairs.jsonl"
            with open(pairs_path, "w", encoding="utf-8") as handle:
                for pair in pairs:
                    handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
            out = work / "wildguard" / f"{safe(label)}.jsonl"
            if len(by_uid(out)) < len(pairs):
                jobs.append({"role": "guard", "model": args.guard_path, "label": label,
                             "items": pairs_path, "out": out})
        run_parallel(args, jobs)
        for label in models:
            out = work / "wildguard" / f"{safe(label)}.jsonl"
            guards[label] = {uid: E.parse_guard_label(row["guard_output"]) for uid, row in by_uid(out).items()}

    # 4. score -----------------------------------------------------------------
    scored = [bench for bench in E.SCORED if bench in data]
    tick(f"[4/5] score {scored}")
    scores = {}
    for label in models:
        path = work / "scores" / f"{safe(label)}.json"
        cache = json.loads(path.read_text()) if path.exists() else {}
        for bench in scored:
            todo = [row for row in data[bench] if row["id"] not in cache]
            if not todo:
                continue
            started = time.time()
            if bench == "livecodebench":
                # Official LiveCodeBench codegen_metrics: sandboxed processes, per-test timeout, pass@1.
                out = E.score_livecodebench(todo, [gens[label][row["id"]]["response"] for row in todo],
                                            workers=args.lcb_workers, timeout=args.lcb_timeout)
            else:
                # math-verify uses signal timeouts, so it must run in the main thread.
                out = [E.score_item(row, gens[label][row["id"]]) for row in todo]
            for row, result in zip(todo, out, strict=True):
                cache[row["id"]] = result
            path.write_text(json.dumps(cache))
            tick(f"{label} {bench}: {len(todo)} items in {time.time() - started:.0f}s")
        scores[label] = cache

    # 5. metrics + paired deltas -------------------------------------------------
    tick("[5/5] metrics and paired deltas")

    def indicator(label, bench, row):
        """Per-item 0/1 value used for paired deltas."""
        if bench in scored:
            return scores[label][row["id"]]["score"]
        guard = guards[label].get(row["id"], {})
        if bench in E.HARMFUL:
            return int(guard.get("harmful_response") is True)  # WildGuard "Harmful response: yes" = unsafe
        return int(guard.get("refusal") is True)  # benign prompts: "Response refusal: yes" = over-refusal

    metrics = []
    for label in models:
        for bench in args.benchmarks:
            rows = data[bench]
            generations = [gens[label][row["id"]] for row in rows]
            entry = {"model": label, "benchmark": bench, "n": len(rows),
                     "prompt_too_long": sum(g["finish_reason"] == "prompt_too_long" for g in generations)}
            if bench in scored:
                values = [scores[label][row["id"]] for row in rows]
                entry["accuracy_%"] = 100 * sum(v["score"] for v in values) / len(values)
                entry["extracted_%"] = 100 * sum(v["extracted"] for v in values) / len(values)
            if rows[0]["mode"] == "gen":
                entry["mean_tokens"] = sum(g["n_tokens"] for g in generations) / len(generations)
                entry["truncated_%"] = 100 * sum(g["finish_reason"] == "length" for g in generations) / len(generations)
                entry["refusal_regex_%"] = 100 * sum(E.is_refusal(g["response"]) for g in generations) / len(generations)
            if bench in guarded:
                labels = [guards[label].get(row["id"], {}) for row in rows]
                entry["unsafe_%"] = 100 * sum(g.get("harmful_response") is True for g in labels) / len(rows)
                entry["refusal_%"] = 100 * sum(g.get("refusal") is True for g in labels) / len(rows)
                entry["judge_parse_fail_%"] = 100 * sum(
                    g.get("harmful_response") is None or g.get("refusal") is None for g in labels) / len(rows)
            metrics.append(entry)

    deltas = []
    for label in models:
        if label == reference:
            continue
        for bench in [*scored, *guarded]:
            rows = data[bench]
            delta = E.paired_delta([indicator(reference, bench, row) for row in rows],
                                   [indicator(label, bench, row) for row in rows])
            metric = "accuracy" if bench in scored else ("unsafe" if bench in E.HARMFUL else "over_refusal")
            deltas.append({"model": label, "reference": reference, "benchmark": bench, "metric": metric, **delta})

    payload = {"models": models, "reference": reference, "benchmarks": args.benchmarks,
               "config": {"max_samples": args.max_samples, "seed": args.sample_seed,
                          "max_new_tokens": E.MAX_NEW_TOKENS, "max_model_len_min": args.max_model_len,
                          "dtype": args.dtype, "guard": args.guard_path, "gpus": args.gpus,
                          "slots_per_gpu": args.slots_per_gpu},
               "manifest": manifest, "metrics": metrics, "paired_deltas": deltas}
    (results / "eval_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    tick(f"wrote {results / 'eval_results.json'}")
    tick("ALL DONE")


if __name__ == "__main__":
    main()
