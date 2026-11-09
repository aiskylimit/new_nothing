"""Offline, resumable CORED pipeline on three B200s; one timing record per stage."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from core.data_utils import read_jsonl, write_json, write_jsonl
from core.local_assets import asset_path, local_model_path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("prepare", "prefix", "key", "phase1", "weights", "merge", "distill")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(part)
    return hasher.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/Qwen2.5-7B-Instruct")
    p.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2"))
    p.add_argument("--run_id", default=None)
    p.add_argument("--output_root", default="outputs/b200")
    p.add_argument("--dataset", default="datasets/s1K-1.1")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--global_batch", type=int, default=33)
    p.add_argument("--micro_batch_size", type=int, default=11)
    p.add_argument("--max_batch_tokens", type=int, default=32768)
    p.add_argument("--cutoff_len", type=int, default=16384)
    p.add_argument("--loss_chunk", type=int, default=256)
    p.add_argument("--entropy_chunk", type=int, default=2048)
    p.add_argument("--key_frac", type=float, default=0.1)
    p.add_argument("--lr_phase1", type=float, default=5e-5)
    p.add_argument("--lr_distill", type=float, default=1e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--attn_implementation", choices=["sdpa", "flash_attention_2"], default="sdpa")
    p.add_argument("--checkpoint_every", type=int, default=0)
    p.add_argument("--no_gradient_checkpointing", action="store_true")
    p.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    p.add_argument("--eval", action="store_true", help="Evaluate final student after training")
    p.add_argument("--eval_n", type=int, default=3)
    p.add_argument("--dry_run", type=int, default=0, help="Stop after N updates; rerun without this flag to resume")
    p.add_argument("--development_smoke", action="store_true", help="Isolated short test on two GPUs; requires --dry_run")
    args = p.parse_args()
    if args.epochs < 1 or args.global_batch < 1 or not 0 < args.key_frac <= 1:
        p.error("Invalid epochs/batch/key_frac")
    return args


def merge_prefix_shards(source, parts, destination):
    expected = [str(row["id"]) for row in read_jsonl(str(source))]
    merged = {}
    for part in parts:
        for row in read_jsonl(str(part)):
            key = str(row["id"])
            if key in merged:
                raise ValueError(f"Duplicate prefix ID: {key}")
            merged[key] = row
    if set(expected) != set(merged) or len(set(expected)) != len(expected):
        raise ValueError("Prefix shards do not exactly cover source IDs")
    write_jsonl(str(destination), [merged[key] for key in expected])


def validate_npz_coverage(source, directory, required):
    ids = [str(row["id"]) for row in read_jsonl(str(source))]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate source IDs")
    actual = {path.stem for path in directory.glob("*.npz")}
    if actual != set(ids):
        raise ValueError(f"Incomplete/stale cache in {directory}: missing={len(set(ids) - actual)}, extra={len(actual - set(ids))}")
    for key in ids:
        with np.load(directory / f"{key}.npz") as cached:
            if not set(required).issubset(cached.files):
                raise ValueError(f"Invalid cache: {directory / key}")
            for field in required:
                if not np.isfinite(cached[field]).all():
                    raise ValueError(f"Nonfinite {field}: {key}")


def main():
    args = parse_args()
    os.chdir(ROOT)
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false")
    model = local_model_path(args.model)
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if args.development_smoke and (len(gpus) != 2 or args.dry_run <= 0 or args.eval):
        raise ValueError("Development smoke requires two GPUs, positive --dry_run, and no legacy --eval")
    world = 2 if args.development_smoke else 3
    if len(gpus) != world or len(set(gpus)) != world:
        raise ValueError("Specify exactly three distinct GPU IDs with --gpus (default 0,1,2)")
    if args.run_id and (args.run_id in {".", ".."} or Path(args.run_id).name != args.run_id):
        raise ValueError("run_id must be a single folder name")
    run_id = args.run_id or f"{Path(model).name}_seed{args.seed}"
    run = (Path(args.output_root) / run_id).resolve()
    run.mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(exist_ok=True)
    dataset = asset_path(args.dataset)
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    sources = ([dataset] if dataset.is_file() else sorted(
        p for p in dataset.rglob("*") if p.is_file() and p.suffix in {".parquet", ".jsonl", ".json", ".arrow"}))
    runtime_keys = {"stages", "eval", "eval_n", "micro_batch_size", "max_batch_tokens", "loss_chunk",
                    "checkpoint_every", "dry_run", "gpus"}
    config = {key: value for key, value in vars(args).items() if key not in runtime_keys}
    implementation = [ROOT / name for name in (
        "train/b200.py", "train/parallel.py", "train/pipeline_b200.py", "train/common.py",
        "train/compute_weights.py", "core/efficient_loss.py", "core/segmentation.py", "core/weighting.py",
        "core/model_utils.py", "core/rep_loss.py", "core/council.py", "core/data_utils.py",
        "proposal_v2/svgd.py", "proposal_v2/prefix_truncate.py", "proposal_v2/keystep_select.py",
        "proposal/merge.py", "data/prepare_s1k.py")]
    model_metadata = {p.name: digest(p) for p in Path(model).iterdir() if p.is_file() and
                      ("token" in p.name or p.name in {"chat_template.jinja", "vocab.json", "merges.txt", "generation_config.json"})}
    weight_metadata = {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                       for p in Path(model).iterdir() if p.is_file() and p.suffix in {".safetensors", ".bin"}}
    config.update(model=model, dataset=str(dataset), model_config_sha256=digest(Path(model) / "config.json"),
                  source_sha256={str(p.relative_to(dataset)): digest(p) for p in sources},
                  implementation_sha256={str(p.relative_to(ROOT)): digest(p) for p in implementation},
                  tokenizer_sha256=model_metadata, weight_file_metadata=weight_metadata)
    manifest = run / "pipeline_config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError(f"Run configuration differs; use another --run_id: {run}")
    manifest.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n")
    with (run / "invocations.jsonl").open("a") as handle:
        handle.write(json.dumps({"unix_time": time.time(), "arguments": vars(args)}) + "\n")
    fingerprint = digest(manifest)
    print(f"[cored] model={model} GPUs={gpus} batch={args.global_batch} run={run}", flush=True)
    train = run / "train_s1k.jsonl"
    prefixed = run / "train_s1k_prefixed.jsonl"
    key = run / "keysteps"
    phase1 = run / "phase1"
    weights = run / "weights"
    merged = run / "merge_ties"
    student = run / "distill"
    python = sys.executable
    base_env = os.environ.copy()
    base_env.setdefault("OMP_NUM_THREADS", "4")
    base_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    worker_accounting = {}
    # Independent inference workers must not inherit torchrun rendezvous state.
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        base_env.pop(name, None)

    def execute(commands, stage):
        processes, handles, worker_starts, worker_ends = [], [], [], {}
        try:
            for rank, (command, devices) in enumerate(commands):
                env = {**base_env, "CUDA_VISIBLE_DEVICES": devices}
                if base_env.get("REBUTTAL_STAGE_DIR"):
                    directory = Path(base_env["REBUTTAL_STAGE_DIR"]) / stage
                    directory.mkdir(parents=True, exist_ok=True)
                    env["REBUTTAL_STAGE_DIR"] = str(directory)
                handle = (run / "logs" / f"{stage}-{rank}.log").open("a")
                handles.append(handle)
                handle.write("\n" + json.dumps(command) + "\n")
                handle.flush()
                processes.append(subprocess.Popen([str(arg) for arg in command], cwd=ROOT, env=env,
                                                   stdout=handle, stderr=subprocess.STDOUT, start_new_session=True))
                worker_starts.append(time.perf_counter())
            pending = set(range(len(processes)))
            while pending:
                for rank in list(pending):
                    result = processes[rank].poll()
                    if result is None:
                        continue
                    pending.remove(rank)
                    worker_ends[rank] = time.perf_counter()
                    if result:
                        raise RuntimeError(f"{stage} worker {rank} failed ({result}); see {run / 'logs' / f'{stage}-{rank}.log'}")
                if pending:
                    time.sleep(0.2)
        finally:
            for process in processes:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for process in processes:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            for handle in handles:
                handle.close()
            worker_accounting[stage] = [{'seconds': worker_ends.get(i, time.perf_counter())-t,
                 'gpu_count': 0 if stage == 'prepare' else len(commands[i][1].split(','))}
                for i,t in enumerate(worker_starts)]

    def one(module, arguments, stage, devices=None):
        execute([([python, "-m", module, *map(str, arguments)], devices or gpus[0])], stage)

    def sharded(module, arguments, stage):
        execute([([python, "-m", module, *map(str, arguments), "--shard_id", str(i), "--num_shards", str(world)], gpu)
                 for i, gpu in enumerate(gpus)], stage)

    def training(stage, destination, extra):
        command = [python, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc_per_node={world}",
                   "-m", "train.b200", "--stage", stage, "--model_name", model,
                   "--train_file", str(prefixed), "--out_dir", str(destination), "--epochs", str(args.epochs),
                   "--seed", str(args.seed), "--global_batch", str(args.global_batch),
                   "--micro_batch_size", str(args.micro_batch_size), "--max_batch_tokens", str(args.max_batch_tokens),
                   "--cutoff_len", str(args.cutoff_len), "--loss_chunk", str(args.loss_chunk),
                   "--attn_implementation", args.attn_implementation, "--resume", "--dry_run", str(args.dry_run),
                   "--checkpoint_every", str(args.checkpoint_every), *map(str, extra)]
        if args.no_gradient_checkpointing:
            command.append("--no_gradient_checkpointing")
        execute([(command, ",".join(gpus))], stage)

    def stage(name, action, validate):
        if name not in args.stages:
            return
        marker = run / f"{name}.done.json"
        if marker.exists():
            if json.loads(marker.read_text())["fingerprint"] != fingerprint:
                raise ValueError(f"Stale stage marker: {marker}")
            validate()
            print(f"[cored] reuse {name}", flush=True)
            return
        print(f"[cored] start {name}; logs: {run / 'logs'}", flush=True)
        started = time.perf_counter()
        try:
            action()
            validate()
        except BaseException:
            with (run / "timings.jsonl").open("a") as handle:
                handle.write(json.dumps({"stage": name, "seconds": time.perf_counter() - started, "status": "failed",
                    "telemetry_dir": str(Path(base_env['REBUTTAL_STAGE_DIR']) / name) if base_env.get('REBUTTAL_STAGE_DIR') else None}) + "\n")
            raise
        dry = bool(args.dry_run and name in {"phase1", "distill"})
        record = {"stage": name, "seconds": time.perf_counter() - started,
                  "status": "dry_run" if dry else "complete", "fingerprint": fingerprint,
                  "telemetry_dir": str(Path(base_env['REBUTTAL_STAGE_DIR']) / name) if base_env.get('REBUTTAL_STAGE_DIR') else None}
        record['workers'] = worker_accounting.get(name, [])
        record['gpu_hours'] = sum(w['seconds']*w['gpu_count']/3600 for w in record['workers'])
        if not dry:
            write_json(str(marker), record)
        with (run / "timings.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"[cored] done {name}: {record['seconds']:.1f}s", flush=True)

    def require(path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    if args.eval or any(name != "prepare" for name in args.stages):
        one("train.hardware_probe", ["--out", run / "hardware.json", "--expected_devices", world], "hardware", ",".join(gpus))

    stage("prepare", lambda: execute([([python, "data/prepare_s1k.py", "--model_name", model,
                                        "--dataset", str(dataset), "--out", str(train)], gpus[0])], "prepare"),
          lambda: require(train))
    parts = [run / f"prefix-{i}.jsonl" for i in range(world)]

    def prefix():
        options = ["--max_model_len", "2048", "--max_tokens", "256", "--n", "1", "--development_smoke"] if args.development_smoke else []
        execute([([python, "-m", "proposal_v2.prefix_truncate", "--model", model, "--in", str(train),
                   "--out", str(parts[i]), "--gpu_mem", "0.9", "--shard_id", str(i), "--num_shards", str(world), *options], gpu)
                 for i, gpu in enumerate(gpus)], "prefix")
        merge_prefix_shards(train, parts, prefixed)

    stage("prefix", prefix, lambda: require(prefixed))
    stage("key", lambda: sharded("proposal_v2.keystep_select", ["--model_name", model,
          "--train_file", prefixed, "--out_dir", key, "--cutoff_len", args.cutoff_len,
          "--seed", args.seed, "--frac", args.key_frac, "--chunk_size", args.entropy_chunk], "key"),
          lambda: validate_npz_coverage(prefixed, key, ["H", "key_mask"]))
    stage("phase1", lambda: training("phase1", phase1, ["--keystep_dir", key, "--lr", args.lr_phase1,
          "--M", "3", "--lora_r", args.lora_r, "--lora_alpha", args.lora_alpha, "--lora_dropout", args.lora_dropout]),
          lambda: [require(phase1 / f"expert_{i}" / "adapter_model.safetensors") for i in range(3)])
    if args.dry_run and "phase1" in args.stages:
        print("[cored] dry-run: Phase 1 checkpoint saved; stop before council/merge/distillation", flush=True)
        return

    def council():
        sharded("train.compute_weights", ["--model_name", model, "--train_file", prefixed,
                "--phase1_dir", phase1, "--weights_dir", weights, "--M", "3",
                "--cutoff_len", args.cutoff_len, "--seed", args.seed,
                "--disagreement_mode", "normalized", "--chunk_size", args.entropy_chunk,
                "--no_cache_expert_losses"], "weights")
        from train.compute_weights import _summarize
        from types import SimpleNamespace
        arrays = {key: [] for key in ("U", "V", "w")}
        for path in sorted(weights.glob("*.npz")):
            with np.load(path) as data:
                for key in arrays:
                    arrays[key].append(data[key])
        summary = _summarize(SimpleNamespace(lambda_U=0.5, lambda_D=0.5, M=3,
                            v_zero_threshold=1e-3, v_zero_max_frac=0.9), arrays["U"], arrays["V"], arrays["w"])
        summary.update(disagreement_mode="normalized", zscore_scope="example", cache_expert_losses=False)
        write_json(str(weights / "summary.json"), summary)

    stage("weights", council, lambda: validate_npz_coverage(prefixed, weights, ["U", "V", "w"]))
    stage("merge", lambda: one("proposal.merge", ["--phase1_dir", phase1, "--M", "3", "--strategy", "ties",
          "--out_rank", args.lora_r, "--density", "0.2", "--out_dir", merged, "--device", "cuda", "--seed", args.seed], "merge"),
          lambda: require(merged / "merged/adapter_model.safetensors"))
    stage("distill", lambda: training("distill", student, ["--weights_dir", weights,
          "--teacher_adapter", merged / "merged", "--lambda_kl", "1", "--lr", args.lr_distill]),
          lambda: require(student / "student/adapter_model.safetensors"))
    if args.eval:
        # Qwen attention head counts need not divide three: use one vLLM replica,
        # not tensor parallel=3. Evaluation is excluded from training-stage times.
        from eval.run_eval import _round_up_lora_rank
        one("eval.run_eval", ["--model_name", model, "--method", run_id, "--adapter", student / "student",
            "--k", args.eval_n, "--seed", args.seed, "--max_lora_rank", _round_up_lora_rank(args.lora_r),
            "--max_model_len", "32768", "--results_dir", run / "eval"], "eval")
    print(f"[cored] finished requested stages; timings={run / 'timings.jsonl'}", flush=True)
    if (student / "complete.json").exists():
        print(f"[cored] student={student / 'student'}", flush=True)


if __name__ == "__main__":
    main()
