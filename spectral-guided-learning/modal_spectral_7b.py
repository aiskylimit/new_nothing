"""Spectral-guided Learning end-to-end on Modal: Qwen2.5-7B-Instruct, Unsloth + LoRA.

Pipeline: data_prep -> gradient_capture -> build_masks -> train(spectral) -> evaluate.
Everything lands on the `spectral-7b` volume: data/, checkpoints/, logs/, results/.

  modal run modal_spectral_7b.py --mode smoke              # 12 samples, validates the chain + VRAM
  modal run --detach modal_spectral_7b.py --mode full      # the real run
  modal run modal_spectral_7b.py --mode eval-only          # re-evaluate an existing checkpoint

Training setup follows P-ALIGN §4 (3 epochs, lr 5e-5, LoRA) on s1K-1.1; eval follows
P-ALIGN/scripts/Inference.sh (n=3, temp 0.6, top-p 0.9, repetition_penalty 1.05,
enable_thinking=False) at a 32k context instead of their 4k cap.
"""

import modal

MODEL = "Qwen/Qwen2.5-7B-Instruct"
TRACK = "qwen25-7b"
DATA_DIR = f"/vol/data/{TRACK}"
CHECKPOINTS_DIR = "/vol/checkpoints"
CKPT_DIR = f"{CHECKPOINTS_DIR}/spectral-{TRACK}"
LOG_DIR = "/vol/logs"
RESULTS_DIR = "/vol/results"

# IWC control arms (docs/iwc-baselines.md): all reuse the spectral gate + capture above, only
# loss_weights differ. iwc-stable is the method; lambda0/shuffled/reverse isolate whether the
# entropy allocation signal is real (vs. any reweighting, vs. the wrong direction).
IWC_VARIANTS = "iwc-stable,iwc-stable-lambda0,iwc-stable-shuffled,iwc-stable-reverse"

# --- training hyperparameters (single GPU: bs1 x ga32 = effective batch 32) ---
EPOCHS = 3
LR = "5.0e-5"
MIN_LR = "1.0e-5"
WARMUP_RATIO = "0.1"
BATCH_SIZE = 1
GRAD_ACC = 32
MAX_SEQ_LEN = 32768
SEED = 42
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# --- spectral knobs (not published by the paper; ours) ---
ENERGY_CUTOFF = "0.95"      # SVD truncation for the consensus subspace
ENERGY_THRESHOLD_P = "0.95"  # Eq. 8 threshold p
CHUNK_SIZE = "1024"

# --- eval (P-ALIGN's sampling regime, 32k context) ---
BENCHMARKS = "math500,aime24,aime25,amc12"
EVAL = dict(temperature="0.6", top_p="0.9", repetition_penalty="1.05", n_samples="3",
            max_tokens="30720", max_model_len="32768",
            batch_size="25")  # persist finished problems every 25, so a stop keeps them

vol = modal.Volume.from_name("spectral-7b", create_if_missing=True)
hf_cache = modal.Volume.from_name("spectral-vram-probe-hf", create_if_missing=True)
VOLUMES = {"/vol": vol, "/root/.cache/huggingface": hf_cache}

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "transformers==4.57.1", "peft>=0.13", "datasets>=3.0",
                 "accelerate>=1.0", "pandas", "pyarrow", "pyyaml", "tqdm", "numpy<2.2",
                 "safetensors", "huggingface_hub[hf_transfer]", "modal")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir("src", "/root/src")
)

# Unsloth pins its own torch/torchao/xformers -- do not pin torch here (see modal_train.py).
unsloth_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("unsloth==2025.11.3", "transformers==4.57.1", "trl==0.23.0", "peft>=0.13",
                 "datasets>=3.0", "accelerate>=1.0", "bitsandbytes", "pandas", "pyarrow",
                 "pyyaml", "tqdm", "numpy<2.2", "safetensors",
                 "huggingface_hub[hf_transfer]", "modal")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir("src", "/root/src")
)

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.0", "transformers==4.57.1", "datasets>=3.0", "pandas", "pyarrow",
                 "pyyaml", "tqdm", "math-verify==0.8.0", "huggingface_hub[hf_transfer]", "modal")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir("src", "/root/src")
)

app = modal.App("spectral-7b")


def _run(cmd: list[str], log_name: str, commit_on: str | None = None) -> None:
    """Run a pipeline step, tee'ing stdout to a durable log on the volume.

    commit_on: commit the volume whenever a stdout line contains this marker, so long steps
    persist partial output instead of losing everything if the container is killed.
    """
    import os
    import subprocess
    import sys

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = f"{LOG_DIR}/{log_name}"
    print(f"\n{'='*70}\n+ {' '.join(cmd)}\n  log -> {log_path}\n{'='*70}", flush=True)
    with open(log_path, "a", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
            if commit_on and commit_on in line:
                vol.commit()
        code = process.wait()
    vol.commit()
    if code != 0:
        raise SystemExit(f"step failed (exit {code}): {' '.join(cmd)} -- see {log_path}")


@app.function(image=base_image, volumes=VOLUMES, timeout=60 * 60, cpu=8.0)
def data_prep(n_samples: int | None = None):
    """s1K-1.1 -> segmented records (prompt + response + per-step token spans)."""
    import os

    from huggingface_hub import snapshot_download

    seg_path = f"{DATA_DIR}/train-segmented.jsonl"
    # n_samples is only set by smoke mode, which intentionally re-derives a small file each
    # time; a full/iwc run (n_samples=None) skips this deterministic, already-done step so a
    # second pipeline (e.g. the IWC arms) doesn't redo it against the shared capture below.
    if n_samples is None and os.path.exists(seg_path):
        print(f"[skip] data_prep: {seg_path} already present", flush=True)
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    local_model = snapshot_download(MODEL, allow_patterns=["*.json", "tokenizer*", "vocab*", "merges*"])
    cmd = ["python", "-u", "/root/src/data_prep.py",
           "--dataset-name", "simplescaling/s1K-1.1",
           "--tokenizer", local_model,
           "--output-path", f"{DATA_DIR}/train-segmented.jsonl",
           "--max-tokens", str(MAX_SEQ_LEN),
           "--chat-template", "--no-enable-thinking"]
    if n_samples:
        cmd += ["--n-samples", str(n_samples)]
    _run(cmd, "01-data-prep.log")


@app.function(image=base_image, gpu="A100-40GB", volumes=VOLUMES, timeout=60 * 60 * 8)
def capture():
    """Analytic loss-gradient-vs-hidden-state per step, SVD -> spectral strengths + entropies.

    Tried loading via Unsloth (gradient_capture_unsloth.py) for the forward-pass speedup --
    reproduced a RuntimeError (mat1/mat2 dtype mismatch, Float vs BFloat16) twice inside
    Unsloth's own compiled Qwen2Attention_forward when called via a raw model.model(...)
    pass outside their generate()/train() entrypoints. Not our code's bug and not worth
    chasing into their fused-kernel internals; reverted to plain AutoModelForCausalLM here.
    """
    import os

    strengths_path = f"{DATA_DIR}/spectral-strengths.parquet"
    if os.path.exists(strengths_path):
        print(f"[skip] capture: {strengths_path} already present", flush=True)
        return
    _run(["python", "-u", "/root/src/gradient_capture.py",
          "--model-name", MODEL,
          "--data-path", f"{DATA_DIR}/train-segmented.jsonl",
          "--output-dir", f"{DATA_DIR}/spectral",
          "--strengths-path", f"{DATA_DIR}/spectral-strengths.parquet",
          "--energy-cutoff", ENERGY_CUTOFF,
          "--chunk-size", CHUNK_SIZE,
          "--verify"], "02-capture.log")


@app.function(image=base_image, volumes=VOLUMES, timeout=60 * 60, cpu=8.0)
def masks():
    """Eq. 8 selection -> train-spectral.jsonl (and the all-ones train-vanilla.jsonl)."""
    import os

    spectral_path = f"{DATA_DIR}/train-spectral.jsonl"
    if os.path.exists(spectral_path):
        print(f"[skip] masks: {spectral_path} already present", flush=True)
        return
    _run(["python", "-u", "/root/src/build_masks.py",
          "--data-path", f"{DATA_DIR}/train-segmented.jsonl",
          "--strengths", f"{DATA_DIR}/spectral-strengths.parquet",
          "--energy-threshold-p", ENERGY_THRESHOLD_P,
          "--vanilla"], "03-masks.log")


@app.function(image=base_image, volumes=VOLUMES, timeout=60 * 30, cpu=4.0)
def build_iwc(variants: str = IWC_VARIANTS):
    """CPU-only: derive the weighted IWC arms from the shared spectral capture above.

    Reuses the exact Eq. 8 selection already computed by `masks()`; only `loss_weights`
    differ per variant (see docs/iwc-baselines.md). Cheap and idempotent -- reruns are just
    a fresh CPU pass over strengths/entropies already on the volume.
    """
    _run(["python", "-u", "/root/src/build_iwc_datasets.py",
          "--data-path", f"{DATA_DIR}/train-segmented.jsonl",
          "--strengths", f"{DATA_DIR}/spectral-strengths.parquet",
          "--energy-threshold-p", ENERGY_THRESHOLD_P,
          "--variants", variants,
          "--seed", str(SEED)], "03b-build-iwc.log")


@app.function(image=unsloth_image, gpu="A100-40GB", volumes=VOLUMES, timeout=60 * 60 * 12)
def train(smoke: bool = False):
    """Unsloth + LoRA masked SFT on the spectral mask, fp32 mixed precision."""
    import os

    os.environ["MODAL_COMMIT_VOLUME"] = "spectral-7b"  # commit the volume each epoch
    cmd = ["python", "-u", "/root/src/train_sft_unsloth.py",
           "--model-name", MODEL,
           "--data-path", f"{DATA_DIR}/train-spectral.jsonl",
           "--output-dir", CKPT_DIR,
           "--epochs", str(EPOCHS),
           "--learning-rate", LR, "--min-learning-rate", MIN_LR,
           "--warmup-ratio", WARMUP_RATIO,
           "--per-device-batch-size", str(BATCH_SIZE),
           "--gradient-accumulation-steps", str(GRAD_ACC),
           "--max-seq-len", str(MAX_SEQ_LEN),
           "--seed", str(SEED),
           "--logging-steps", "1",
           "--save-strategy", "epoch",
           "--optim", "adamw_torch",          # LoRA optimizer state is tiny; no need for 8-bit
           "--use-lora",
           "--lora-r", str(LORA_R), "--lora-alpha", str(LORA_ALPHA),
           "--lora-dropout", str(LORA_DROPOUT), "--lora-target-modules", LORA_TARGETS,
           "--float32-mixed-precision",
           "--metrics-log", f"{LOG_DIR}/train-metrics.json"]
    if smoke:
        cmd.append("--smoke")
    _run(cmd, "04-train.log")

    import torch

    print(f"\npeak VRAM: {torch.cuda.max_memory_reserved()/1024**3:.1f} GiB "
          f"of {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GiB")


@app.function(image=base_image, gpu="A100-40GB", volumes=VOLUMES, timeout=60 * 60 * 12)
def train_iwc(variant: str, smoke: bool = False):
    """LoRA masked SFT on one IWC arm, via train_sft.py -- the only path with loss_weights.

    train_sft_unsloth.py (used by `train()` above) has no weighted-loss support, so IWC arms
    go through the plain HF Trainer + masked_cross_entropy path instead. Every hyperparameter
    below is copy-identical to `train()`'s spectral run (same constants) -- only the dataset
    (loss_weights) and image differ, so a result is attributable to allocation, not drift.
    No DeepSpeed here: single GPU, and modal_vram_probe.py already validated this exact
    LoRA r16 + gradient-checkpointing shape fits without ZeRO2 offload.
    """
    import os

    os.environ["MODAL_COMMIT_VOLUME"] = "spectral-7b"
    ckpt_dir = f"{CHECKPOINTS_DIR}/{variant}-{TRACK}"
    cmd = ["python", "-u", "/root/src/train_sft.py",
           "--model-name", MODEL,
           "--data-path", f"{DATA_DIR}/train-{variant}.jsonl",
           "--output-dir", ckpt_dir,
           "--epochs", str(EPOCHS),
           "--learning-rate", LR, "--min-learning-rate", MIN_LR,
           "--warmup-ratio", WARMUP_RATIO,
           "--per-device-batch-size", str(BATCH_SIZE),
           "--gradient-accumulation-steps", str(GRAD_ACC),
           "--attn-implementation", "sdpa",
           "--max-seq-len", str(MAX_SEQ_LEN),
           "--seed", str(SEED),
           "--logging-steps", "1",
           "--save-strategy", "epoch",
           "--save-total-limit", "2",
           "--resume",
           "--use-lora",
           "--lora-r", str(LORA_R), "--lora-alpha", str(LORA_ALPHA),
           "--lora-dropout", str(LORA_DROPOUT), "--lora-target-modules", LORA_TARGETS,
           "--no-lora-merge",  # adapter-only checkpoint, loaded by evaluate() via --lora-adapter
           "--metrics-log", f"{LOG_DIR}/train-metrics-{variant}.json"]
    if smoke:
        cmd.append("--smoke")
    _run(cmd, f"04-train-{variant}.log")

    import torch

    print(f"\npeak VRAM: {torch.cuda.max_memory_reserved()/1024**3:.1f} GiB "
          f"of {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GiB")


@app.function(image=vllm_image, gpu="A100-80GB", volumes=VOLUMES, timeout=60 * 60 * 20)
def evaluate(benchmarks: str = BENCHMARKS, tag: str = f"spectral-{TRACK}", shard: str | None = None,
             ckpt_dir: str = CKPT_DIR):
    """vLLM eval in P-ALIGN's sampling regime at a 32k context; Pass@1 + Pass@3.

    One benchmark per subprocess so the volume is committed between them: a 1500-generation
    math500 pass is long enough that losing everything to one failure is not acceptable.
    With --shard i/n this container evaluates only its slice; merge_shards joins them.
    """
    import shutil
    from pathlib import Path

    suffix = f"-shard{shard.replace('/', 'of')}" if shard else ""
    run_dir = Path(RESULTS_DIR) / tag
    for name in benchmarks.split(","):
        cmd = ["python", "-u", "/root/src/evaluate.py",
               "--model", ckpt_dir, "--base-model", MODEL, "--lora-adapter", "--lora-r", str(LORA_R),
               "--tag", tag, "--benchmarks", name,
               "--temperature", EVAL["temperature"], "--top-p", EVAL["top_p"],
               "--repetition-penalty", EVAL["repetition_penalty"],
               "--n-samples", EVAL["n_samples"],
               "--max-tokens", EVAL["max_tokens"], "--max-model-len", EVAL["max_model_len"],
               "--gpu-memory-utilization", "0.90", "--no-enforce-eager",
               "--batch-size", EVAL["batch_size"],
               "--seed", str(SEED),
               "--chat-template", "--no-enable-thinking",
               "--grader", "palign",
               "--results-dir", RESULTS_DIR]
        if shard:
            cmd += ["--shard", shard]
        _run(cmd, f"05-eval-{name}{suffix}.log", commit_on="problems written ->")
        shutil.copy(run_dir / f"summary{suffix}.json", run_dir / f"summary-{name}{suffix}.json")
        vol.commit()


@app.function(image=vllm_image, volumes=VOLUMES, timeout=60 * 60, cpu=4.0)
def merge_shards(benchmarks: str, tag: str, n: int):
    """Concatenate each benchmark's shard raw files, then rescore the whole thing.

    Scoring the joined raw file rather than averaging the shards' summaries keeps Pass@k
    exact: Pass@k is a per-problem any-correct rate, which does not survive averaging two
    summaries computed over different problem subsets.
    """
    import json
    import subprocess
    from pathlib import Path

    run_dir = Path(RESULTS_DIR) / tag
    raw_dir = run_dir / "raw"
    summaries = []
    for name in benchmarks.split(","):
        joined = raw_dir / f"{name}.jsonl"
        with joined.open("w") as out:
            for index in range(n):
                part = raw_dir / f"{name}-shard{index}of{n}.jsonl"
                out.write(part.read_text())
        scored = subprocess.run(
            ["python", "-u", "/root/src/evaluate.py", "--rescore", str(joined), "--grader", "palign"],
            capture_output=True, text=True, check=True,
        ).stdout
        summary = json.loads(scored)
        summary["model"] = tag
        summaries.append(summary)

    (run_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    vol.commit()
    print()
    print("=== merged summary ===")
    for row in summaries:
        k = row["samples_per_problem"]
        print(f"  {row['benchmark']:10s} pass@1={row['pass@1']:.1%}  "
              f"pass@{k}={row.get(f'pass@{k}', 0):.1%}  len={row['length']:.0f}  "
              f"trunc={row['truncation_rate']:.1%}")


@app.function(image=base_image, volumes=VOLUMES, timeout=60 * 60 * 20, cpu=1.0)
def eval_sharded(benchmarks: str, tag: str, n: int, gpu: str, ckpt_dir: str = CKPT_DIR):
    """Fan out n GPU shards and merge them, server-side.

    The fan-out lives in a Modal function rather than the local entrypoint so a detached run
    survives the client disconnecting: otherwise the shards would finish but nothing would
    ever merge them.
    """
    handles = [
        evaluate.with_options(gpu=gpu).spawn(benchmarks, tag, f"{i}/{n}", ckpt_dir=ckpt_dir)
        for i in range(n)
    ]
    for handle in handles:
        handle.get()
    merge_shards.remote(benchmarks, tag, n)


@app.local_entrypoint()
def main(mode: str = "full", gpu: str = "A100-40GB", benchmarks: str = BENCHMARKS,
         shards: int = 1, variant: str = "iwc-stable"):
    """mode: smoke | full | iwc-smoke | iwc | prep. gpu overrides every GPU step (e.g. L40S, A100-80GB).

    prep runs only data_prep/capture/masks/build_iwc (no train, no eval) -- the cheapest way
    to get real capture output onto the volume for iwc_diagnostics.py before spending any
    GPU time on training.

    iwc/iwc-smoke train `variant` (one of IWC_VARIANTS) via train_iwc(); they reuse
    data_prep/capture/masks (each now a no-op if its output already exists on the volume --
    the shared, expensive capture step is not repeated for a second arm).

    shards > 1 splits each benchmark round-robin across that many GPUs running at once,
    then merges. Round-robin, not by difficulty: long CoT tracks problem hardness, so a
    hard/easy split would leave one GPU idle while the other finishes.
    """
    smoke = mode in ("smoke", "iwc-smoke")
    opts = {"gpu": gpu}

    if mode in ("smoke", "full", "iwc-smoke", "iwc", "prep"):
        data_prep.remote(n_samples=12 if smoke else None)
        capture.with_options(**opts).remote()
        masks.remote()

    if mode == "prep":
        # Data-only: no train, no eval. Gets train-segmented.jsonl + spectral-strengths.parquet
        # (real capture output, for iwc_diagnostics.py) plus every IWC control dataset, without
        # spending any GPU time on the much more expensive train_sft.py/vLLM eval steps.
        build_iwc.remote(variants=IWC_VARIANTS)
        print(f"\ndone (prep). shared artifacts on volume 'spectral-7b': {DATA_DIR}")
        return

    if mode in ("smoke", "full"):
        train.with_options(**opts).remote(smoke=smoke)
        eval_benchmarks = "aime24" if smoke else benchmarks
        tag = f"smoke-{TRACK}" if smoke else f"spectral-{TRACK}"
        ckpt_dir = CKPT_DIR
    elif mode in ("iwc-smoke", "iwc"):
        build_iwc.remote(variants=variant)
        train_iwc.with_options(**opts).remote(variant=variant, smoke=smoke)
        eval_benchmarks = "aime24" if smoke else benchmarks
        tag = f"smoke-{variant}-{TRACK}" if smoke else f"{variant}-{TRACK}"
        ckpt_dir = f"{CHECKPOINTS_DIR}/{variant}-{TRACK}"
    else:
        raise SystemExit(f"unknown mode: {mode!r} (expected smoke|full|iwc-smoke|iwc|prep)")

    if shards > 1:
        eval_sharded.remote(eval_benchmarks, tag, shards, gpu, ckpt_dir=ckpt_dir)
    else:
        evaluate.with_options(**opts).remote(eval_benchmarks, tag, ckpt_dir=ckpt_dir)
    print(f"\ndone ({mode}). artifacts on volume 'spectral-7b': {LOG_DIR}, {ckpt_dir}, {RESULTS_DIR}")
