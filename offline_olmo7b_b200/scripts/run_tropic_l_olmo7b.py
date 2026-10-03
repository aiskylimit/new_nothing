"""TROPIC-L (TROPIC_Proposal_v8, leverage-allocated trust regions) on
allenai/Olmo-3-7B-Think, as an offline-deployment baseline (TAG=
"tropic_l_olmo", --output-dir default "results_tropic_l_olmo7b"). OLMo
sibling of scripts/run_tropic_l_8b.py on the online cluster: SAME algorithm
and CLI surface (Eq. 12's leverage allocation replacing v6/v7's single-
coordinate tilt, self-value process credit in log-space per-rollout norm by
default), only the model, offline-deployment mechanisms, and sampling recipe
differ - see below.

This offline package had no TROPIC-L script at all before this file, and no
process-credit/leverage machinery in its tropic/ copy - `tropic/process_credit.py`,
`tropic/leverage.py`, `tropic/loss_l.py`, `tropic/loss_pc.py` (needed by
loss_l.py for `tilt_sampled_token`), and `tropic/ema_teacher.py` (needed for
--anchor ema) were copied verbatim from the online cluster's tropic/ package
(model-agnostic - process_credit.py's AnchorTeacher wraps tropic.model.
ContextualPolicy generically, no Qwen-specific assumptions), and
`tropic/primitives.py` had v8's `bisect_step_size_vec`/`project_to_kl_ball_vec`
appended (the rest of that file was already identical to the online copy).
Only the credit SOURCE actually built anywhere in this codebase is self-value
(SV) - Outcome/SV+O/external-PRM sources described in the v6 proposal are not
implemented (v6/v7's own comment: "not built here yet, by explicit request").

Same experimental setting as this package's other OLMo scripts (LoRA
r=64/alpha=128, SAME target_modules - verified against transformers' real
modeling_olmo3.py source on 2026-09-15: Olmo3Attention/Olmo3MLP use
q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj, byte-identical names
to Qwen3), lr=5e-6/max_grad_norm=0.1/gradient_checkpointing, checkpoint
cadence 5/10/15/20/25/40/50/60/75/80/100 (same as run_tropic_l_4b.py/
run_tropic_l_8b.py - includes the early 5/10/15, a TROPIC-L-specific
convention kept regardless of what this package's OTHER (TROPIC-G) OLMo
script uses), eval protocol (30 problems x k=12 x {aime25,aime26,hmmt25} by
default), effective_batch_size=4 (already run_tropic_l_8b.py's own default
for an 8B-scale model - unchanged here), --skip-train/--skip-eval/
--model-path/--vllm-gpu-memory-utilization/--training-steps offline
mechanisms, TROPIC_TRAIN_DATA_PATH/TROPIC_EVAL_DATA_DIR env-var overrides.
UNLIKE run_tropic_l_8b.py (which uses OPSD's real train recipe, temperature=
1.1/top_p=0.95/top_k=20): this package's own unified, unrestricted sampling
recipe (temperature=1.0/top_p=1.0/top_k=-1) is used for train AND eval, same
as this package's other 3 scripts, for a fair cross-method comparison on this
specific offline deployment.

CRITICAL DIFFERENCE FROM QWEN3 - NO NATIVE THINKING TOGGLE, TRICK REQUIRED:
see run_tropic_g_topk64_olmo7b.py's docstring in this same folder for the
full explanation (verified live against the real allenai/Olmo-3-7B-Think
tokenizer on 2026-09-15) - CONFIG["empty_think_suffix"] = "\\n\\n</think>\\n\\n"
closes the template's auto-opened <think> tag empty for the STUDENT's
non-thinking rollout only (never for the thinking teacher/eval), threaded
through tropic.data's load_opsd_math_examples/build_teacher_context_only_prefix
exactly as that script already does. UNVERIFIED END-TO-END on this exact
model - inspect real student rollouts early before trusting a full run.

New CLI surface (same as run_tropic_l_8b.py - Section 4.5/6.8's ablation
factors): --alloc {uniform,entropy,leverage} (default leverage), --rho/--eta
(Eq. 12, defaults 1.0/0.5), --eps-max-mult (default 5.0), --kappa (default
0.0, v6/v7 tilt now an ablation), --value-space {log,prob} (default log, v8's
fix), --norm {rollout,batch} (default rollout, v8's fix), --sigma-min/
--delta-min/--degenerate-threshold (rollout-norm only), --m-boundaries
(default 4), --anchor {frozen,ema,sync}, --ema-decay/--sync-every,
--omega (first-error emphasis, default 0.0 off), --mu (optional hinge,
default 0.0), --top-k (default 64, this project's real sparsification
default, not "full vocabulary").

Usage - one-time training (no eval), THEN eval per checkpoint step:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/run_tropic_l_olmo7b.py \\
        --model-path /path/to/local/Olmo-3-7B-Think --seed 0 --output-dir results_tropic_l_olmo7b --skip-eval \\
        --use-vllm-rollout --vllm-gpu-ids 1,2,3 --vllm-base-port 8100
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/run_tropic_l_olmo7b.py \\
        --model-path /path/to/local/Olmo-3-7B-Think --skip-train \\
        --checkpoint-path results_tropic_l_olmo7b/tropic_l_olmo_checkpoint_step100 \\
        --output-dir results_tropic_l_olmo7b --eval-engine vllm --vllm-gpu-ids 1,2,3 --vllm-base-port 8100

RECOMMENDATION (same as every other TROPIC-L script): dry-run first with
--training-steps 3 and inspect the per-step diagnostics (s-by-leverage-decile,
frac_capped, mean_J, frac_ahat_saturated/deadzoned) AND a few raw student
rollouts (the empty-think-suffix trick is unverified end-to-end for OLMo)
before committing to a full run.
"""
import argparse
import json
import re
import time
import gc
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="results_tropic_l_olmo7b",
                     help="Where to write results/checkpoints.")
parser.add_argument("--seed", type=int, default=0,
                     help="Seed for the training-example draw AND torch's global RNG.")
parser.add_argument("--model-path", default=None,
                     help="Local directory holding a raw download of the base model (offline "
                          "deployment - e.g. a no-internet server pre-populated via `download.txt`) "
                          "- overrides CONFIG['model_name']. Omit to use the normal online repo id.")
parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=None,
                     help="Passed through to `vllm serve --gpu-memory-utilization` for every "
                          "replica (train and eval). Omit to use vLLM's own default (0.9).")
parser.add_argument("--vllm-max-model-len", type=int, default=41000,
                     help="Passed through to `vllm serve --max-model-len` for every replica "
                          "(train and eval). Without this, vLLM auto-detects the model's own "
                          "max_position_embeddings (can be far larger than anything this project "
                          "ever sends it), which makes it reserve per-sequence KV cache for a "
                          "context length nothing reaches - artificially capping how many "
                          "sequences can run concurrently even with plenty of free VRAM. Default "
                          "41000 covers eval's max_new_tokens=38912 plus a short prompt; pass 0 "
                          "to fall back to vLLM's own auto-detection.")
parser.add_argument("--training-steps", type=int, default=100,
                     help="Overrides CONFIG['train']['training_steps'] (default 100) - lower this "
                          "for a quick dry run (e.g. 3) before committing to a full run.")
parser.add_argument("--alpha", type=float, default=0.25,
                     help="Eq. 9's classifier-free-guidance debias weight - default 0.25.")
parser.add_argument("--alloc", choices=["uniform", "entropy", "leverage"], default="leverage",
                     help="Trust-region allocation policy (Section 4.5). 'leverage' is v8's "
                          "proposed method (TROPIC-L); 'entropy' is TROPIC-L_H (no value needed "
                          "at all, eta forced to effectively infinite); 'uniform' recovers the "
                          "v5-v7 fixed radius exactly (TROPIC-G).")
parser.add_argument("--rho", type=float, default=1.0, help="Eq. 12's leverage exponent (Table 1: 1).")
parser.add_argument("--eta", type=float, default=0.5, help="Eq. 12's value floor (Table 1: 0.5).")
parser.add_argument("--eps-max-mult", type=float, default=5.0,
                     help="eps_max = this * eps_mean (Table 1: 5).")
parser.add_argument("--kappa", type=float, default=0.0,
                     help="The v6/v7 tilt strength, now an ablation - default 0.0 (off, Table 1's "
                          "new default). Nonzero re-enables Eq. 14's single-coordinate tilt "
                          "alongside allocation.")
parser.add_argument("--value-space", choices=["log", "prob"], default="log",
                     help="'log' (default, v8's fix): tropic.process_credit.compute_logvalue, "
                          "never exponentiated. 'prob': the OLD v6/v7 exponentiated self-value "
                          "(compute_self_value) - a measured ablation.")
parser.add_argument("--norm", choices=["rollout", "batch"], default="rollout",
                     help="'rollout' (default, v8's fix): normalize Ahat by THIS roll-out's own "
                          "mean/std (tropic.process_credit.progress_v8). 'batch': the OLD v7 "
                          "batch-level sigma_hat - an ablation.")
parser.add_argument("--sigma-min", type=float, default=0.1,
                     help="Absolute nat-space floor on sigma_hat (Table 1: 0.1 nat). Only used "
                          "with --norm rollout.")
parser.add_argument("--delta-min", type=float, default=0.05,
                     help="Dead-zone threshold in nats (Table 1: 0.05 nat). Only used with "
                          "--norm rollout.")
parser.add_argument("--degenerate-threshold", type=float, default=0.1,
                     help="Roll-out-level zero-credit threshold on (max delta - min delta), in "
                          "nats - NOT given an exact numeric default in Table 1; this default "
                          "matches --sigma-min as a documented, honest placeholder. Only used "
                          "with --norm rollout.")
parser.add_argument("--omega", type=float, default=0.0,
                     help="Algorithm 1's optional first-error-emphasis weight - default 0.0 (off).")
parser.add_argument("--mu", type=float, default=0.0,
                     help="Optional Eq. 19 hinge weight - default 0.0.")
parser.add_argument("--anchor", choices=["frozen", "ema", "sync"], default="frozen",
                     help="Section 4.2's teacher anchor schedule.")
parser.add_argument("--ema-decay", type=float, default=0.999, help="tau, only used with --anchor ema.")
parser.add_argument("--sync-every", type=int, default=20, help="M, only used with --anchor sync.")
parser.add_argument("--m-boundaries", type=int, default=4,
                     help="Max self-value evaluations per roll-out (Table 1: 4).")
parser.add_argument("--min-step-len", type=int, default=32, help="L_min (Table 1: 32).")
parser.add_argument("--min-steps", type=int, default=4, help="J_min - segmentation fallback floor (Table 1: 4).")
parser.add_argument("--max-steps", type=int, default=16, help="J_max (Table 1: 16).")
parser.add_argument("--chunk-size", type=int, default=96,
                     help="Token chunk size for the segmentation fallback's last rung.")
parser.add_argument("--top-k", type=int, default=64,
                     help="Loss sparsification top_k - default 64, this project's real "
                          "sparsification default, not Table 1's 'full vocabulary'.")
parser.add_argument("--eps-mean", type=float, default=0.1, help="Average trust-region radius (Table 1: 0.1).")
parser.add_argument("--beta", type=float, default=0.1, help="Skewed-KL parameter (Table 1: 0.1).")
parser.add_argument("--eval-benchmarks", nargs="+", default=["aime25", "aime26", "hmmt25"],
                     choices=["aime24", "aime25", "aime26", "hmmt25"])
parser.add_argument("--eval-engine", choices=["vllm", "hf"], default="hf",
                     help="Backend for eval generation. 'hf' uses plain transformers.generate(). "
                          "'vllm' generates via data-parallel vLLM server replicas - requires "
                          "--vllm-gpu-ids; several times faster for this workload.")
parser.add_argument("--vllm-gpu-ids", default=None,
                     help="Comma-separated GPU ids to run vLLM replicas on (e.g. '1,2,3') - MUST "
                          "be free GPUs, disjoint from this process's own CUDA_VISIBLE_DEVICES.")
parser.add_argument("--vllm-base-port", type=int, default=8100)
parser.add_argument("--vllm-executable", default="vllm")
parser.add_argument("--use-vllm-rollout", action="store_true",
                     help="Generate training rollouts via data-parallel vLLM server replicas "
                          "instead of sequential HF model.generate() calls. Requires --vllm-gpu-ids.")
parser.add_argument("--skip-train", action="store_true",
                     help="Skip training; load an already-trained checkpoint for eval instead "
                          "(requires --checkpoint-path).")
parser.add_argument("--checkpoint-path", default=None)
parser.add_argument("--skip-eval", action="store_true",
                     help="Train and checkpoint as usual, but skip eval - eval is instead run "
                          "separately by other processes pointed at the saved checkpoints.")
parser.add_argument("--shard-index", type=int, default=0)
parser.add_argument("--num-shards", type=int, default=1)
args = parser.parse_args()
if args.skip_train and not args.checkpoint_path:
    parser.error("--skip-train requires --checkpoint-path")
if args.use_vllm_rollout and not args.vllm_gpu_ids:
    parser.error("--use-vllm-rollout requires --vllm-gpu-ids")
if args.anchor == "sync" and args.alpha > 0.0:
    parser.error("--anchor sync has no raw-logits forward pass, so --alpha must be 0.0 with this "
                 "anchor - see tropic.process_credit.AnchorTeacher.score_logits.")

t_start = time.time()
RESULTS_DIR = Path(__file__).resolve().parent.parent / args.output_dir
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

CHECKPOINT_STEPS = {5, 10, 15, 20, 25, 40, 50, 60, 75, 80, 100}


def tick(label):
    print(f"[{time.time() - t_start:8.1f}s] {label}", flush=True)


tick("import torch/transformers/datasets/peft")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

torch.manual_seed(args.seed)
tick(f"imports done, torch.manual_seed({args.seed})")

LORA_CONFIG = dict(
    r=64, lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

CONFIG = dict(
    model_name=args.model_path or "allenai/Olmo-3-7B-Think",
    only_correct=True, teacher_thinking=True, student_thinking=False,
    max_length=20000,
    # OLMo-3-7B-Think has no native enable_thinking toggle - this closes its
    # auto-opened <think> tag empty when enable_thinking=False (student only;
    # eval and the thinking teacher never see this) - see
    # run_tropic_g_topk64_olmo7b.py's own docstring (UNVERIFIED end-to-end).
    empty_think_suffix="\n\n</think>\n\n",
    lora=LORA_CONFIG,
    anchor=args.anchor, ema_decay=args.ema_decay, sync_every=args.sync_every,
    answer_forcing_suffix="\n\nFinal answer: \\boxed{",  # Eq. 10's u
    train=dict(training_steps=args.training_steps, effective_batch_size=4, lr=5e-6, max_grad_norm=0.1,
               max_new_tokens=1024, temperature=1.0, top_p=1.0, top_k=-1, seed=args.seed,
               gradient_checkpointing=True),
    tropic_l=dict(epsilon_mean=args.eps_mean, beta=args.beta, top_k=args.top_k,
                  default_mass=1e-5, alpha=args.alpha, mu=args.mu),
    alloc=dict(policy=args.alloc, rho=args.rho, eta=args.eta,
               eps_max=args.eps_mean * args.eps_max_mult, kappa=args.kappa, omega=args.omega),
    process_credit=dict(min_len=args.min_step_len, max_steps=args.max_steps, min_steps=args.min_steps,
                        chunk_size=args.chunk_size, m_boundaries=args.m_boundaries,
                        value_space=args.value_space, norm=args.norm,
                        sigma_min=args.sigma_min, delta_min=args.delta_min,
                        degenerate_threshold=args.degenerate_threshold),
    eval=dict(benchmarks=args.eval_benchmarks, num_problems=30, k=12,
              max_new_tokens=38912, temperature=1.0, top_p=1.0, top_k=-1,
              min_p=0.0, enable_thinking=True, gen_batch_size=4),
)
print(json.dumps(CONFIG, indent=2), flush=True)

tick("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
tick("tokenizer loaded")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
tick(f"device={device} dtype={dtype}")
if not torch.cuda.is_available():
    print("WARNING: no GPU detected - this config will be extremely slow on CPU.", flush=True)


def load_fresh_model():
    try:
        base = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"], dtype=dtype, attn_implementation="flash_attention_2"
        ).to(device)
    except (ImportError, ValueError) as e:
        tick(f"flash_attention_2 unavailable ({e}); falling back to default attention implementation")
        base = AutoModelForCausalLM.from_pretrained(CONFIG["model_name"], dtype=dtype).to(device)
    if CONFIG["train"].get("gradient_checkpointing", False):
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        base.config.use_cache = False
    lora_cfg = LoraConfig(task_type="CAUSAL_LM", **CONFIG["lora"])
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    return model


def load_model_from_checkpoint(checkpoint_path):
    try:
        base = AutoModelForCausalLM.from_pretrained(
            CONFIG["model_name"], dtype=dtype, attn_implementation="flash_attention_2"
        ).to(device)
    except (ImportError, ValueError) as e:
        tick(f"flash_attention_2 unavailable ({e}); falling back to default attention implementation")
        base = AutoModelForCausalLM.from_pretrained(CONFIG["model_name"], dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base, checkpoint_path)
    tick(f"loaded checkpoint from {checkpoint_path}")
    return model


def clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def infer_step_suffix(checkpoint_path):
    if not checkpoint_path:
        return ""
    m = re.search(r"_checkpoint_step(\d+)/?$", str(checkpoint_path).rstrip("/\\"))
    return f"_step{m.group(1)}" if m else ""


ALL_BENCHMARKS = {"aime25", "aime26", "hmmt25"}


def bench_suffix_for(benchmarks):
    if set(benchmarks) == ALL_BENCHMARKS:
        return ""
    return "_" + "-".join(sorted(benchmarks))


from tropic.data import load_opsd_math_examples, build_teacher_context_only_prefix
from tropic.model import ContextualPolicy
from tropic.rollout import generate_rollout
from tropic.loss_l import tropic_l_loss
from tropic.leverage import allocate_epsilon, leverage as compute_leverage, token_entropy
from tropic.primitives import debias_logits, kl_categorical, normalize_log
from tropic.process_credit import (
    AnchorTeacher, batch_sigma_hat, compute_logvalue_at_boundaries, compute_self_value_at_boundaries,
    expand_sparse_values, extract_final_answer, first_error_weights, is_degenerate_rollout,
    progress_advantage, progress_v8, segment_steps_v8, select_evaluation_boundaries,
    tilt_per_token, token_step_index,
)
from tropic.eval import load_benchmark, score_generations
from tropic.vllm_rollout_client import (
    generate_eval_batch_parallel, generate_rollout_batch_parallel,
    launch_vllm_replicas, refresh_lora_adapter, shutdown_vllm_replicas,
)


train_examples = None
if not args.skip_train:
    num_raw_examples = int(CONFIG["train"]["training_steps"] * CONFIG["train"]["effective_batch_size"] * 1.05)
    tick(f"loading + filtering OPSD dataset, num_examples={num_raw_examples}")
    raw_examples = load_opsd_math_examples(
        tokenizer, num_examples=num_raw_examples,
        only_correct=CONFIG["only_correct"], teacher_thinking=CONFIG["teacher_thinking"],
        student_thinking=CONFIG["student_thinking"], seed=CONFIG["train"]["seed"],
        max_length=CONFIG["max_length"], empty_think_suffix=CONFIG["empty_think_suffix"],
    )
    train_examples = []
    for ex in raw_examples:
        a_star = extract_final_answer(ex.reference_solution)
        if a_star is not None:
            train_examples.append((ex, a_star))
    dropped = len(raw_examples) - len(train_examples)
    tick(f"dataset ready: {len(train_examples)} examples with a \\boxed{{}} answer, "
         f"{dropped} dropped ({dropped / max(1, len(raw_examples)):.1%}, no \\boxed{{}} in reference solution)")


def build_anchor(model, policy):
    if CONFIG["anchor"] == "frozen":
        return AnchorTeacher("frozen", policy)
    if CONFIG["anchor"] == "ema":
        return AnchorTeacher("ema", policy, model=model, ema_decay=CONFIG["ema_decay"])
    return AnchorTeacher("sync", policy, model=model, sync_interval=CONFIG["sync_every"])


def leverage_decile_summary(leverage_flat: list, s_flat: list) -> list:
    """Mean s within each of the 10 deciles of THIS STEP's leverage values."""
    if not leverage_flat:
        return [None] * 10
    lev = torch.tensor(leverage_flat, dtype=torch.float64)
    s = torch.tensor(s_flat, dtype=torch.float64)
    order = torch.argsort(lev)
    n = len(order)
    deciles = []
    for d in range(10):
        lo = (d * n) // 10
        hi = max(lo + 1, ((d + 1) * n) // 10)
        idx = order[lo:hi]
        deciles.append(s[idx].mean().item() if len(idx) else None)
    return deciles


def compute_value_and_advantage(anchor, ex, a_star, generated_ids, pc, alloc_cfg):
    """One roll-out's worth of Section 4.4-4.5 pipeline: segment -> value ->
    progress -> per-token leverage inputs. Returns (Ahat [J] or None,
    step_of_token [T], fallback_level, num_steps, diag)."""
    boundaries, fallback_level = segment_steps_v8(
        tokenizer, generated_ids.cpu(), min_len=pc["min_len"], max_steps=pc["max_steps"],
        min_steps=pc["min_steps"], chunk_size=pc["chunk_size"],
    )
    num_steps = len(boundaries) - 1
    step_of_token = token_step_index(boundaries).to(generated_ids.device)
    if num_steps == 0:
        return torch.zeros(0), step_of_token, fallback_level, num_steps, {}

    answer_ids = tokenizer(a_star, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
    suffix_ids = tokenizer(CONFIG["answer_forcing_suffix"], add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
    eval_idx = select_evaluation_boundaries(num_steps, pc["m_boundaries"])
    eval_positions = [boundaries[i] for i in eval_idx]

    if pc["value_space"] == "log":
        values_eval = compute_logvalue_at_boundaries(anchor, ex.student_prompt_ids, generated_ids, eval_positions, suffix_ids, answer_ids)
    else:  # "prob" ablation: the OLD v6/v7 exponentiated self-value
        values_eval = compute_self_value_at_boundaries(anchor, ex.student_prompt_ids, generated_ids, eval_positions, suffix_ids, answer_ids)
    values_full = expand_sparse_values(eval_idx, values_eval.cpu(), num_steps)
    deltas = values_full[1:] - values_full[:-1]

    if pc["norm"] == "rollout":
        if is_degenerate_rollout(deltas, pc["degenerate_threshold"]):
            ahat = torch.zeros(num_steps)
        else:
            ahat = progress_v8(deltas, sigma_min=pc["sigma_min"], delta_min=pc["delta_min"])
    else:  # "batch" ablation: caller supplies a batch-level sigma via extra_state (see train_run)
        ahat = deltas  # raw increments; the caller finishes normalization once sigma_hat is known

    diag = {"J": num_steps, "fallback_level": fallback_level, "deltas": deltas}
    return ahat, step_of_token, fallback_level, num_steps, diag


def train_run(model, examples, cfg, tag, use_vllm_rollout=False, vllm_gpu_ids=None,
              vllm_base_port=8100, vllm_executable="vllm"):
    fixed_teacher = cfg["anchor"] == "frozen"
    policy = ContextualPolicy(model, tokenizer, fixed_teacher=fixed_teacher)
    anchor = build_anchor(model, policy)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    tick(f"[{tag}] optimizer over {n_trainable} trainable params, anchor={cfg['anchor']}, alloc={cfg['alloc']['policy']}")
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg["train"]["lr"])
    training_steps = cfg["train"]["training_steps"]
    effective_batch_size = cfg["train"]["effective_batch_size"]
    example_idx = 0
    logs = []
    run_start = time.time()
    pc = cfg["process_credit"]
    alloc_cfg = cfg["alloc"]

    vllm_procs, vllm_ports, vllm_scratch_dir = None, None, None
    if use_vllm_rollout:
        vllm_scratch_dir = RESULTS_DIR / f"{tag}_vllm_scratch_adapter"
        tick(f"[{tag}] launching {len(vllm_gpu_ids)} vLLM rollout replica(s) on GPU {vllm_gpu_ids}")
        vllm_procs, vllm_ports = launch_vllm_replicas(
            cfg["model_name"], vllm_gpu_ids, vllm_base_port, cfg["lora"]["r"],
            log_dir=str(RESULTS_DIR), vllm_executable=vllm_executable,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=(args.vllm_max_model_len or None),
        )
        model.save_pretrained(str(vllm_scratch_dir))
        refresh_lora_adapter(vllm_ports, str(vllm_scratch_dir))
        tick(f"[{tag}] vLLM replicas ready, initial adapter loaded")

    try:
        for step in range(training_steps):
            t0 = time.time()
            optimizer.zero_grad()
            step_examples = [examples[(example_idx + i) % len(examples)] for i in range(effective_batch_size)]
            example_idx += effective_batch_size
            if use_vllm_rollout:
                vllm_rollouts = generate_rollout_batch_parallel(
                    vllm_ports, [ex.student_prompt_ids for ex, _ in step_examples], tokenizer,
                    max_new_tokens=cfg["train"]["max_new_tokens"], temperature=cfg["train"]["temperature"],
                    top_p=cfg["train"]["top_p"], top_k=cfg["train"].get("top_k"),
                )

            # --- Pass 1: rollouts + segmentation + value (no_grad) ---------
            rollout_cache = []
            all_raw_deltas = []
            for micro in range(effective_batch_size):
                ex, a_star = step_examples[micro]
                if use_vllm_rollout:
                    generated_ids, gen_len = vllm_rollouts[micro]
                    generated_ids = generated_ids.to(model.device)
                else:
                    generated_ids, gen_len = generate_rollout(
                        model, tokenizer, ex.student_prompt_ids,
                        max_new_tokens=cfg["train"]["max_new_tokens"],
                        temperature=cfg["train"]["temperature"], top_p=cfg["train"]["top_p"],
                        top_k=cfg["train"].get("top_k"),
                    )
                ahat_or_deltas, step_of_token, fallback_level, num_steps, diag = compute_value_and_advantage(
                    anchor, ex, a_star, generated_ids, pc, alloc_cfg
                )
                if pc["norm"] == "batch" and num_steps > 0:
                    all_raw_deltas.append(ahat_or_deltas)
                rollout_cache.append((ex, generated_ids, gen_len, ahat_or_deltas, step_of_token, fallback_level, num_steps))

            batch_sigma = None
            if pc["norm"] == "batch" and all_raw_deltas:
                batch_sigma = batch_sigma_hat(torch.cat(all_raw_deltas), pc["sigma_min"])

            # --- Pass 2: real forward/backward with Ahat/leverage/epsilon_t
            micro_losses, micro_s, micro_g, micro_hinge = [], [], [], []
            all_leverage, all_s_for_decile, all_epsilon_t, all_entropy = [], [], [], []
            j_values, fallback_levels, frac_capped_list = [], [], []
            ahat_saturated, ahat_deadzoned, ahat_total = 0, 0, 0
            last_ex = last_generated_ids = last_log_old = None
            for ex, generated_ids, gen_len, ahat_or_deltas, step_of_token, fallback_level, num_steps in rollout_cache:
                j_values.append(num_steps)
                fallback_levels.append(fallback_level)

                if num_steps > 0:
                    if pc["norm"] == "batch":
                        ahat = progress_advantage(
                            torch.cat([torch.zeros(1), ahat_or_deltas.cumsum(0)]),
                            batch_sigma if batch_sigma is not None else pc["sigma_min"],
                        )
                    else:
                        ahat = ahat_or_deltas
                    ahat_total += ahat.numel()
                    ahat_saturated += int((ahat.abs() >= 1.0 - 1e-6).sum().item())
                    ahat_deadzoned += int((ahat == 0.0).sum().item())
                    abs_a_per_token = ahat.to(generated_ids.device)[step_of_token].abs()
                    tilt = tilt_per_token(ahat.to(generated_ids.device), step_of_token, alloc_cfg["kappa"]) if alloc_cfg["kappa"] != 0.0 else None
                    token_weights = (
                        first_error_weights(ahat, alloc_cfg["omega"], step_of_token.cpu()).to(generated_ids.device)
                        if alloc_cfg["omega"] > 0 else None
                    )
                else:
                    abs_a_per_token = torch.zeros(generated_ids.shape[0], device=generated_ids.device)
                    tilt, token_weights = None, None

                log_old = policy.forward_checkpoint(ex.student_prompt_ids, generated_ids)
                log_student = policy.forward_student(ex.student_prompt_ids, generated_ids)
                log_teacher = anchor.score_logits(ex.teacher_prefix_ids, generated_ids)
                if cfg["tropic_l"]["alpha"] > 0.0:
                    context_only_prefix_ids = build_teacher_context_only_prefix(
                        tokenizer, ex.reference_solution,
                        enable_thinking=CONFIG["teacher_thinking"], max_length=CONFIG["max_length"],
                        empty_think_suffix=CONFIG["empty_think_suffix"],
                    )
                    log_teacher_context_only = anchor.score_logits(context_only_prefix_ids, generated_ids)
                else:
                    log_teacher_context_only = None

                H = token_entropy(log_old.detach())
                if alloc_cfg["policy"] == "uniform":
                    epsilon_t = torch.full_like(H, cfg["tropic_l"]["epsilon_mean"])
                    lev = torch.ones_like(H)
                else:
                    eta = 1e6 if alloc_cfg["policy"] == "entropy" else alloc_cfg["eta"]
                    lev = compute_leverage(H, abs_a_per_token, alloc_cfg["rho"], eta)
                    epsilon_t = allocate_epsilon(lev, cfg["tropic_l"]["epsilon_mean"], alloc_cfg["eps_max"])
                frac_capped_list.append((epsilon_t >= alloc_cfg["eps_max"] - 1e-9).float().mean().item())

                loss, s, hinge = tropic_l_loss(
                    log_student, log_teacher, log_old, generated_ids,
                    epsilon_t=epsilon_t,
                    beta=cfg["tropic_l"]["beta"], top_k=cfg["tropic_l"]["top_k"],
                    default_mass=cfg["tropic_l"]["default_mass"], alpha=cfg["tropic_l"]["alpha"],
                    mu=cfg["tropic_l"]["mu"], tilt=tilt, token_weights=token_weights,
                    teacher_context_only_logits=log_teacher_context_only,
                )
                micro_s.append(s.mean().item())
                micro_hinge.append(hinge.item())
                all_leverage.extend(lev.detach().cpu().tolist())
                all_s_for_decile.extend(s.detach().cpu().tolist())
                all_epsilon_t.extend(epsilon_t.detach().cpu().tolist())
                all_entropy.extend(H.detach().cpu().tolist())
                if cfg["tropic_l"]["alpha"] > 0.0:
                    pi_bar_teacher_log = normalize_log(
                        debias_logits(log_teacher, log_teacher_context_only, cfg["tropic_l"]["alpha"])
                    )
                    micro_g.append(kl_categorical(log_old, pi_bar_teacher_log).mean().item())

                (loss / effective_batch_size).backward()
                micro_losses.append(loss.item())
                last_ex, last_generated_ids, last_log_old = ex, generated_ids, log_old

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=cfg["train"]["max_grad_norm"])
            optimizer.step()
            anchor.update(step)  # AFTER optimizer.step() (Algorithm 1 line 16)

            if use_vllm_rollout:
                model.save_pretrained(str(vllm_scratch_dir))
                refresh_lora_adapter(vllm_ports, str(vllm_scratch_dir))

            with torch.no_grad():
                log_student_final = policy.forward_student(last_ex.student_prompt_ids, last_generated_ids)
                drift = kl_categorical(log_student_final, last_log_old).mean().item()

            mean_loss = sum(micro_losses) / len(micro_losses)
            eps_quantile_grid = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
            record = {
                "tag": tag, "step": step, "loss": mean_loss, "elapsed_sec": time.time() - t0,
                "kl_student_old_drift": drift,
                "mean_s": sum(micro_s) / len(micro_s) if micro_s else None,
                "s_by_leverage_decile": leverage_decile_summary(all_leverage, all_s_for_decile),
                "eps_t_quantiles": (
                    torch.quantile(torch.tensor(all_epsilon_t, dtype=torch.float64), eps_quantile_grid).tolist()
                    if all_epsilon_t else None
                ),
                "frac_capped": sum(frac_capped_list) / len(frac_capped_list) if frac_capped_list else None,
                "mean_entropy": (sum(all_entropy) / len(all_entropy)) if all_entropy else None,
                "mean_J": sum(j_values) / len(j_values) if j_values else None,
                "fallback_level_counts": {lvl: fallback_levels.count(lvl) for lvl in set(fallback_levels)},
                "frac_ahat_saturated": (ahat_saturated / ahat_total) if ahat_total else None,
                "frac_ahat_deadzoned": (ahat_deadzoned / ahat_total) if ahat_total else None,
            }
            if micro_g:
                record["G_k"] = sum(micro_g) / len(micro_g)
            if micro_hinge:
                record["mean_hinge"] = sum(micro_hinge) / len(micro_hinge)
            logs.append(record)

            steps_done = step + 1
            avg_sec = (time.time() - run_start) / steps_done
            eta_min = avg_sec * (training_steps - steps_done) / 60.0
            record["avg_sec_per_step"] = round(avg_sec, 2)
            record["eta_min"] = round(eta_min, 1)
            tick(f"[{tag}] {json.dumps(record)}")
            if mean_loss != mean_loss:
                raise RuntimeError(f"[{tag}] loss became NaN at step={step}")

            if (step + 1) in CHECKPOINT_STEPS:
                ckpt_dir = RESULTS_DIR / f"{tag}_checkpoint_step{step + 1}"
                model.save_pretrained(str(ckpt_dir))
                tick(f"[{tag}] checkpoint saved: {ckpt_dir}")

            del last_generated_ids, last_log_old, rollout_cache
            if step % 10 == 0:
                clear_cuda_cache()
    finally:
        if use_vllm_rollout:
            shutdown_vllm_replicas(vllm_procs)
            tick(f"[{tag}] vLLM rollout replicas shut down")

    optimizer.zero_grad(set_to_none=True)
    del optimizer, policy
    clear_cuda_cache()
    (RESULTS_DIR / f"{tag}_train_logs.json").write_text(json.dumps(logs, indent=2))
    tick(f"[{tag}] training done, logs saved to {RESULTS_DIR}/{tag}_train_logs.json")
    return logs


@torch.no_grad()
def generate_k_samples(model, problem_text, k, cfg_eval):
    model.eval()
    messages = [{"role": "user", "content": problem_text + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."}]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", enable_thinking=cfg_eval["enable_thinking"])
    input_ids = ids["input_ids"] if hasattr(ids, "__getitem__") and not torch.is_tensor(ids) else ids
    input_ids = input_ids.to(model.device)
    top_k_cfg = cfg_eval.get("top_k", 50)
    top_k_hf = 0 if top_k_cfg is None or top_k_cfg < 0 else top_k_cfg
    gen_batch_size = cfg_eval.get("gen_batch_size", k)

    completions = []
    remaining = k
    while remaining > 0:
        this_batch = min(gen_batch_size, remaining)
        batch = input_ids.repeat(this_batch, 1)
        attention_mask = torch.ones_like(batch)
        out = model.generate(
            input_ids=batch, attention_mask=attention_mask,
            max_new_tokens=cfg_eval["max_new_tokens"], do_sample=True,
            temperature=cfg_eval["temperature"], top_p=cfg_eval["top_p"],
            top_k=top_k_hf, min_p=cfg_eval.get("min_p", 0.0),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        completions.extend(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True))
        del batch, attention_mask, out
        clear_cuda_cache()
        remaining -= this_batch
    return completions


def evaluate_model(model, cfg, tag, shard_index=0, num_shards=1):
    suffix = f"_shard{shard_index}of{num_shards}" if num_shards > 1 else ""
    results = {}
    all_generations = {}
    for bench_name in cfg["eval"]["benchmarks"]:
        problems = load_benchmark(bench_name, num_problems=cfg["eval"]["num_problems"])
        if num_shards > 1:
            problems = problems[shard_index::num_shards]
        generations, golds = [], []
        for i, p in enumerate(problems):
            tick(f"[{tag}{suffix}] {bench_name} problem {i+1}/{len(problems)} (id={p.problem_id})")
            gens = generate_k_samples(model, p.problem, cfg["eval"]["k"], cfg["eval"])
            generations.append(gens)
            golds.append(p.answer)
        all_generations[bench_name] = {"problem_ids": [p.problem_id for p in problems], "golds": golds, "generations": generations}
        if num_shards == 1:
            res = score_generations(bench_name, generations, golds)
            results[bench_name] = res
            tick(f"[{tag}] {bench_name} avg@{res.k}={res.avg_at_k:.3f} pass@{res.k}={res.pass_at_k:.3f}")
        else:
            tick(f"[{tag}{suffix}] {bench_name} done ({len(problems)} problems in this shard)")

    (RESULTS_DIR / f"{tag}{suffix}_generations.json").write_text(json.dumps(all_generations, indent=2))
    if num_shards == 1:
        serializable = {k: {"avg_at_k": v.avg_at_k, "pass_at_k": v.pass_at_k, "k": v.k} for k, v in results.items()}
        (RESULTS_DIR / f"{tag}_eval_results.json").write_text(json.dumps(serializable, indent=2))
    tick(f"[{tag}{suffix}] eval done - raw generations saved under {RESULTS_DIR}/")
    return results


def evaluate_model_vllm(checkpoint_dir, cfg, tag, vllm_gpu_ids, vllm_base_port=8100, vllm_executable="vllm"):
    tick(f"[{tag}] launching {len(vllm_gpu_ids)} vLLM eval replica(s) on GPU {vllm_gpu_ids}")
    procs, ports = launch_vllm_replicas(
        cfg["model_name"], vllm_gpu_ids, vllm_base_port, cfg["lora"]["r"],
        log_dir=str(RESULTS_DIR), vllm_executable=vllm_executable,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=(args.vllm_max_model_len or None),
    )
    try:
        refresh_lora_adapter(ports, str(checkpoint_dir))
        tick(f"[{tag}] vLLM eval replicas ready, checkpoint adapter loaded from {checkpoint_dir}")

        results = {}
        all_generations = {}
        top_k_cfg = cfg["eval"].get("top_k", -1)
        for bench_name in cfg["eval"]["benchmarks"]:
            problems = load_benchmark(bench_name, num_problems=cfg["eval"]["num_problems"])
            prompts = []
            for p in problems:
                messages = [{"role": "user", "content": p.problem + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=cfg["eval"]["enable_thinking"])
                prompts.append(text)

            tick(f"[{tag}] {bench_name}: generating {len(prompts)} problems x {cfg['eval']['k']} samples via vLLM")
            generations = generate_eval_batch_parallel(
                ports, prompts, max_tokens=cfg["eval"]["max_new_tokens"], temperature=cfg["eval"]["temperature"],
                top_p=cfg["eval"]["top_p"], n=cfg["eval"]["k"],
                top_k=top_k_cfg if top_k_cfg and top_k_cfg > 0 else None,
                min_p=cfg["eval"].get("min_p", 0.0) or None,
            )
            golds = [p.answer for p in problems]

            res = score_generations(bench_name, generations, golds)
            results[bench_name] = res
            all_generations[bench_name] = {"problem_ids": [p.problem_id for p in problems], "golds": golds, "generations": generations}
            tick(f"[{tag}] {bench_name} avg@{res.k}={res.avg_at_k:.3f} pass@{res.k}={res.pass_at_k:.3f}")

            (RESULTS_DIR / f"{tag}_generations.json").write_text(json.dumps(all_generations, indent=2))
            serializable = {k: {"avg_at_k": v.avg_at_k, "pass_at_k": v.pass_at_k, "k": v.k} for k, v in results.items()}
            (RESULTS_DIR / f"{tag}_eval_results.json").write_text(json.dumps(serializable, indent=2))

        tick(f"[{tag}] eval done (vLLM) - scores + raw generations saved under {RESULTS_DIR}/")
        return results
    finally:
        shutdown_vllm_replicas(procs)
        tick(f"[{tag}] vLLM eval replicas shut down")


def run_eval(model, tag, checkpoint_dir_for_vllm):
    if args.eval_engine == "hf":
        result = evaluate_model(model, CONFIG, tag, shard_index=args.shard_index, num_shards=args.num_shards)
        del model
        clear_cuda_cache()
        tick(f"model_{tag} freed")
    else:
        del model
        clear_cuda_cache()
        tick(f"model_{tag} freed - GPU clear for vLLM")
        vllm_gpu_ids = [int(x) for x in args.vllm_gpu_ids.split(",")] if args.vllm_gpu_ids else None
        if not vllm_gpu_ids:
            parser.error("--eval-engine vllm requires --vllm-gpu-ids")
        result = evaluate_model_vllm(checkpoint_dir_for_vllm, CONFIG, tag, vllm_gpu_ids, args.vllm_base_port, args.vllm_executable)
    return result


def train_then_eval(tag, method_label):
    eval_tag = tag + infer_step_suffix(args.checkpoint_path if args.skip_train else None) + bench_suffix_for(args.eval_benchmarks)
    if args.skip_train:
        tick(f"=== {tag}: --skip-train, loading checkpoint {args.checkpoint_path} ===")
        model = load_model_from_checkpoint(args.checkpoint_path)
    else:
        model = load_fresh_model()
        model.train()
        vllm_gpu_ids = [int(x) for x in args.vllm_gpu_ids.split(",")] if args.vllm_gpu_ids else None
        train_run(model, train_examples, CONFIG, tag, use_vllm_rollout=args.use_vllm_rollout,
                  vllm_gpu_ids=vllm_gpu_ids, vllm_base_port=args.vllm_base_port, vllm_executable=args.vllm_executable)
        if args.skip_eval:
            tick(f"=== {tag}: --skip-eval, training done, NOT evaluating in this process ===")
            del model
            clear_cuda_cache()
            return
    checkpoint_dir_for_vllm = (
        Path(args.checkpoint_path) if args.skip_train
        else RESULTS_DIR / f"{tag}_checkpoint_step{CONFIG['train']['training_steps']}"
    )
    eval_results[method_label] = run_eval(model, eval_tag, checkpoint_dir_for_vllm)


eval_results = {}

tick(f"=== TROPIC-L (alloc={args.alloc}, rho={args.rho}, eta={args.eta}, kappa={args.kappa}, "
     f"anchor={args.anchor}): load (with LoRA), train, eval, free ===")
train_then_eval("tropic_l_olmo", "TROPIC-L")

if args.skip_eval:
    tick(f"ALL DONE (training only, --skip-eval) - checkpoints saved under {RESULTS_DIR}/.")
    sys.exit(0)

if args.num_shards > 1:
    tick(f"ALL DONE (eval shard {args.shard_index}/{args.num_shards}) - "
         f"run scripts/combine_shards.py once all shards finish to score the full problem set.")
    sys.exit(0)

tick(f"ALL DONE - results saved under {RESULTS_DIR}/")
