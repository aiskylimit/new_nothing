"""SDPO (Self-Distillation Policy Optimization, arXiv:2601.20802) on
allenai/Olmo-3-7B-Think, copied from offline_rlsd_sdpo_b200/scripts/
run_sdpo_experiment_8b.py (same effective rollout budget as that 8B script -
group_size=2, questions_per_step=2 - since OLMo-3-7B is closer in scale to
Qwen3-8B than 4B; not separately verified for OLMo's own memory profile).
Same LoRA config (r=64, alpha=128, SAME target_modules - verified against
transformers' real modeling_olmo3.py source on 2026-09-15: Olmo3Attention/
Olmo3MLP use q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj,
byte-identical names to Qwen3, not assumed), same 100 training steps /
lr=5e-6 / max_grad_norm=0.1 / gradient_checkpointing,
same checkpoint cadence (20/25/40/50/60/75/80/100), same eval protocol (30
problems x k=12 samples x {aime24,aime25,hmmt25}, same HF-generate eval
engine, same --skip-train/--skip-eval/--shard-index/--num-shards sharding
mechanism for the "train on 1 GPU, eval sharded across 4 GPUs" workflow).

STANDALONE script (per explicit instruction: do not modify
scripts/run_full_experiment.py or any existing tropic/*.py module - every
pre-existing file in this repo is untouched by this baseline). Builds on the
same NEW, self-contained modules the RLSD baseline uses (tropic/chat_prompt.py,
tropic/verifier_data.py, tropic/group_rollout.py) plus tropic/sdpo.py (the
actual SDPO distillation loss, verified against the real reference
implementation at github.com/lasgroup/SDPO), on top of the EXISTING,
unmodified `tropic.model.ContextualPolicy` and `tropic.eval`.

FIDELITY / DESIGN NOTES (specific to this training loop; see tropic/sdpo.py's
own docstring for the loss-level fidelity notes - top-K approximation,
the generalized-JSD `alpha` mixture weight, and an important CORRECTION
about what `alpha` actually means):

  - Implements Sec 3's "Learning without Rich Environment Feedback" setting:
    a group of G rollouts is sampled per question, each graded pass/fail by
    the SAME math_verify-based verifier this project's eval already uses
    (tropic.eval.grade), and a peer's CORRECT rollout (if any exists in the
    group) is used as the self-teacher's "successful previous rollout"
    feedback for the INCORRECT rollouts in that same group (Sec 3: "SDPO
    treats successful attempts sampled in the current batch as 'feedback'
    for failed attempts on the same question").
  - Ambiguity resolved (Table 2's "if the model's original attempt was
    successful, this attempt is passed as the correct solution" describes a
    single-response Test-Time-Training setting with no group at all, Sec 5 -
    it does not unambiguously specify what happens to an ALREADY-CORRECT
    rollout within a GROUP-based training step): here, CORRECT rollouts
    contribute NO self-distillation loss this step (there is no "failure" to
    correct); only INCORRECT rollouts are trained, each against a peer's
    correct solution if one exists in the group. This avoids the otherwise-
    tautological case of a correct rollout using itself as its own "correct
    solution" target.
  - If ZERO rollouts in a group are correct, that whole question contributes
    no loss this step (no peer solution exists for any of its rollouts) -
    same documented tradeoff as RLSD's all-identical-reward GRPO collapse.
  - KNOWN LIMITATION (same as RLSD, flagged not silently patched): training
    rollouts use max_new_tokens=1024 (this project's own TROPIC-P/OPSD
    training-rollout length, matching "same setting"), which a "thinking"
    Qwen3 model frequently will not finish within - so a nontrivial fraction
    of rollouts, and even whole groups, may never produce a gradeable
    \\boxed{} answer. Not resolved unilaterally here; see run_rlsd_experiment.py's
    matching note.

CRITICAL DIFFERENCE FROM QWEN3 - NO NATIVE THINKING TOGGLE, TRICK REQUIRED:
verified live against the REAL allenai/Olmo-3-7B-Think tokenizer on
2026-09-15 (not assumed): its chat template has NO `enable_thinking` branch
at all - passing it is silently ignored. Its `add_generation_prompt=True`
output already ends on an OPENED `<think>` tag, so the model always starts
reasoning by default (matches the self-teacher's own always-thinking prefix
here, unmodified). The STUDENT rollout must be non-thinking -
CONFIG["empty_think_suffix"] = "\\n\\n</think>\\n\\n" is threaded through
tropic.data.build_student_prompt (via tropic.verifier_data.
load_opsd_math_examples_with_answer) so the auto-opened `<think>` is
immediately closed empty (the community "empty think prefill" trick
documented for DeepSeek-R1-style models - see huggingface.co/deepseek-ai/
DeepSeek-R1-Distill-Qwen-14B/discussions/11). UNVERIFIED END-TO-END: only
confirmed the rendered PROMPT ends in `<think>\\n\\n</think>\\n\\n` - NOT
confirmed Olmo-3-7B-Think actually skips reasoning after seeing this the way
DeepSeek-R1 does. Inspect real student rollout text early in a run before
trusting this.

Usage - identical shape to run_rlsd_experiment_olmo7b.py:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/run_sdpo_experiment_olmo7b.py \\
        --model-path /path/to/local/Olmo-3-7B-Think --seed 0 --output-dir results_sdpo_olmo7b_seed0 --skip-eval
    CUDA_VISIBLE_DEVICES=$i PYTHONPATH=. python scripts/run_sdpo_experiment_olmo7b.py \\
        --model-path /path/to/local/Olmo-3-7B-Think --output-dir results_sdpo_olmo7b_seed0 --skip-train \\
        --checkpoint-path results_sdpo_olmo7b_seed0/sdpo_checkpoint_step100 \\
        --eval-benchmarks hmmt25 --shard-index $i --num-shards 4
"""
import argparse
import json
import re
import time
import gc
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="results_sdpo_olmo7b")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--eval-benchmarks", nargs="+", default=["aime25", "aime26", "hmmt25"],
                     choices=["aime24", "aime25", "aime26", "hmmt25"])
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
parser.add_argument("--eval-engine", choices=["vllm", "hf"], default="hf",
                     help="Backend for eval generation. 'hf' (default) uses plain "
                          "transformers.generate(). 'vllm' generates via data-parallel vLLM "
                          "server replicas (tropic.vllm_rollout_client) - requires "
                          "--vllm-gpu-ids (free GPUs, disjoint from this process's own "
                          "CUDA_VISIBLE_DEVICES); several times faster for this workload (many "
                          "long sequences).")
parser.add_argument("--vllm-gpu-ids", default=None,
                     help="Comma-separated GPU ids to run vLLM eval replicas on (e.g. '5,6,7') - "
                          "MUST be free GPUs, disjoint from this process's own CUDA_VISIBLE_DEVICES "
                          "and from any other job. Required with --eval-engine vllm.")
parser.add_argument("--vllm-base-port", type=int, default=8100,
                     help="First port for the vLLM replica servers (replica i uses base_port+i) - "
                          "use a DIFFERENT range per concurrently-running vLLM-enabled job on the "
                          "same machine to avoid port collisions.")
parser.add_argument("--vllm-executable", default="vllm",
                     help="Path to the `vllm` CLI - pass an ABSOLUTE PATH when vLLM lives in a "
                          "different conda env than this process.")
parser.add_argument("--use-vllm-rollout", action="store_true",
                     help="Generate training rollouts via data-parallel vLLM server replicas "
                          "(tropic.vllm_rollout_client) instead of sequential HF model.generate() "
                          "calls. Requires --vllm-gpu-ids.")
parser.add_argument("--skip-train", action="store_true")
parser.add_argument("--checkpoint-path", default=None)
parser.add_argument("--skip-eval", action="store_true")
parser.add_argument("--shard-index", type=int, default=0)
parser.add_argument("--num-shards", type=int, default=1)
args = parser.parse_args()
if args.skip_train and not args.checkpoint_path:
    parser.error("--skip-train requires --checkpoint-path")
if args.use_vllm_rollout and not args.vllm_gpu_ids:
    parser.error("--use-vllm-rollout requires --vllm-gpu-ids")

t_start = time.time()
RESULTS_DIR = Path(__file__).resolve().parent.parent / args.output_dir
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

CHECKPOINT_STEPS = {20, 25, 40, 50, 60, 75, 80, 100}


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
    student_thinking=False,
    max_length=20000,
    # OLMo-3-7B-Think has no native enable_thinking toggle - this closes its
    # auto-opened <think> tag empty when enable_thinking=False (student only)
    # - see this file's own docstring (UNVERIFIED end-to-end).
    empty_think_suffix="\n\n</think>\n\n",
    lora=LORA_CONFIG,
    # temperature=1.0/top_p=1.0/top_k=-1 (unrestricted) is this offline B200 deployment's own
    # deliberate choice (see offline_rlsd_sdpo_b200's project_commands.sh) - NOT copied from the
    # online run_sdpo_experiment_4b.py, which keeps OPSD's real recipe unchanged.
    train=dict(training_steps=args.training_steps, lr=5e-6, max_grad_norm=0.1,
               max_new_tokens=1024, temperature=1.0, top_p=1.0, top_k=-1, seed=args.seed,
               gradient_checkpointing=True),
    sdpo=dict(
        group_size=2,           # REDUCED from the paper-matching G=8 for safety margin on this
                                 # untested-on-B200 pipeline (the 4B script uses G=4) - note this
                                 # changes the self-teacher's group statistics, not just a
                                 # memory/throughput knob
        questions_per_step=2,   # 2*2=4 rollouts/step; the 4B script uses 4*4=16
        top_k=100,              # verified live against the REAL reference script's
                                 # `distillation_topk=100` (run_local_sdpo.sh, github.com/lasgroup/SDPO)
        default_mass=1e-5,      # numerical-stability tail floor (reused from TROPIC-P's Remark 4
                                 # sparsification primitive)
        alpha=0.5,               # generalized-JSD mixture weight between student/teacher - verified
                                  # live against the REAL reference script's own default
                                  # (run_local_sdpo.sh: ALPHA=0.5) - see tropic/sdpo.py's module
                                  # docstring for the full correction/derivation. NOT a teacher-
                                  # regularization parameter (an earlier version of this file
                                  # wrongly treated it as one - see the CORRECTION note there).
    ),
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


ALL_BENCHMARKS = {"aime24", "aime25", "hmmt25"}


def infer_step_suffix(checkpoint_path):
    if not checkpoint_path:
        return ""
    m = re.search(r"_checkpoint_step(\d+)/?$", str(checkpoint_path).rstrip("/\\"))
    return f"_step{m.group(1)}" if m else ""


def bench_suffix_for(benchmarks):
    if set(benchmarks) == ALL_BENCHMARKS:
        return ""
    return "_" + "-".join(sorted(benchmarks))


from tropic.verifier_data import load_opsd_math_examples_with_answer
from tropic.model import ContextualPolicy
from tropic.group_rollout import generate_rollout_group
from tropic.sdpo import build_sdpo_self_teacher_prefix, sdpo_loss
from tropic.eval import grade, load_benchmark, score_generations
from tropic.vllm_rollout_client import (
    generate_eval_batch_parallel, generate_rollout_batch_parallel,
    launch_vllm_replicas, refresh_lora_adapter, shutdown_vllm_replicas,
)

train_examples = None
if not args.skip_train:
    num_train_examples = CONFIG["train"]["training_steps"] * CONFIG["sdpo"]["questions_per_step"]
    tick(f"loading + filtering OPSD dataset (with gold answers), num_examples={num_train_examples}")
    train_examples = load_opsd_math_examples_with_answer(
        tokenizer,
        num_examples=num_train_examples,
        only_correct=True,
        student_thinking=CONFIG["student_thinking"],
        seed=CONFIG["train"]["seed"],
        max_length=CONFIG["max_length"],
        empty_think_suffix=CONFIG["empty_think_suffix"],
    )
    tick(f"dataset ready, {len(train_examples)} examples")


def train_run_sdpo(model, examples, cfg, tag, use_vllm_rollout=False, vllm_gpu_ids=None,
                    vllm_base_port=8100, vllm_executable="vllm"):
    # fixed_teacher=False: SDPO's self-teacher is a SINGLE live forward pass
    # (Eq. 1's literal stopgrad(pi_theta(.|x,f,y<t))) - verified against the
    # real reference implementation (core_algos.py's compute_self_distillation_loss)
    # that there is no second/frozen-teacher pass at all (see tropic/sdpo.py's
    # module docstring CORRECTION note - an earlier version of this file
    # wrongly added one).
    policy = ContextualPolicy(model, tokenizer, fixed_teacher=False)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    tick(f"[{tag}] optimizer over {n_trainable} trainable params")
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg["train"]["lr"])

    training_steps = cfg["train"]["training_steps"]
    questions_per_step = cfg["sdpo"]["questions_per_step"]
    group_size = cfg["sdpo"]["group_size"]
    top_k = cfg["sdpo"]["top_k"]
    default_mass = cfg["sdpo"]["default_mass"]
    alpha = cfg["sdpo"]["alpha"]
    example_idx = 0
    logs = []
    run_start = time.time()

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
            total_rollouts = questions_per_step * group_size
            micro_losses, micro_correct, micro_trained = [], [], 0

            step_examples = [examples[(example_idx + i) % len(examples)] for i in range(questions_per_step)]
            example_idx += questions_per_step

            if use_vllm_rollout:
                flat_prompts = [ex.student_prompt_ids for ex in step_examples for _ in range(group_size)]
                flat_rollouts = generate_rollout_batch_parallel(
                    vllm_ports, flat_prompts, tokenizer,
                    max_new_tokens=cfg["train"]["max_new_tokens"], temperature=cfg["train"]["temperature"],
                    top_p=cfg["train"]["top_p"], top_k=cfg["train"]["top_k"],
                )
                grouped_rollouts = [
                    [(ids.to(device), gl) for ids, gl in flat_rollouts[i * group_size:(i + 1) * group_size]]
                    for i in range(questions_per_step)
                ]

            for q_idx, ex in enumerate(step_examples):
                if use_vllm_rollout:
                    rollouts = grouped_rollouts[q_idx]
                else:
                    rollouts = generate_rollout_group(
                        model, tokenizer, ex.student_prompt_ids, group_size,
                        max_new_tokens=cfg["train"]["max_new_tokens"],
                        temperature=cfg["train"]["temperature"], top_p=cfg["train"]["top_p"],
                        top_k=cfg["train"]["top_k"],
                    )
                texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids, _ in rollouts]
                correct_flags = [grade(t, ex.answer) for t in texts]
                micro_correct.extend(correct_flags)

                successful_text = next((t for t, ok in zip(texts, correct_flags) if ok), None)
                if successful_text is None:
                    continue  # no peer solution available this question - skip (see module docstring)

                self_teacher_prefix_ids = build_sdpo_self_teacher_prefix(
                    tokenizer, ex.question, successful_text,
                    enable_thinking=True, max_length=cfg["max_length"],
                    empty_think_suffix=cfg["empty_think_suffix"],
                )

                for (generated_ids, gen_len), is_correct in zip(rollouts, correct_flags):
                    if is_correct:
                        continue  # only failed rollouts get self-distillation feedback (see docstring)

                    log_student = policy.forward_student(ex.student_prompt_ids, generated_ids)
                    log_teacher = policy.forward_teacher(self_teacher_prefix_ids, generated_ids)

                    loss = sdpo_loss(log_student, log_teacher, generated_ids, top_k, default_mass, alpha)
                    (loss / total_rollouts).backward()
                    micro_losses.append(loss.item())
                    micro_trained += 1

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=cfg["train"]["max_grad_norm"])
            optimizer.step()

            if use_vllm_rollout:
                model.save_pretrained(str(vllm_scratch_dir))
                refresh_lora_adapter(vllm_ports, str(vllm_scratch_dir))

            mean_loss = sum(micro_losses) / len(micro_losses) if micro_losses else 0.0
            record = {
                "tag": tag, "step": step, "loss": mean_loss, "elapsed_sec": time.time() - t0,
                "frac_correct_rollouts": sum(micro_correct) / len(micro_correct) if micro_correct else 0.0,
                "rollouts_trained_this_step": micro_trained,
            }
            logs.append(record)
            avg_sec = (time.time() - run_start) / (step + 1)
            record["eta_min"] = round(avg_sec * (training_steps - step - 1) / 60.0, 1)
            tick(f"[{tag}] {json.dumps(record)}")
            if mean_loss != mean_loss:
                raise RuntimeError(f"[{tag}] loss became NaN at step={step}")

            if (step + 1) in CHECKPOINT_STEPS:
                ckpt_dir = RESULTS_DIR / f"{tag}_checkpoint_step{step + 1}"
                model.save_pretrained(str(ckpt_dir))
                tick(f"[{tag}] checkpoint saved: {ckpt_dir}")

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
    """Identical to run_full_experiment.py's own eval sampler."""
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
    """Identical logic/shard contract to run_full_experiment.py's own
    `evaluate_model`."""
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
        all_generations[bench_name] = {
            "problem_ids": [p.problem_id for p in problems],
            "golds": golds,
            "generations": generations,
        }
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
    """Evaluates the ALREADY-SAVED checkpoint via data-parallel vLLM server
    replicas (tropic.vllm_rollout_client - the same separate-process/HTTP
    architecture used for training rollouts elsewhere in this project),
    generating every (problem, k-sample) completion in batched, continuous-
    batched calls - matches OPSD's own eval/evaluate_math.py approach
    (LoRA-served completions, k samples/problem). Must be called AFTER the
    HF training/eval model for this tag has been `del`eted and
    `clear_cuda_cache()`d, since the replicas need their own GPU(s)
    (`vllm_gpu_ids`, disjoint from this process's own GPU)."""
    from tropic.vllm_rollout_client import (
        generate_eval_batch_parallel, launch_vllm_replicas, refresh_lora_adapter, shutdown_vllm_replicas,
    )

    tick(f"[{tag}] launching {len(vllm_gpu_ids)} vLLM eval replica(s) on GPU {vllm_gpu_ids} "
         f"(ports {vllm_base_port}..{vllm_base_port + len(vllm_gpu_ids) - 1})")
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
        benchmarks = cfg["eval"]["benchmarks"]
        problems_by_bench = {b: load_benchmark(b, num_problems=cfg["eval"]["num_problems"]) for b in benchmarks}
        all_prompts, spans = [], {}
        for bench_name in benchmarks:
            start = len(all_prompts)
            for p in problems_by_bench[bench_name]:
                messages = [{"role": "user", "content": p.problem + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."}]
                all_prompts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=cfg["eval"]["enable_thinking"],
                ))
            spans[bench_name] = (start, len(all_prompts))

        tick(f"[{tag}] {len(all_prompts)} problems x {cfg['eval']['k']} samples across {', '.join(benchmarks)} in one vLLM batch")
        all_outputs = generate_eval_batch_parallel(
            ports, all_prompts,
            max_tokens=cfg["eval"]["max_new_tokens"], temperature=cfg["eval"]["temperature"],
            top_p=cfg["eval"]["top_p"], n=cfg["eval"]["k"],
            top_k=top_k_cfg if top_k_cfg and top_k_cfg > 0 else None,
            min_p=cfg["eval"].get("min_p", 0.0) or None,
        )

        for bench_name in benchmarks:
            start, end = spans[bench_name]
            problems = problems_by_bench[bench_name]
            generations = all_outputs[start:end]
            golds = [p.answer for p in problems]
            res = score_generations(bench_name, generations, golds)
            results[bench_name] = res
            all_generations[bench_name] = {
                "problem_ids": [p.problem_id for p in problems],
                "golds": golds,
                "generations": generations,
            }
            tick(f"[{tag}] {bench_name} avg@{res.k}={res.avg_at_k:.3f} pass@{res.k}={res.pass_at_k:.3f}")

        (RESULTS_DIR / f"{tag}_generations.json").write_text(json.dumps(all_generations, indent=2))
        serializable = {k: {"avg_at_k": v.avg_at_k, "pass_at_k": v.pass_at_k, "k": v.k} for k, v in results.items()}
        (RESULTS_DIR / f"{tag}_eval_results.json").write_text(json.dumps(serializable, indent=2))

        tick(f"[{tag}] eval done (vLLM) - scores + raw generations saved under {RESULTS_DIR}/")
        return results
    finally:
        shutdown_vllm_replicas(procs)
        tick(f"[{tag}] vLLM eval replicas shut down")


TAG = "sdpo"
EVAL_TAG = TAG + infer_step_suffix(args.checkpoint_path if args.skip_train else None) + bench_suffix_for(args.eval_benchmarks)

if args.skip_train:
    tick(f"=== {TAG}: --skip-train, loading checkpoint {args.checkpoint_path} ===")
    model = load_model_from_checkpoint(args.checkpoint_path)
else:
    model = load_fresh_model()
    model.train()
    vllm_gpu_ids = [int(x) for x in args.vllm_gpu_ids.split(",")] if args.vllm_gpu_ids else None
    train_run_sdpo(model, train_examples, CONFIG, TAG,
                    use_vllm_rollout=args.use_vllm_rollout, vllm_gpu_ids=vllm_gpu_ids,
                    vllm_base_port=args.vllm_base_port, vllm_executable=args.vllm_executable)
    if args.skip_eval:
        tick(f"=== {TAG}: --skip-eval, training done, NOT evaluating in this process. "
             f"Checkpoints saved under {RESULTS_DIR}/. Now launch eval shards with "
             f"--skip-train --checkpoint-path {RESULTS_DIR}/{TAG}_checkpoint_step<N> "
             f"on separate GPUs, then run scripts/combine_shards.py --tag {TAG}.")
        sys.exit(0)

if args.eval_engine == "hf":
    result = evaluate_model(model, CONFIG, EVAL_TAG, shard_index=args.shard_index, num_shards=args.num_shards)
    del model
    clear_cuda_cache()
else:
    checkpoint_dir_for_vllm = (
        Path(args.checkpoint_path) if args.skip_train
        else RESULTS_DIR / f"{TAG}_checkpoint_step{CONFIG['train']['training_steps']}"
    )
    del model
    clear_cuda_cache()
    tick(f"model_{TAG} freed - GPU clear for vLLM")
    eval_vllm_gpu_ids = [int(x) for x in args.vllm_gpu_ids.split(",")] if args.vllm_gpu_ids else None
    if not eval_vllm_gpu_ids:
        parser.error("--eval-engine vllm requires --vllm-gpu-ids")
    result = evaluate_model_vllm(checkpoint_dir_for_vllm, CONFIG, EVAL_TAG, eval_vllm_gpu_ids,
                                  args.vllm_base_port, args.vllm_executable)

if args.num_shards > 1:
    step_num = infer_step_suffix(args.checkpoint_path).lstrip("_step") or "<N>"
    tick(f"ALL DONE (eval shard {args.shard_index}/{args.num_shards}) - once all shards finish: "
         f"python scripts/combine_shards.py --output-dir {args.output_dir} --tag {TAG} "
         f"--num-shards {args.num_shards} --checkpoint-step {step_num} --benchmarks {' '.join(args.eval_benchmarks)}")
    sys.exit(0)

tick(f"ALL DONE - results saved under {RESULTS_DIR}/")
