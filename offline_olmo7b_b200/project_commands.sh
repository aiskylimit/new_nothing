
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.

source /mnt/local/uvenvs/tropic/bin/activate

# ============================================================
# 1. Paths - MUST match how download.txt's @PROJECT@ was actually resolved.
#    Fill in PROJECT_NAME before running. Dataset paths reuse the SAME
#    locations as offline_rlsd_sdpo_b200/offline_tropic_g_b200's own
#    download.txt (already downloaded there if those ran first under the
#    same @PROJECT@) - only the OLMo model itself is new.
# ============================================================
PROJECT_NAME="aiskylimit_new_nothingnew_2"
BASE_DIR="/mnt/local/${PROJECT_NAME}/tropic_baselines"
MODEL_OLMO="${BASE_DIR}/models/Olmo-3-7B-Think"
export TROPIC_TRAIN_DATA_PATH="${BASE_DIR}/data/train"
export TROPIC_EVAL_DATA_DIR="${BASE_DIR}/data/eval"

# ============================================================
# 2. GPU topology - 2 GPUs (0,1): GPU 0 (main/training) + GPU 1 (1 vLLM
#    replica) during TRAINING - same 2-GPU shape as
#    offline_rlsd_sdpo_b200/offline_tropic_g_b200, kept as-is per explicit
#    request (no extra replica GPUs reserved for the whole run).
#
#    EVAL is different: by the time eval_checkpoint() runs, it is a
#    SEPARATE process invocation (train already fully exited) - GPU 0 is
#    genuinely idle then, not just "freed in-process". So eval uses BOTH
#    GPU 0 and GPU 1 as 2 data-parallel vLLM replicas (EVAL_VLLM_GPU_IDS),
#    splitting each benchmark's problems across them for ~2x eval
#    throughput - no vLLM sleep/wake_up trickery needed, since there is no
#    process ever alive on GPU 0 at the same time eval's replicas start.
#    Change MAIN_GPU/VLLM_GPU_IDS/EVAL_VLLM_GPU_IDS below if that ever changes.
# ============================================================
MAIN_GPU=0
VLLM_GPU_IDS="1"
EVAL_VLLM_GPU_IDS="0,1"
VLLM_BASE_PORT=8100
VLLM_EXECUTABLE="vllm"
GPU_MEM_UTIL=0.9
# Without this, vLLM auto-detects Olmo-3-7B-Think's own max_position_embeddings
# (can be far larger than anything this project ever sends it) and sizes its
# KV-cache pool/max concurrent sequence count against THAT - reserving
# per-sequence memory for a context length nothing reaches, so only a small
# number of sequences can run concurrently even on a B200's 183GB (observed
# live: ~17GB used out of 183GB on the single vLLM replica). Capping this to
# what's actually needed (eval's max_new_tokens=38912 plus a short prompt)
# lets the SAME gpu_memory_utilization budget fit many more concurrent
# sequences, directly increasing throughput without adding replica GPUs.
VLLM_MAX_MODEL_LEN=41000

# Per-method eval cadence (checkpoints are still SAVED at all steps below by
# each script's own CHECKPOINT_STEPS - 10 for TROPIC-G (adds 10/15) - this
# only controls which of those get EVALUATED. RLSD/SDPO's own cadence
# (CHECKPOINTS_RLSD/CHECKPOINTS_SDPO="25 50 75 100") no longer applies here -
# both already fully trained+evaluated in a prior run (see section 3 below).
CHECKPOINTS_TROPIC="10 15 20 25 40 50 75 100"
BENCHMARKS="aime25 aime26 hmmt25"

train() {
  local script=$1 model_path=$2 output_dir=$3 tag=$4
  local final_ckpt="${output_dir}/${tag}_checkpoint_step100"
  if [ -d "$final_ckpt" ]; then
    echo "skip (already trained): $final_ckpt exists"
    return 0
  fi
  echo "=== training: $script -> $output_dir ==="
  env CUDA_VISIBLE_DEVICES=${MAIN_GPU} python "scripts/${script}" \
      --model-path "${model_path}" --seed 0 --output-dir "${output_dir}" --skip-eval \
      --use-vllm-rollout --vllm-gpu-ids "${VLLM_GPU_IDS}" --vllm-base-port ${VLLM_BASE_PORT} \
      --vllm-executable "${VLLM_EXECUTABLE}" --vllm-gpu-memory-utilization ${GPU_MEM_UTIL} \
      --vllm-max-model-len ${VLLM_MAX_MODEL_LEN} \
      2>&1 | tee "${output_dir}_train.log"
}

train_dry_run() {
  local script=$1 model_path=$2 output_dir=$3
  echo "=== DRY RUN (3 steps, sanity check only): $script ==="
  env CUDA_VISIBLE_DEVICES=${MAIN_GPU} python "scripts/${script}" \
      --model-path "${model_path}" --seed 0 --output-dir "${output_dir}_dryrun" --skip-eval \
      --training-steps 3 \
      --use-vllm-rollout --vllm-gpu-ids "${VLLM_GPU_IDS}" --vllm-base-port ${VLLM_BASE_PORT} \
      --vllm-executable "${VLLM_EXECUTABLE}" --vllm-gpu-memory-utilization ${GPU_MEM_UTIL} \
      --vllm-max-model-len ${VLLM_MAX_MODEL_LEN} \
      2>&1 | tee "${output_dir}_dryrun_train.log"
  echo "Dry run finished - check ${output_dir}_dryrun_train.log for errors before continuing."
  echo "IMPORTANT (OLMo-specific): also inspect a few RAW student rollouts in"
  echo "this dry run (not just that it doesn't crash) - the empty-think-prefill"
  echo "trick (tropic/data.py's empty_think_suffix) is UNVERIFIED end-to-end for"
  echo "Olmo-3-7B-Think; confirm the model actually skips reasoning and doesn't"
  echo "just ignore the closed empty <think></think> block before trusting a full run."
}

eval_checkpoint() {
  local script=$1 model_path=$2 output_dir=$3 tag=$4 step=$5
  local ckpt="${output_dir}/${tag}_checkpoint_step${step}"
  local eval_log="${output_dir}_eval_step${step}.log"
  if [ -f "$eval_log" ] && grep -q "ALL DONE" "$eval_log" 2>/dev/null; then
    echo "skip (already evaluated): $eval_log shows ALL DONE"
    return 0
  fi
  if [ ! -d "$ckpt" ]; then
    echo "skip (no checkpoint): $ckpt"
    return 0
  fi
  echo "=== eval: $script step ${step} ==="
  env CUDA_VISIBLE_DEVICES=${MAIN_GPU} python "scripts/${script}" \
      --model-path "${model_path}" --skip-train \
      --checkpoint-path "$ckpt" \
      --output-dir "${output_dir}" \
      --eval-engine vllm --vllm-gpu-ids "${EVAL_VLLM_GPU_IDS}" --vllm-base-port ${VLLM_BASE_PORT} \
      --vllm-executable "${VLLM_EXECUTABLE}" --vllm-gpu-memory-utilization ${GPU_MEM_UTIL} \
      --vllm-max-model-len ${VLLM_MAX_MODEL_LEN} \
      --eval-benchmarks ${BENCHMARKS} \
      2>&1 | tee "${output_dir}_eval_step${step}.log"
}

# Aggregates avg@12/pass@12 lines from whichever eval log(s) match the given
# glob pattern(s) into one JSON + printed table - called once at the very
# end for all 5 methods (RLSD/SDPO's own eval logs are already on disk from
# a prior run - see section 3 below).
aggregate_results() {
  local outfile=$1
  shift
  python - "$outfile" "$@" <<'PYEOF'
import glob, json, re, sys

outfile = sys.argv[1]
patterns = sys.argv[2:]

pattern = re.compile(
    r"\[(?P<tag>[\w]+)_step(?P<step>\d+)\]\s+(?P<bench>aime25|aime26|hmmt25)\s+"
    r"avg@12=(?P<avg>[\d.]+)\s+pass@12=(?P<pass_>[\d.]+)"
)
files = []
for p in patterns:
    files.extend(glob.glob(p))
rows = []
for path in sorted(set(files)):
    text = open(path, encoding="utf-8", errors="replace").read()
    for m in pattern.finditer(text):
        rows.append({
            "file": path, "tag": m["tag"], "step": int(m["step"]), "benchmark": m["bench"],
            "avg@12": float(m["avg"]), "pass@12": float(m["pass_"]),
        })

with open(outfile, "w") as f:
    json.dump(rows, f, indent=2)

print(f"{'tag':<16} {'step':>4}  {'benchmark':<8} {'avg@12':>7} {'pass@12':>8}")
for r in rows:
    print(f"{r['tag']:<16} {r['step']:>4}  {r['benchmark']:<8} {r['avg@12']:>7.3f} {r['pass@12']:>8.3f}")
print(f"\n{len(rows)} rows written to {outfile}")
PYEOF
}

# ============================================================
# 3. RLSD and SDPO on OLMo-3-7B-Think already trained+evaluated (all 4
#    checkpoints each, confirmed ALL DONE in a prior run of this script -
#    see results_rlsd_olmo7b/ and results_sdpo_olmo7b/ already on disk).
#    Their scripts/modules were removed from this package (run_rlsd_
#    experiment_olmo7b.py, run_sdpo_experiment_olmo7b.py, tropic/sdpo.py,
#    tropic/verifier_data.py) so this script no longer re-trains or
#    re-evaluates them - rerunning this file now goes straight to
#    TROPIC-G/TROPIC-L below. The existing results_rlsd_olmo7b_eval_step*.log/
#    results_sdpo_olmo7b_eval_step*.log files are untouched and still feed
#    into the final aggregate_results call at the bottom of this script.
# ============================================================

# ============================================================
# 4. Train + eval TROPIC-G (top-k64 only - the real baseline's sparsification
#    setting; the full-vocab variant was dropped from this package) - the
#    proposal. Already ran on Qwen3-4B/8B separately (offline_tropic_g_b200/)
#    - this is only the OLMo addition. Check its own results later with a separate command
#    (grep the results_tropic_g*_eval_step*.log files, or re-run
#    aggregate_results against them) rather than waiting on this script to finish.
# ============================================================
train run_tropic_g_topk64_olmo7b.py "${MODEL_OLMO}" results_tropic_g_topk64_olmo7b tropic_g_topk64_olmo
for step in $CHECKPOINTS_TROPIC; do eval_checkpoint run_tropic_g_topk64_olmo7b.py "${MODEL_OLMO}" results_tropic_g_topk64_olmo7b tropic_g_topk64_olmo "$step"; done

# ============================================================
# 5. Train + eval TROPIC-L (leverage-allocated trust regions, TROPIC_Proposal_v8)
#    - runs only after TROPIC-G above is completely done. Self-value (SV)
#    process credit only (the only credit source built anywhere in this
#    codebase) - checkpoints saved at 5/10/15/20/25/40/50/60/75/80/100
#    (TROPIC-L's own cadence, includes the early 5/10/15 regardless of
#    CHECKPOINTS_TROPIC below which only controls which get EVALUATED here).
# ============================================================
train run_tropic_l_olmo7b.py "${MODEL_OLMO}" results_tropic_l_olmo7b tropic_l_olmo
for step in $CHECKPOINTS_TROPIC; do eval_checkpoint run_tropic_l_olmo7b.py "${MODEL_OLMO}" results_tropic_l_olmo7b tropic_l_olmo "$step"; done

# ============================================================
# 6. Aggregate every avg@12/pass@12 line (all 5 methods) into one final
#    table + JSON.
# ============================================================
aggregate_results final_results.json "results_*_eval_step*.log"
echo "ALL DONE - see final_results.json for the aggregated table."
