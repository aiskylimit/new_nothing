#!/bin/bash
# GRIT (GRPO safety + NSPO null-space projection + KL trust region) on Qwen2.5-3B-Instruct, 4 x B200 180 GB.
# Order: train (40 steps) -> eval base + checkpoints (safety + general) -> summary table.
# Layout: TRAIN_GPUS run the FP32 actor (torchrun, one rank per GPU, each with its own bf16 Qwen3Guard);
#   ROLLOUT_GPU runs the FP32 vLLM rollout alone. Eval spreads its vLLM engines over EVAL_GPUS.
# Every stage is resumable: training resumes from the latest checkpoint, eval resumes per model/benchmark.
# Memory: training starts with the most aggressive MEM_CONFIGS entry and falls back to the next one on
#   CUDA OOM (resuming from the latest checkpoint); the entry in use is kept in mem_config_in_use.txt.
# Eval: EVAL_SLOTS vLLM engines per GPU run models in parallel (base, step 20, step 40 at once).
# commands.sh runs exactly "cd ./GRIT" + "bash ./project_commands.sh"; to change a setting, edit its
# default below (GPU_IDS, RUN_ID, LAMBDA_PRES, MAX_STEPS, STAGES, EVAL_STEPS, EVAL_MAX_SAMPLES, MEM_CONFIGS,
# MEM_CONFIG, EVAL_GPUS, EVAL_SLOTS). The same names also work as env vars when running by hand.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
export PYTHONPATH=.

VENV=/mnt/local/uvenvs/grit
if [ ! -f "${VENV}/bin/activate" ]; then
  echo "ERROR: ${VENV} does not exist. Run the env job first: header '#i grit.txt' in commands.sh" >&2
  exit 1
fi
source "${VENV}/bin/activate"

# Paths - must match download.txt.
PROJECT_NAME="aiskylimit_new_nothingnew_2"
BASE_DIR="/mnt/local/${PROJECT_NAME}/grit"
MODEL="${BASE_DIR}/models/Qwen2.5-3B-Instruct"
GUARD="${BASE_DIR}/models/Qwen3Guard-Gen-4B"   # training reward
JUDGE="${BASE_DIR}/models/wildguard"            # eval judge (safety + over-refusal)
ARTIFACTS="${BASE_DIR}/prepared/Qwen--Qwen2.5-3B-Instruct"
RAW="${BASE_DIR}/data/raw"
TASK_FILE="${ARTIFACTS}/data/task/task_train.parquet"
EVAL_FILE="${ARTIFACTS}/data/eval/evaluation_prompts.parquet"
PRESERVE_FILE="${ARTIFACTS}/data/preservation/split_v1/qwen--qwen2.5-3b-instruct/monitor/preserve_contexts.parquet"
PROJECTORS="${ARTIFACTS}/artifacts/qwen--qwen2.5-3b-instruct_split_v1_projectors.pt"

export HF_HOME="${BASE_DIR}/.cache/huggingface"
# Each rollout sync writes a ~12 GB FP32 weight file to a temp dir, and vLLM/torch.compile keep caches:
# keep both on /mnt/local instead of a possibly small /tmp or home directory.
export TMPDIR="${BASE_DIR}/tmp" VLLM_CACHE_ROOT="${BASE_DIR}/.cache/vllm"
mkdir -p "${TMPDIR}" "${VLLM_CACHE_ROOT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
# vLLM's engine opens a 1-process torch.distributed group; keep it on loopback (as the other projects'
# MASTER_ADDR=127.0.0.1) instead of the host's outward IP. Applies to the rollout and every eval engine.
export VLLM_HOST_IP=127.0.0.1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# commands.sh only runs "cd ./GRIT && bash ./project_commands.sh", so every default below must be right
# as is.
# GPU_IDS: the B200s assigned to GRIT on the shared server. The last one is the vLLM rollout GPU, the
# others train (one torchrun rank each; 321 prompts and 48 KL contexts are split across them).
# A single GPU (e.g. GPU_IDS="4") switches to the colocated layout: actor + vLLM (sleep mode) on it.
# Eval uses all of them. Change it here if needed.
GPU_IDS="4,5,6,7"
ROLLOUT_GPU=${GPU_IDS##*,}
TRAIN_GPUS=${GPU_IDS%,*}
COLOCATED=()
if [[ "${GPU_IDS}" != *,* ]]; then COLOCATED=(--allow-colocated-rollout); fi
NUM_TRAIN_GPUS=$(( $(tr -cd , <<< "${TRAIN_GPUS}" | wc -c) + 1 ))
echo "GPU_IDS=${GPU_IDS}: train on ${TRAIN_GPUS} (${NUM_TRAIN_GPUS} ranks), rollout on ${ROLLOUT_GPU}"
for gpu in ${GPU_IDS//,/ }; do
  FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')
  echo "  GPU ${gpu}: ${FREE_MB} MiB free"
  if [ "${FREE_MB}" -lt 150000 ]; then
    echo "ERROR: GPU ${gpu} has only ${FREE_MB} MiB free; GRIT needs whole B200s (>= 150 GB each)" >&2
    nvidia-smi
    exit 1
  fi
done
RUN_ID=${RUN_ID:-grit_3b_lam1}
LAMBDA_PRES=${LAMBDA_PRES:-1}
MAX_STEPS=${MAX_STEPS:-40}
STAGES=${STAGES:-train,eval,summary}
# Checkpoints are saved every 5 steps; only these are evaluated (base is always evaluated).
EVAL_STEPS=${EVAL_STEPS:-"20 40"}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-}   # e.g. 50 for a quick eval smoke test
EVAL_GPUS=${EVAL_GPUS:-${GPU_IDS}}
# vLLM engines per eval GPU: enough to run base, step 20 and step 40 at once.
EVAL_SLOTS=${EVAL_SLOTS:-$(( (2 + $(tr -cd , <<< "${EVAL_GPUS}" | wc -c)) > 3 ? 1 : 3 ))}
# Memory configs, TASK_MB:PRES_MB:REWARD_BS:ROLLOUT_MEM:ROLLOUT_SLEEP, most aggressive first.
# ROLLOUT_MEM is vLLM's share of the dedicated rollout GPU; sleep mode is only useful when colocated.
# Colocated: vLLM gets a smaller share and sleeps between rollouts; the last entry is the original setting.
if [ ${#COLOCATED[@]} -gt 0 ]; then
  MEM_CONFIGS=${MEM_CONFIGS:-"16:4:64:0.40:1 8:2:32:0.40:1 8:2:32:0.30:0"}
else
  MEM_CONFIGS=${MEM_CONFIGS:-"16:4:64:0.85:0 8:2:32:0.85:0 4:1:16:0.85:0"}
fi
OUT_DIR="${BASE_DIR}/outputs/${RUN_ID}"  # checkpoints (~12 GB each) stay on /mnt/local
EVAL_WORK="${BASE_DIR}/eval/${RUN_ID}"   # generations, WildGuard labels
RESULTS="${ROOT}/results_${RUN_ID}"      # small files only: logs, metrics, eval results, summary
mkdir -p "${RESULTS}/logs" "${RESULTS}/train" "${RESULTS}/eval" "${BASE_DIR}/outputs"
# Everything printed below also goes to one log per run (console output is unchanged).
exec > >(tee -a "${RESULTS}/logs/full_run_$(date +%Y%m%d_%H%M%S).log") 2>&1
for path in "${MODEL}/config.json" "${GUARD}/config.json" "${ARTIFACTS}/SHA256SUMS.json"; do
  [ -f "${path}" ] || { echo "ERROR: missing ${path}. Run the download job first: GRIT/download.txt as a '#d' job"; exit 1; }
done
# Eval inputs are checked now, not after hours of training.
if [[ ",${STAGES}," == *",eval,"* ]]; then
  [ -f "${JUDGE}/config.json" ] || { echo "ERROR: missing ${JUDGE}/config.json (allenai/wildguard, see download.txt)"; exit 1; }
  python -c 'import sys; from scripts import eval_lib as E; [E.source_file(sys.argv[1], b) for b in E.SOURCES]' "${RAW}" \
    || { echo "ERROR: eval benchmark files missing under ${RAW}. Run the download job first: GRIT/download.txt"; exit 1; }
fi

T0=$(date +%s)
on_exit() {  # always show where things stand, also after a failure or a timeout kill
  local code=$?
  if [ "${code}" != "0" ]; then
    echo; echo "=== FAILED/STOPPED (exit ${code}) after $(( $(date +%s) - T0 ))s - partial summary follows; logs: ${RESULTS}/logs ==="
    copy_train_results "${OUT_DIR}" "${RESULTS}/train" || true
    python scripts/summarize_results.py --results-dir "${RESULTS}" --title "GRIT ${RUN_ID} (partial)" || true
  fi
}
trap on_exit EXIT
trap 'exit 143' TERM INT  # a timeout kill still runs on_exit with a non-zero code
stage() { echo; echo "=== [$(( $(date +%s) - T0 ))s] $* ==="; }
want() { [[ ",${STAGES}," == *",$1,"* ]]; }

# Fixed 3B configuration (PLAN.md section 3); memory knobs come from the selected MEM_CONFIG.
TRAIN_ARGS=(
  --model-path "${MODEL}" --safety-model-path "${GUARD}"
  --task-file "${TASK_FILE}" --preserve-file "${PRESERVE_FILE}" --evaluation-file "${EVAL_FILE}"
  --projectors-path "${PROJECTORS}" --projector-relaxation 0.05
  --rollout-gpu "${ROLLOUT_GPU}" ${COLOCATED[@]+"${COLOCATED[@]}"}
  --rollout-dtype float32
  --task-batch-size 321 --preserve-batch-size 48
  --generations 5 --max-prompt-length 512 --max-response-length 256 --max-preserve-length 2304
  --task-optimizer adamw --lr 1e-6 --weight-decay 0 --adam-eps 1e-8
  --lambda-pres "${LAMBDA_PRES}" --epsilon-pres 0.05 --top-k 64 --base-statistics-dtype float32
  --use-curvature --fd-radius 0.05 --fd-check-interval 10
  --gradient-checkpointing --seed 66
)

mem_args() {  # mem_args TASK_MB:PRES_MB:REWARD_BS:ROLLOUT_MEM:ROLLOUT_SLEEP -> trainer flags
  local task pres reward mem sleep
  IFS=: read -r task pres reward mem sleep <<< "$1"
  MEM_ARGS=(--task-microbatch-size "${task}" --preservation-microbatch-size "${pres}"
            --reward-batch-size "${reward}" --rollout-memory-utilization "${mem}")
  [ "${sleep}" == "1" ] && MEM_ARGS+=(--rollout-sleep)
  return 0
}

latest_checkpoint() {  # newest complete checkpoint that can still be resumed
  local dir
  dir=$(ls -1d "$1"/step_* 2>/dev/null | sort | while read -r d; do
          [ -f "$d/complete.json" ] && [ -f "$d/training.pt" ] && echo "$d"; done | tail -1 || true)
  echo "${dir}"
}

copy_train_results() {  # small training outputs into the project folder
  local src=$1 dst=$2
  [ -d "${src}" ] || return 0
  mkdir -p "${dst}"
  for f in metrics.jsonl diagnostics_summary.json; do [ -f "${src}/${f}" ] && cp "${src}/${f}" "${dst}/"; done
  [ -d "${src}/validation" ] && cp -r "${src}/validation" "${dst}/"
  return 0
}

train() {  # train <output_dir> <max_steps> <save_steps> <validation_steps> <log>
  local out=$1 steps=$2 save=$3 validation=$4 log=$5 resume=() latest
  latest=$(latest_checkpoint "${out}")
  if [ -n "${latest}" ]; then
    if [ "$(basename "${latest}")" == "$(printf 'step_%06d' "${steps}")" ]; then
      echo "skip: ${out} already reached step ${steps}"; return 0
    fi
    echo "resume from ${latest}"; resume=(--resume "${latest}")
  elif [ -d "${out}" ]; then
    # Crashed before the first checkpoint: keep it for inspection, start clean.
    mv "${out}" "${out}.failed.$(date +%Y%m%d_%H%M%S)"
  fi
  env CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" torchrun --standalone --nproc_per_node "${NUM_TRAIN_GPUS}" scripts/train_grit.py "${TRAIN_ARGS[@]}" "${MEM_ARGS[@]}" \
      --output-dir "${out}" --max-steps "${steps}" --save-steps "${save}" --validation-steps "${validation}" \
      --prune-old-training-state ${resume[@]+"${resume[@]}"} 2>&1 | tee -a "${log}"
}

# ============================================================
# 1. Train (resumes from the latest checkpoint; checkpoints every 5 steps, validation every 5)
# ============================================================
if want train; then
  # Explicit MEM_CONFIG > the one an earlier job fell back to after an OOM > the most aggressive entry.
  MEM_CONFIG=${MEM_CONFIG:-$(cat "${RESULTS}/mem_config_in_use.txt" 2>/dev/null || echo "${MEM_CONFIGS%% *}")}
  # Only a CUDA out-of-memory error falls back to the next, less aggressive config (resuming from the
  # latest checkpoint; no partial step is ever committed). Any other error stops the job as is.
  fallbacks="${MEM_CONFIG}"; seen=0
  for config in ${MEM_CONFIGS}; do
    [ "${seen}" == "1" ] && fallbacks="${fallbacks} ${config}"
    [ "${config}" == "${MEM_CONFIG}" ] && seen=1
  done
  trained=0
  for config in ${fallbacks}; do
    mem_args "${config}"
    echo "${config}" > "${RESULTS}/mem_config_in_use.txt"
    stage "1. train ${RUN_ID}: lambda_pres=${LAMBDA_PRES}, ${MAX_STEPS} steps, MEM_CONFIG=${config}"
    log_start=$(( $(wc -l 2>/dev/null < "${RESULTS}/logs/train.log" || echo 0) + 1 ))
    if train "${OUT_DIR}" "${MAX_STEPS}" 5 5 "${RESULTS}/logs/train.log"; then trained=1; break; fi
    if ! tail -n "+${log_start}" "${RESULTS}/logs/train.log" | grep -qi "out of memory"; then
      copy_train_results "${OUT_DIR}" "${RESULTS}/train"
      echo "train FAILED (not an out-of-memory error) - see the traceback above / ${RESULTS}/logs/train.log"
      exit 1
    fi
    echo "train ran OUT OF MEMORY with MEM_CONFIG=${config}; falling back to the next config"
    echo "${config} OOM $(date)" >> "${RESULTS}/memory_config_failures.txt"
  done
  copy_train_results "${OUT_DIR}" "${RESULTS}/train"
  [ "${trained}" == "1" ] || { echo "training ran out of memory with every MEM_CONFIG - see ${RESULTS}/logs/train.log"; exit 1; }
fi

# ============================================================
# 2. Eval: base + selected checkpoints; safety (PKU-SafeRLHF 1,000, AdvBench, HarmBench, JailbreakBench,
#    SORRY-Bench, HarmfulQA, ALERT) and over-refusal (XSTest safe, OR-Bench hard-1k, AlpacaEval) judged by
#    WildGuard; general benchmarks scored by their official rules.
# ============================================================
if want eval; then
  stage "2. eval (base + steps: ${EVAL_STEPS})"
  copy_train_results "${OUT_DIR}" "${RESULTS}/train"
  MODELS=(--model "base=${MODEL}")
  for step in ${EVAL_STEPS}; do
    ckpt="${OUT_DIR}/$(printf 'step_%06d' "${step}")"
    if [ -f "${ckpt}/complete.json" ]; then MODELS+=(--model "step_$(printf '%06d' "${step}")=${ckpt}")
    else echo "skip (no complete checkpoint): ${ckpt}"; fi
  done
  python scripts/eval_run.py "${MODELS[@]}" --raw-root "${RAW}" --safety-file "${EVAL_FILE}" \
      --guard-path "${JUDGE}" --work-dir "${EVAL_WORK}" --results-dir "${RESULTS}/eval" \
      --gpus "${EVAL_GPUS}" --slots-per-gpu "${EVAL_SLOTS}" \
      ${EVAL_MAX_SAMPLES:+--max-samples "${EVAL_MAX_SAMPLES}"} 2>&1 | tee -a "${RESULTS}/logs/eval.log"
fi

# ============================================================
# 3. Summary: training curve + eval table + paired deltas -> console, summary.md, summary.json
# ============================================================
if want summary; then
  stage "3. summary"
  copy_train_results "${OUT_DIR}" "${RESULTS}/train"
  python scripts/summarize_results.py --results-dir "${RESULTS}" --title "GRIT ${RUN_ID}"
fi

echo "ALL DONE - $(( $(date +%s) - T0 ))s. Results: ${RESULTS}/summary.md"
