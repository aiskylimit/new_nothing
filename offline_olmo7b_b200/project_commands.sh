#!/bin/bash
# RLSD and SDPO on OLMo-3-7B-Think: trained from scratch on this B200 server, then evaluated.
# Order: dry run -> train RLSD -> eval RLSD -> train SDPO -> eval SDPO -> aggregate.
# Training: GPU 0 (main) + GPU 1 (vLLM rollout replica). Eval: GPUs 0,1 (vLLM replicas).
# Outputs go to *_retrain directories so earlier results are kept.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
export PYTHONPATH=.

source /mnt/local/uvenvs/tropic/bin/activate

PROJECT_NAME="aiskylimit_new_nothing"
BASE_DIR="/mnt/local/${PROJECT_NAME}/tropic_baselines"
MODEL_OLMO="${BASE_DIR}/models/Olmo-3-7B-Think"
export TROPIC_TRAIN_DATA_PATH="${BASE_DIR}/data/train"
export TROPIC_EVAL_DATA_DIR="${BASE_DIR}/data/eval"

MAIN_GPU=0
VLLM_GPU_IDS="1"
EVAL_VLLM_GPU_IDS="0,1"
VLLM_BASE_PORT=8100
VLLM_EXECUTABLE="vllm"
GPU_MEM_UTIL=0.95
VLLM_MAX_MODEL_LEN=41000
CHECKPOINTS="25 50 75 100"
BENCHMARKS="aime25 aime26 hmmt25"

train() {
  local script=$1 output_dir=$2
  echo "=== training: $script -> $output_dir ==="
  env CUDA_VISIBLE_DEVICES=${MAIN_GPU} python "scripts/${script}" \
      --model-path "${MODEL_OLMO}" --seed 0 --output-dir "${output_dir}" --skip-eval \
      --use-vllm-rollout --vllm-gpu-ids "${VLLM_GPU_IDS}" --vllm-base-port ${VLLM_BASE_PORT} \
      --vllm-executable "${VLLM_EXECUTABLE}" --vllm-gpu-memory-utilization ${GPU_MEM_UTIL} \
      --vllm-max-model-len ${VLLM_MAX_MODEL_LEN} \
      2>&1 | tee "${output_dir}_train.log"
}

train_dry_run() {
  local script=$1 output_dir=$2
  echo "=== DRY RUN (3 steps, sanity check only): $script ==="
  env CUDA_VISIBLE_DEVICES=${MAIN_GPU} python "scripts/${script}" \
      --model-path "${MODEL_OLMO}" --seed 0 --output-dir "${output_dir}_dryrun" --skip-eval \
      --training-steps 3 \
      --use-vllm-rollout --vllm-gpu-ids "${VLLM_GPU_IDS}" --vllm-base-port ${VLLM_BASE_PORT} \
      --vllm-executable "${VLLM_EXECUTABLE}" --vllm-gpu-memory-utilization ${GPU_MEM_UTIL} \
      --vllm-max-model-len ${VLLM_MAX_MODEL_LEN} \
      2>&1 | tee "${output_dir}_dryrun_train.log"
}

eval_checkpoint() {
  local script=$1 tag=$2 output_dir=$3 step=$4
  local ckpt="${ROOT}/${output_dir}/${tag}_checkpoint_step${step}"
  local log="${ROOT}/${output_dir}_eval_step${step}.log"
  if [ -f "$log" ] && grep -q "ALL DONE" "$log"; then echo "skip (already evaluated): $log"; return 0; fi
  if [ ! -d "$ckpt" ]; then echo "skip (no checkpoint): $ckpt"; return 0; fi
  echo "=== eval: $script step ${step} ==="
  env CUDA_VISIBLE_DEVICES=${MAIN_GPU} python "scripts/${script}" \
      --model-path "${MODEL_OLMO}" --skip-train --checkpoint-path "$ckpt" \
      --output-dir "${ROOT}/${output_dir}" \
      --eval-engine vllm --vllm-gpu-ids "${EVAL_VLLM_GPU_IDS}" --vllm-base-port ${VLLM_BASE_PORT} \
      --vllm-executable "${VLLM_EXECUTABLE}" --vllm-gpu-memory-utilization ${GPU_MEM_UTIL} \
      --vllm-max-model-len ${VLLM_MAX_MODEL_LEN} \
      --eval-benchmarks ${BENCHMARKS} \
      2>&1 | tee "$log"
}

train_dry_run run_rlsd_experiment_olmo7b.py results_rlsd_olmo7b_retrain

train run_rlsd_experiment_olmo7b.py results_rlsd_olmo7b_retrain
for step in $CHECKPOINTS; do eval_checkpoint run_rlsd_experiment_olmo7b.py rlsd results_rlsd_olmo7b_retrain "$step"; done

train run_sdpo_experiment_olmo7b.py results_sdpo_olmo7b_retrain
for step in $CHECKPOINTS; do eval_checkpoint run_sdpo_experiment_olmo7b.py sdpo results_sdpo_olmo7b_retrain "$step"; done

python - <<'PYEOF'
import glob, json, re

pattern = re.compile(
    r"\[(?P<tag>[A-Za-z]+)_step(?P<step>\d+)[^\]]*\]\s+(?P<bench>aime25|aime26|hmmt25)\s+"
    r"avg@12=(?P<avg>[\d.]+)\s+pass@12=(?P<pass_>[\d.]+)"
)
rows = []
for path in sorted(glob.glob("results_*_retrain_eval_step*.log")):
    text = open(path, encoding="utf-8", errors="replace").read()
    for m in pattern.finditer(text):
        rows.append({
            "file": path, "tag": m["tag"], "step": int(m["step"]), "benchmark": m["bench"],
            "avg@12": float(m["avg"]), "pass@12": float(m["pass_"]),
        })

with open("final_results.json", "w") as f:
    json.dump(rows, f, indent=2)

print(f"{'tag':<8} {'step':>4}  {'benchmark':<8} {'avg@12':>7} {'pass@12':>8}")
for r in rows:
    print(f"{r['tag']:<8} {r['step']:>4}  {r['benchmark']:<8} {r['avg@12']:>7.3f} {r['pass@12']:>8.3f}")
print(f"\n{len(rows)} rows written to final_results.json")
PYEOF

echo "ALL DONE - see final_results.json for the aggregated table."
