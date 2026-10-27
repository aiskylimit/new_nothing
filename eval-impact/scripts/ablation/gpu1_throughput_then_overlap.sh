#!/bin/bash
# GPU 1 — Idea 3 (throughput, ~100 steps) then Idea 2 (top-1000 overlap, full train)
# Pair: Qwen2.5-7B-Instruct → GPT2-1.5B
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.3}"
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/bin" ]]; then
  export PATH="${CONDA_PREFIX}/bin:${CUDA_HOME}/bin:${PATH}"
else
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../common/setup_env.sh
source "${SCRIPT_DIR}/../common/setup_env.sh"

DSKD_ROOT="${BASE_PATH}/DSKDv2"
LOG_DIR="${BASE_PATH}/logs/ablation"
mkdir -p "${LOG_DIR}"
TP_SUMMARY="${LOG_DIR}/gpu1_throughput_summary.tsv"
BI_SUMMARY="${LOG_DIR}/gpu1_bi_cost.json"
OV_SUMMARY="${LOG_DIR}/gpu1_overlap_summary.tsv"
echo -e "method\tstart\tend\telapsed_sec\ttrain_only_sec\tpeak_gpu_mem_mib\tsec_per_step\th_per_epoch_est\tstatus\tsave_path" > "${TP_SUMMARY}"
echo -e "max_overlap\tstart\tend\tstatus\tsave_path" > "${OV_SUMMARY}"
BI_NUM_SAMPLES=256
STEPS_PER_EPOCH=1430
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"

CKPT_TYPE="gpt2"
CKPT_NAME="GPT2-1.5B"
CKPT_PATH="${BASE_PATH}/model_hub/${CKPT_NAME}"
TEACHER_MODEL_TYPE="qwen"
TEACHER_MODEL_NAME="Qwen2.5-7B-Instruct"
TEACHER_MODEL_PATH="${BASE_PATH}/model_hub/${TEACHER_MODEL_NAME}"
TEACHER_PEFT_PATH="${BASE_PATH}/lora_path/${TEACHER_MODEL_NAME}"
DATA_DIR="${BASE_PATH}/data/dolly/"
PROJECTOR_CONFIG_PATH="${BASE_PATH}/configs/projector_config.json"

BATCH_SIZE=4
LR=0.001
GRAD_ACC=2
EVAL_BATCH_SIZE=8
LORA_RANK=256
LORA_ALPHA=8
LORA_DROPOUT=0.1
MAX_LENGTH=512
PRECISION="bf16"
SEED=10
TOTAL_ITERS=100
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1
MASTER_ADDR=localhost

COMMON_PEFT=(
  --peft lora
  --peft-lora-r "${LORA_RANK}"
  --peft-lora-alpha "${LORA_ALPHA}"
  --peft-lora-dropout "${LORA_DROPOUT}"
)

COMMON_MODEL=(
  --model-type "${CKPT_TYPE}"
  --model-path "${CKPT_PATH}"
  --n-gpu "${GPUS_PER_NODE}"
  --teacher-model-type "${TEACHER_MODEL_TYPE}"
  --teacher-model-path "${TEACHER_MODEL_PATH}"
  --teacher-peft-path "${TEACHER_PEFT_PATH}"
  --teacher-model-fp16
  --gradient-checkpointing
)

COMMON_DATA=(
  --data-dir "${DATA_DIR}"
  --num-workers 0
  --dev-num 1000
)

COMMON_HP=(
  --lr "${LR}"
  --batch-size "${BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRAD_ACC}"
  --warmup-iters 0
  --lr-decay-style cosine
  --weight-decay 1e-2
  --clip-grad 1.0
  --total-iters "${TOTAL_ITERS}"
  --max-length "${MAX_LENGTH}"
  --max-prompt-length 256
  --do-train
  --save-interval 999
  --eval-interval 999
  --log-interval 10
  --keep-best-n-checkpoints 1
  --seed "${SEED}"
  --do-sample
  --top-k 0
  --top-p 1.0
  --temperature 1.0
)

wait_models() {
  echo "[GPU1] Waiting for model weights..."
  for p in "${CKPT_PATH}/config.json" "${TEACHER_MODEL_PATH}/config.json" "${TEACHER_PEFT_PATH}/adapter_config.json"; do
    while [[ ! -f "${p}" ]]; do
      echo "  missing ${p}; sleep 30s"
      sleep 30
    done
  done
  echo "[GPU1] Models ready."
}

start_gpu_mem_monitor() {
  local OUT_CSV="$1"
  local GPU="$2"
  : > "${OUT_CSV}"
  echo "ts,mem_used_mib" >> "${OUT_CSV}"
  # Must detach stdio, otherwise MEM_PID="$(...)" waits forever on the background job.
  (
    while true; do
      mem="$(nvidia-smi --id="${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
      echo "$(date +%s),${mem:-NA}" >> "${OUT_CSV}"
      sleep 1
    done
  ) >/dev/null 2>&1 &
  echo $!
}

run_bi_calibration() {
  echo "========== [GPU1][BI] one-time calibration START =========="
  export PYTHONPATH="${BASE_PATH}"
  set +e
  python "${SCRIPT_DIR}/bench_bi_cost.py" \
    --teacher-model-path "${TEACHER_MODEL_PATH}" \
    --teacher-peft-path "${TEACHER_PEFT_PATH}" \
    --data-dir "${DATA_DIR}" \
    --num-samples "${BI_NUM_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --max-length "${MAX_LENGTH}" \
    --dtype bf16 \
    --out "${BI_SUMMARY}"
  local RC=$?
  set -e
  if [[ ${RC} -eq 0 ]]; then
    echo "[GPU1][BI] wrote ${BI_SUMMARY}"
    # free GPU before throughput runs
    python - <<'PY'
import torch, gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
PY
  else
    echo "[GPU1][BI] FAILED rc=${RC}"
  fi
  echo "========== [GPU1][BI] one-time calibration END =========="
}

run_timed() {
  local NAME="$1"
  shift
  local SAVE_PATH="$1"
  shift
  local ENTRY="$1"
  shift
  local PYPATH="$1"
  shift

  mkdir -p "${SAVE_PATH}"
  local PORT="62$((RANDOM % 90 + 10))"
  local DIST="--nproc_per_node ${GPUS_PER_NODE} --nnodes ${NNODES} --node_rank ${NODE_RANK} --master_addr ${MASTER_ADDR} --master_port ${PORT}"
  local START END ELAPSED STATUS T0 T1 MEM_CSV MEM_PID PEAK_MEM TRAIN_ONLY SEC_PER H_EPOCH
  START="$(date -Iseconds)"
  MEM_CSV="${SAVE_PATH}/gpu_mem.csv"
  MEM_PID="$(start_gpu_mem_monitor "${MEM_CSV}" "${GPU_ID}")"
  T0="$(date +%s)"
  echo "========== [GPU1][throughput] ${NAME} START ${START} =========="

  export NCCL_DEBUG=""
  export WANDB_DISABLED=True
  export TF_CPP_MIN_LOG_LEVEL=3
  export PYTHONPATH="${PYPATH}"

  set +e
  torchrun ${DIST} "${ENTRY}" "$@" \
    --save-dir "${SAVE_PATH}" \
    2>&1 | tee "${SAVE_PATH}/train.log"
  local RC=${PIPESTATUS[0]}
  set -e

  T1="$(date +%s)"
  END="$(date -Iseconds)"
  ELAPSED=$((T1 - T0))
  kill "${MEM_PID}" 2>/dev/null || true
  wait "${MEM_PID}" 2>/dev/null || true

  PEAK_MEM="$(python - <<PY
import csv
vals=[]
try:
  with open("${MEM_CSV}") as f:
    r=csv.DictReader(f)
    for row in r:
      try: vals.append(float(row["mem_used_mib"]))
      except Exception: pass
except FileNotFoundError:
  pass
print(int(max(vals)) if vals else -1)
PY
)"

  # Prefer logged step_time sum if available (excludes model load); else wall clock.
  read -r TRAIN_ONLY SEC_PER H_EPOCH < <(python - <<PY
import re
elapsed=${ELAPSED}
steps=${TOTAL_ITERS}
spe=${STEPS_PER_EPOCH}
train_only = None
step_times=[]
try:
  text=open("${SAVE_PATH}/train.log", errors="ignore").read()
  # stop confirmation
  m=re.findall(r"global_step=(\d+)", text)
  last=int(m[-1]) if m else steps
  for m in re.finditer(r"step_time=([0-9.]+)", text):
    step_times.append(float(m.group(1)))
  if step_times:
    # log_interval averages; approximate train-only = mean(step_time)*last_step
    train_only = (sum(step_times)/len(step_times)) * max(last, 1)
except Exception:
  pass
if train_only is None or train_only <= 0:
  train_only = float(elapsed)
sec_per = train_only / max(steps, 1)
h_epoch = sec_per * spe / 3600.0
print(f"{train_only:.1f} {sec_per:.4f} {h_epoch:.4f}")
PY
)

  if [[ ${RC} -eq 0 ]]; then STATUS="ok"; else STATUS="fail_rc${RC}"; fi
  echo -e "${NAME}\t${START}\t${END}\t${ELAPSED}\t${TRAIN_ONLY}\t${PEAK_MEM}\t${SEC_PER}\t${H_EPOCH}\t${STATUS}\t${SAVE_PATH}" >> "${TP_SUMMARY}"
  echo "========== [GPU1][throughput] ${NAME} END ${END} elapsed=${ELAPSED}s train_only=${TRAIN_ONLY}s peak_mem=${PEAK_MEM}MiB status=${STATUS} =========="
  echo "[estimate] ${NAME}: ${SEC_PER}s/step → ~${H_EPOCH} h/epoch (assume ${STEPS_PER_EPOCH} steps/epoch)"
  return 0  # continue remaining methods even if one fails
}

# ---------- Idea 3: throughput ----------
run_throughput() {
  local ROOT_OUT="${BASE_PATH}/outputs/${CKPT_NAME}/throughput_bench"
  mkdir -p "${ROOT_OUT}"

  # 1) DSKD
  run_timed "DSKD" "${ROOT_OUT}/DSKD" \
    "${BASE_PATH}/code/distillation_baseline.py" "${BASE_PATH}" \
    --base-path "${BASE_PATH}" "${COMMON_MODEL[@]}" "${COMMON_DATA[@]}" "${COMMON_HP[@]}" "${COMMON_PEFT[@]}" \
    --task "throughput_DSKD" --criterion dual_space_kd_with_cma \
    --kd-rate 0.85 --kd-temperature 2.0 --kd-objective forward_kl \
    --projector-lr 0.001 --projector-config-path "${PROJECTOR_CONFIG_PATH}" \
    --precision "${PRECISION}" --deepspeed \
    --deepspeed_config "${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"

  # 2) DSKD + IMPACT
  run_timed "DSKD+IMPACT" "${ROOT_OUT}/DSKD_IMPACT" \
    "${BASE_PATH}/code/distillation_baseline.py" "${BASE_PATH}" \
    --base-path "${BASE_PATH}" "${COMMON_MODEL[@]}" "${COMMON_DATA[@]}" "${COMMON_HP[@]}" "${COMMON_PEFT[@]}" \
    --task "throughput_DSKD_IMPACT" --criterion DSKD_IMPACT \
    --kd-rate 0.85 --kd-temperature 2.0 --kd-objective forward_kl \
    --projector-lr 0.001 --projector-config-path "${PROJECTOR_CONFIG_PATH}" \
    --impact-lambda 0.25 --impact-top-k 4 --impact-bi-tau 1.0 --impact-lambda-reg 1.0 \
    --impact-bi-calib-offline --impact-bi-calib-samples 256 \
    --precision "${PRECISION}" --deepspeed \
    --deepspeed_config "${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"

  # 3) DSKDv2
  run_timed "DSKDv2" "${ROOT_OUT}/DSKDv2" \
    "${DSKD_ROOT}/code/distillation.py" "${DSKD_ROOT}" \
    --base-path "${DSKD_ROOT}" "${COMMON_MODEL[@]}" --model-dtype "${PRECISION}" \
    "${COMMON_DATA[@]}" "${COMMON_HP[@]}" "${COMMON_PEFT[@]}" \
    --task "throughput_dskd_v2" --criterion dual_space_kd_v2_with_eta \
    --kd-rate 0.8 --kd-temperature 2.0 --kd-objective forward_kl \
    --init-t2s-projector --init-s2t-projector --projector-lr 0.0005 --topk-vocab -1 \
    --deepspeed --deepspeed_config "${DSKD_ROOT}/configs/deepspeed/ds_config_bf16.json"

  # 4) DSKDv2 + IMPACT (main)
  run_timed "DSKDv2+IMPACT" "${ROOT_OUT}/DSKDv2_IMPACT" \
    "${DSKD_ROOT}/code/distillation.py" "${DSKD_ROOT}" \
    --base-path "${DSKD_ROOT}" "${COMMON_MODEL[@]}" --model-dtype "${PRECISION}" \
    "${COMMON_DATA[@]}" "${COMMON_HP[@]}" "${COMMON_PEFT[@]}" \
    --task "throughput_dskd_v2_impact" --criterion dual_space_kd_v2_impact \
    --kd-rate 0.8 --kd-temperature 2.0 --kd-objective forward_kl \
    --init-t2s-projector --init-s2t-projector --projector-lr 0.0005 --topk-vocab -1 \
    --impact-lambda 0.25 --impact-top-k 4 --impact-bi-tau 1.0 --impact-lambda-reg 1.0 \
    --impact-bi-calib-offline --impact-bi-calib-samples 256 \
    --deepspeed --deepspeed_config "${DSKD_ROOT}/configs/deepspeed/ds_config_bf16.json"

  # 5) ALM
  run_timed "ALM" "${ROOT_OUT}/ALM" \
    "${BASE_PATH}/code/distillation_alm.py" "${BASE_PATH}:${BASE_PATH}/ALM/tokenkit-main:${BASE_PATH}/ALM" \
    --base-path "${BASE_PATH}" "${COMMON_MODEL[@]}" "${COMMON_DATA[@]}" "${COMMON_HP[@]}" "${COMMON_PEFT[@]}" \
    --task "throughput_ALM" --criterion ALM --kd-temperature 2.0 --kd-objective forward_kl \
    --alm-binarization-temp 100.0 --alm-bias-threshold 0.1 --alm-loss-weight 3.0 \
    --alm-mode "merge_by_space_prob+append_space" \
    --multitask-aggregation-fn approx_gradmag_preserve_mag \
    --precision "${PRECISION}" --deepspeed \
    --deepspeed_config "${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"

  # 6) ALM + IMPACT
  run_timed "ALM+IMPACT" "${ROOT_OUT}/ALM_IMPACT" \
    "${BASE_PATH}/code/distillation_alm.py" "${BASE_PATH}:${BASE_PATH}/ALM/tokenkit-main:${BASE_PATH}/ALM" \
    --base-path "${BASE_PATH}" "${COMMON_MODEL[@]}" "${COMMON_DATA[@]}" "${COMMON_HP[@]}" "${COMMON_PEFT[@]}" \
    --task "throughput_ALM_IMPACT" --criterion ALM_IMPACT --kd-temperature 2.0 --kd-objective forward_kl \
    --alm-binarization-temp 100.0 --alm-bias-threshold 0.1 --alm-loss-weight 3.0 \
    --alm-mode "merge_by_space_prob+append_space" \
    --multitask-aggregation-fn approx_gradmag_preserve_mag \
    --impact-lambda 1.0 --impact-top-k 6 --impact-bi-tau 1.0 --impact-lambda-reg 1.0 \
    --impact-bi-calib-offline --impact-bi-calib-samples 256 \
    --precision "${PRECISION}" --deepspeed \
    --deepspeed_config "${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"

  # 7) SRA
  run_timed "SRA" "${ROOT_OUT}/SRA" \
    "${BASE_PATH}/code/distillation.py" "${BASE_PATH}" \
    --base-path "${BASE_PATH}" "${COMMON_MODEL[@]}" "${COMMON_DATA[@]}" "${COMMON_HP[@]}" "${COMMON_PEFT[@]}" \
    --task "throughput_SRA" --criterion SRA --kd-temperature 2.0 --kd-objective forward_kl \
    --sra-alpha 0.15 --sra-geom-weight 50.0 --sra-span-power 1.0 --sra-span-loss \
    --sra-student-layers -1 --sra-teacher-layers -1 --sra-hidden-loss-weights 1.0 \
    --projector-lr 0.0005 --projector-config-path "${PROJECTOR_CONFIG_PATH}" \
    --precision "${PRECISION}" --deepspeed \
    --deepspeed_config "${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"

  # 8) SRA + IMPACT
  run_timed "SRA+IMPACT" "${ROOT_OUT}/SRA_IMPACT" \
    "${BASE_PATH}/code/distillation.py" "${BASE_PATH}" \
    --base-path "${BASE_PATH}" "${COMMON_MODEL[@]}" "${COMMON_DATA[@]}" "${COMMON_HP[@]}" "${COMMON_PEFT[@]}" \
    --task "throughput_SRA_IMPACT" --criterion SRA_IMPACT --kd-temperature 2.0 --kd-objective forward_kl \
    --sra-alpha 0.15 --sra-geom-weight 50.0 --sra-span-power 1.0 --sra-span-loss \
    --sra-student-layers -1 --sra-teacher-layers -1 --sra-hidden-loss-weights 1.0 \
    --projector-lr 0.0005 --projector-config-path "${PROJECTOR_CONFIG_PATH}" \
    --impact-lambda 0.25 --impact-top-k 4 --impact-bi-tau 1.0 --impact-lambda-reg 1.0 \
    --impact-bi-calib-offline --impact-bi-calib-samples 256 \
    --precision "${PRECISION}" --deepspeed \
    --deepspeed_config "${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"

  # 9) RKD (random ProjectorTA init OK for timing)
  run_timed "RKD" "${ROOT_OUT}/ResidualKD" \
    "${BASE_PATH}/code/distillation_residualkd.py" "${BASE_PATH}" \
    --base-path "${BASE_PATH}" "${COMMON_MODEL[@]}" --model-dtype bf16 \
    "${COMMON_DATA[@]}" "${COMMON_HP[@]}" "${COMMON_PEFT[@]}" \
    --task "throughput_ResidualKD" --criterion ResidualKD \
    --residualkd-lambda-res 0.5 --residualkd-lambda-warmup 50 \
    --residualkd-d-bottleneck 64 --residualkd-cross-tokenizer \
    --projector-lr 0.001 \
    --deepspeed --deepspeed_config "${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"

  # 10) RKD + IMPACT
  run_timed "RKD+IMPACT" "${ROOT_OUT}/ResidualKD_IMPACT" \
    "${BASE_PATH}/code/distillation_residualkd.py" "${BASE_PATH}" \
    --base-path "${BASE_PATH}" "${COMMON_MODEL[@]}" --model-dtype bf16 \
    "${COMMON_DATA[@]}" "${COMMON_HP[@]}" "${COMMON_PEFT[@]}" \
    --task "throughput_ResidualKD_IMPACT" --criterion ResidualKD_IMPACT \
    --residualkd-lambda-res 0.5 --residualkd-lambda-warmup 50 \
    --residualkd-d-bottleneck 64 --residualkd-cross-tokenizer \
    --projector-lr 0.001 \
    --impact-lambda 1.0 --impact-top-k 4 --impact-bi-tau 1.0 --impact-lambda-reg 1.0 \
    --impact-bi-calib-offline --impact-bi-calib-samples 256 \
    --deepspeed --deepspeed_config "${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"

  # 11) E_SCOT
  run_timed "E_SCOT" "${ROOT_OUT}/E_SCOT" \
    "${BASE_PATH}/code/distillation.py" "${BASE_PATH}" \
    --base-path "${BASE_PATH}" "${COMMON_MODEL[@]}" "${COMMON_DATA[@]}" "${COMMON_HP[@]}" "${COMMON_PEFT[@]}" \
    --task "throughput_E_SCOT" --criterion E_SCOT \
    --kd-rate 0.5 --kd-temperature 2.0 --kd-objective forward_kl \
    --e-scot-lambda 0.25 --e-scot-alphas "0.5,1.0,2.0" \
    --precision "${PRECISION}" --deepspeed \
    --deepspeed_config "${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"

  # 12) E_SCOT + IMPACT
  run_timed "E_SCOT+IMPACT" "${ROOT_OUT}/E_SCOT_IMPACT" \
    "${BASE_PATH}/code/distillation.py" "${BASE_PATH}" \
    --base-path "${BASE_PATH}" "${COMMON_MODEL[@]}" "${COMMON_DATA[@]}" "${COMMON_HP[@]}" "${COMMON_PEFT[@]}" \
    --task "throughput_E_SCOT_IMPACT" --criterion E_SCOT_IMPACT \
    --kd-rate 0.5 --kd-temperature 2.0 --kd-objective forward_kl \
    --e-scot-lambda 0.25 --e-scot-alphas "0.5,1.0,2.0" \
    --impact-lambda 10.0 --impact-top-k 10 --impact-bi-tau 1.0 --impact-lambda-reg 1.0 \
    --impact-bi-calib-offline --impact-bi-calib-samples 256 \
    --precision "${PRECISION}" --deepspeed \
    --deepspeed_config "${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"

  echo "[GPU1] Throughput summary:"
  cat "${TP_SUMMARY}"
}

# ---------- Idea 2: top-1000 overlap full train ----------
run_overlap_top1000() {
  local MODE_OVERLAP=1000
  local EPOCH=15
  local TASK="dskd_v2_impact_overlap${MODE_OVERLAP}"
  local CRITERION="dual_space_kd_v2_impact"
  local CONFIG="forward_kl-lora-rank=${LORA_RANK}-alpha=${LORA_ALPHA}-dropout=${LORA_DROPOUT}-${PRECISION}"
  local SETTING="criterion=${CRITERION}__${CONFIG}__teacher=${TEACHER_MODEL_NAME}__kd^rate=0.8__kd^temp=2.0__impact^λ=0.25__impact^k=4__max_overlap=${MODE_OVERLAP}__epoch=${EPOCH}__bsz=${BATCH_SIZE}x${GRAD_ACC}x1=$((BATCH_SIZE * GRAD_ACC))__lr=${LR}__proj^lr=0.0005"
  local SAVE_PATH="${BASE_PATH}/outputs/${CKPT_NAME}/${TASK}/${SETTING}"
  mkdir -p "${SAVE_PATH}"

  local PORT="63$((RANDOM % 90 + 10))"
  local DIST="--nproc_per_node ${GPUS_PER_NODE} --nnodes ${NNODES} --node_rank ${NODE_RANK} --master_addr ${MASTER_ADDR} --master_port ${PORT}"
  local START END STATUS
  START="$(date -Iseconds)"
  echo "========== [GPU1][overlap] max_overlap=${MODE_OVERLAP} START ${START} =========="

  export PYTHONPATH="${DSKD_ROOT}"
  export NCCL_DEBUG=""
  export WANDB_DISABLED=True
  export TF_CPP_MIN_LOG_LEVEL=3

  set +e
  torchrun ${DIST} "${DSKD_ROOT}/code/distillation.py" \
    --base-path "${DSKD_ROOT}" \
    --model-type "${CKPT_TYPE}" --model-path "${CKPT_PATH}" --model-dtype "${PRECISION}" \
    --n-gpu "${GPUS_PER_NODE}" \
    --teacher-model-type "${TEACHER_MODEL_TYPE}" \
    --teacher-model-path "${TEACHER_MODEL_PATH}" \
    --teacher-peft-path "${TEACHER_PEFT_PATH}" \
    --teacher-model-fp16 --gradient-checkpointing \
    --data-dir "${DATA_DIR}" --num-workers 0 --dev-num 1000 \
    --task "${TASK}" \
    --lr "${LR}" --batch-size "${BATCH_SIZE}" --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRAD_ACC}" --warmup-iters 0 \
    --lr-decay-style cosine --weight-decay 1e-2 --clip-grad 1.0 \
    --num-epochs "${EPOCH}" \
    --kd-rate 0.8 --kd-temperature 2.0 --kd-objective forward_kl \
    --peft lora --peft-lora-r "${LORA_RANK}" --peft-lora-alpha "${LORA_ALPHA}" --peft-lora-dropout "${LORA_DROPOUT}" \
    --init-t2s-projector --init-s2t-projector --projector-lr 0.0005 --topk-vocab -1 \
    --impact-lambda 0.25 --impact-top-k 4 --impact-bi-tau 1.0 --impact-lambda-reg 1.0 \
    --impact-bi-calib-offline --impact-bi-calib-samples 256 \
    --impact-max-overlap "${MODE_OVERLAP}" \
    --max-length "${MAX_LENGTH}" --max-prompt-length 256 \
    --do-train --do-valid --eval-gen \
    --save-interval 1 --eval-interval 1 --log-interval 50 \
    --save-dir "${SAVE_PATH}" --keep-best-n-checkpoints 1 \
    --criterion "${CRITERION}" --seed "${SEED}" \
    --deepspeed --deepspeed_config "${DSKD_ROOT}/configs/deepspeed/ds_config_bf16.json" \
    --do-sample --top-k 0 --top-p 1.0 --temperature 1.0 \
    2>&1 | tee "${SAVE_PATH}/train.log"
  local RC=${PIPESTATUS[0]}
  set -e

  END="$(date -Iseconds)"
  if [[ ${RC} -eq 0 ]]; then STATUS="ok"; else STATUS="fail_rc${RC}"; fi
  echo -e "${MODE_OVERLAP}\t${START}\t${END}\t${STATUS}\t${SAVE_PATH}" >> "${OV_SUMMARY}"
  echo "========== [GPU1][overlap] END ${END} status=${STATUS} =========="
  cat "${OV_SUMMARY}"
}

wait_models
run_bi_calibration
run_throughput
run_overlap_top1000
echo "[GPU1] Done. Throughput: ${TP_SUMMARY} | BI: ${BI_SUMMARY} | Overlap: ${OV_SUMMARY}"
