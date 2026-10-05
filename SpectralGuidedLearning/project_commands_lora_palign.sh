#!/usr/bin/env bash
# LoRA SFT on P-ALIGN's data for P-ALIGN's own students, then eval (3 sampling seeds) and compare.
#   bash project_commands_lora_palign.sh <model>      model: qwen25-7b | qwen3-8b | r1-qwen-7b
#     qwen25-7b  Qwen/Qwen2.5-7B-Instruct  default ARMS="iwc gain"
#     qwen3-8b   Qwen/Qwen3-8B             default ARMS="iwc gain"
#     r1-qwen-7b DeepSeek-R1-Distill-Qwen-7B (the first, mistaken 7B run; kept reproducible)
# Arms (override with ARMS="iwc gain nll"):
#   iwc  -> iwc-nogate-l05-lora : IWC-Stable, no gate (p=1.0), lambda 0.5, tau 2, clip 2, weights from the student's per-step ENTROPY
#   gain -> iwc-gain-l05-lora   : same formula, weights from per-step ANSWER-INFORMATION GAIN of the student
#   nll  -> sft-nll-lora        : plain NLL = P-ALIGN through this pipeline
# Same data, format and recipe as the 1.5B track (project_commands_iwc_palign_r1-qwen-1.5b.sh): 966 P-ALIGN
# samples, thinking OFF (Qwen2.5: no think block; Qwen3: empty <think></think>, exactly P-ALIGN's
# enable_thinking=false), 3 epochs, eff. batch 32, lr 5e-5 -> 1e-5 cosine, DeepSpeed ZeRO-2, except LoRA
# (r16 / alpha16 / dropout 0.05 / all linear, as P-ALIGN's configs; adapter-only checkpoints).
# Signals (entropy, answer gain) come from the student being trained, so they are recomputed per model.
# Seed protocol: train once (seed 42), evaluate with 3 vLLM sampling seeds (42/43/44); seed 42 goes to
# results/, 43/44 to results_evalseed/<arm>-<track>-e<seed>.
#   GPUS=0 bash project_commands_lora_palign.sh qwen25-7b
#   GPUS=0 bash project_commands_lora_palign.sh qwen3-8b
#   ARMS="gain" ...   # subset;  LR=1e-4 ...  # lora lr (default 5e-5);  FORCE=1 ...  # redo existing outputs
# Phases are resumable (existing outputs are reused).
set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE}"

CUDA_GPUS="${CUDA_VISIBLE_DEVICES:-}"
export GPUS="${GPUS:-${CUDA_GPUS:+${CUDA_GPUS//,/ }}}"
export GPUS="${GPUS:-0}"
read -ra GPU_LIST <<< "${GPUS}"

MODEL_KEY="${1:?usage: $0 <qwen25-7b|qwen3-8b|r1-qwen-7b>}"
case "${MODEL_KEY}" in
  qwen25-7b)  MODEL_NAME="Qwen/Qwen2.5-7B-Instruct";                 DEFAULT_ARMS="iwc gain" ;;
  qwen3-8b)   MODEL_NAME="Qwen/Qwen3-8B";                            DEFAULT_ARMS="iwc gain" ;;
  r1-qwen-7b) MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B";  DEFAULT_ARMS="iwc gain" ;;
  *) echo "unknown model '${MODEL_KEY}' (qwen25-7b | qwen3-8b | r1-qwen-7b)" >&2; exit 2 ;;
esac
ARMS="${ARMS:-${DEFAULT_ARMS}}"
# Weights dtype for training: bfloat16 weights, with fp32 master weights for the trainable (LoRA) parameters kept by DeepSpeed
# ZeRO-2 -- the same precision scheme as the 1.5B full-FT track. MODEL_DTYPE=float32 (fp32 weights + bf16 autocast, no
# DeepSpeed) is available but off by default; its outputs get a "-fp32" track suffix so the two never mix.
DEFAULT_DTYPE=bfloat16
export MODEL_DTYPE="${MODEL_DTYPE:-${DEFAULT_DTYPE}}"
FORCE="${FORCE:-0}"

export TRACK="${MODEL_KEY}-palign"
[[ "${MODEL_DTYPE}" == float32 ]] && TRACK="${TRACK}-fp32"
export TRACK
export MODEL_NAME
export BASE_MODEL="${MODEL_NAME}"
export DATASET_NAME="${DATASET_NAME:-${BASE}/references/P-ALIGN/data/palign_sft_qwen2.5-7b.json.gz}"
# Thinking OFF, as P-ALIGN trains and as eval runs (prompt ends with an empty <think>\n\n</think>\n\n).
export ENABLE_THINKING=false
export PALIGN_PROMPT=true

# IWC-Stable, no gate: the settings of the best 1.5B variants.
export IWC_ENERGY_THRESHOLD_P=1.0
export IWC_INTERPOLATION="${IWC_INTERPOLATION:-0.5}"
export IWC_TEMPERATURE="${IWC_TEMPERATURE:-2.0}"
export IWC_CLIP="${IWC_CLIP:-2.0}"

export PROJECT_ENV="${PROJECT_ENV:-$(cd "${BASE}/.." && pwd)/iwc}"
source "${PROJECT_ENV}/bin/activate"
export PYTHONPATH="${BASE}/src"
mkdir -p logs

DATA_DIR="data/${TRACK}"
# Reuse the tokenized data and the signals (entropy / answer gain, computed by the frozen student) of the bf16 track.
BF16_DIR="data/${MODEL_KEY}-palign"
if [[ "${TRACK}" != "${MODEL_KEY}-palign" && -d "${BF16_DIR}" && ! -d "${DATA_DIR}" ]]; then cp -r "${BF16_DIR}" "${DATA_DIR}"; fi
SEGMENTED="${DATA_DIR}/train-segmented.jsonl"
has_arm() { [[ " ${ARMS} " == *" $1 "* ]]; }

# Build one weighted dataset: <signal parquet> -> train-<out-variant>.jsonl (p=1.0, lambda/tau/clip above).
# Runs in a scratch dir so build_iwc_datasets.py cannot overwrite another arm's train-iwc-stable.jsonl.
build_weighted() {   # $1 = signal parquet, $2 = out variant name
  local build="${DATA_DIR}-build-$2"
  rm -rf "${build}"; mkdir -p "${build}"
  ln -s "${BASE}/${SEGMENTED}" "${build}/train-segmented.jsonl"
  python -m sgl.allocation.build --data-path "${build}/train-segmented.jsonl" --strengths "$1" \
    --energy-threshold-p "${IWC_ENERGY_THRESHOLD_P}" --temperature "${IWC_TEMPERATURE}" \
    --interpolation "${IWC_INTERPOLATION}" --clip "${IWC_CLIP}" --variants iwc-stable \
    2>&1 | tee "logs/${TRACK}-$2-masks.log"
  mv "${build}/train-iwc-stable.jsonl" "${DATA_DIR}/train-$2.jsonl"
  mv "${build}/iwc-selection-stats.json" "${DATA_DIR}/$2-selection-stats.json"
  rm -rf "${build}"
  # IWC-Stable must preserve token mass: the log has to say mean weighted/selected mass=1.000000.
  grep -q "mass=1.000000" "logs/${TRACK}-$2-masks.log" || { echo "$2: token mass not preserved" >&2; exit 1; }
}

# ============================ DATA =============================
# P-ALIGN traces tokenized with the student's tokenizer + the all-ones train-vanilla.jsonl (NLL baseline).
[[ -f "${SEGMENTED}" ]] || bash scripts/data/data_r1-qwen-1.5b.sh

# ===================== SIGNALS + WEIGHTS =======================
if has_arm iwc; then
  # Spectral capture is the only producer of step entropies (its SVD strengths are ignored at p=1.0).
  [[ -f "${DATA_DIR}/spectral-strengths.parquet" ]] || bash scripts/capture/capture_r1-qwen-1.5b.sh
  [[ -f "${DATA_DIR}/train-iwc-stable-nogate-l05.jsonl" ]] \
    || build_weighted "${DATA_DIR}/spectral-strengths.parquet" iwc-stable-nogate-l05
fi
if has_arm gain; then
  GAINS="${DATA_DIR}/signals/step_answer_gain.json"
  SIGNAL="${DATA_DIR}/signals/signal-answer-gain.parquet"
  if [[ ! -f "${GAINS}" ]]; then   # log p(gold answer | prefix up to step k) under the frozen student, via vLLM
    CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" python -m sgl.signals.answer_gain --data-path "${SEGMENTED}" \
      --model "${MODEL_NAME}" --gpu-memory-utilization "${GAIN_GPU_MEM_UTIL:-0.4}" --output "${GAINS}" 2>&1 | tee "logs/${TRACK}-answer-gain.log"
  fi
  [[ -f "${SIGNAL}" ]] || python -m sgl.signals.gain_parquet --data-path "${SEGMENTED}" --gains "${GAINS}" \
    --output "${SIGNAL}" $([[ -f "${DATA_DIR}/spectral-strengths.parquet" ]] && echo "--strengths ${DATA_DIR}/spectral-strengths.parquet")
  [[ -f "${DATA_DIR}/train-iwc-stable-gain-nogate-l05.jsonl" ]] \
    || build_weighted "${SIGNAL}" iwc-stable-gain-nogate-l05
fi

# ====================== TRAIN + EVAL PER ARM ===================
# arm key -> "<arm name>|<data variant>|<extra train_sft.py args>"
declare -A ARM_SPEC=(
  [iwc]="iwc-nogate-l05-lora|iwc-stable-nogate-l05|"
  [gain]="iwc-gain-l05-lora|iwc-stable-gain-nogate-l05|"
  [nll]="sft-nll-lora|vanilla|--objective nll"
)
for key in ${ARMS}; do
  [[ -n "${ARM_SPEC[${key}]:-}" ]] || { echo "unknown arm '${key}' (iwc gain nll)" >&2; exit 2; }
  IFS='|' read -r ARM VARIANT EXTRA <<< "${ARM_SPEC[${key}]}"
  CKPT="${BASE}/checkpoints/${ARM}-${TRACK}"
  echo "===================== ARM ${ARM} (${VARIANT}) ====================="

  if [[ "${FORCE}" != 1 && -f "${CKPT}/adapter_config.json" ]]; then
    echo "adapter exists at ${CKPT} -- skipping training (FORCE=1 to retrain)"
  else
    # shellcheck disable=SC2086
    bash scripts/iwc/train_lora.sh "${ARM}" "${VARIANT}" ${EXTRA} \
      || { echo "ARM ${ARM}: TRAINING FAILED" >&2; continue; }
  fi

  # Fresh shell env so the eval script picks the vLLM env itself; sampling seed 42 -> results/, 43/44 -> results_evalseed/.
  for eseed in 42 43 44; do
    if [[ "${eseed}" == 42 ]]; then TAG="${ARM}-${TRACK}"; RDIR="${BASE}/results"; else TAG="${ARM}-${TRACK}-e${eseed}"; RDIR="${BASE}/results_evalseed"; fi
    if [[ "${FORCE}" != 1 && -f "${RDIR}/${TAG}/summary.json" ]]; then echo "eval ${TAG} exists -- skipping"; continue; fi
    ( unset VIRTUAL_ENV; EVAL_SEED="${eseed}" RESULTS_DIR="${RDIR}" \
        bash scripts/eval/eval_lora_palign.sh "${CKPT}" "${TAG}" ) \
      || echo "ARM ${ARM}: EVAL (sampling seed ${eseed}) FAILED" >&2
  done
done

# =========================== COMPARE ===========================
# results/comparison-table.md + results/eval-summary.json (sampling seeds 43/44 are in results_evalseed/)
"${PROJECT_ENV}/bin/python" -m sgl.eval.compare
