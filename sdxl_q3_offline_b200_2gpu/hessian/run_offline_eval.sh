#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${TRAIN_RUN:?TRAIN_RUN is required}"
: "${FINAL_STEP:?FINAL_STEP is required}"
PYTHON="$VENV_DIR/bin/python"
DRIVER="$PROJECT_ROOT/hessian/evaluate_standalone.py"
EVAL_NAME="${EVAL_NAME:-${TRAIN_RUN}_eval}"
EVAL_DIR="$EVAL_ROOT_OVERRIDE/$EVAL_NAME"
EVAL_GPUS="${EVAL_GPUS:-$GPU_IDS}"
IFS=',' read -r -a gpu_array <<< "$EVAL_GPUS"
mkdir -p "$EVAL_DIR/logs"

common=(--project-root "$PROJECT_ROOT" --work-root "$RUNTIME_ROOT" --eval-name "$EVAL_NAME" \
  --train-run-name "$TRAIN_RUN" --checkpoints "$FINAL_STEP" --pilot-count "$PILOT_PROMPT_COUNT")

if [[ ! -f "$RUNS_DIR/$TRAIN_RUN/exit-code.txt" ]] \
   || [[ "$(<"$RUNS_DIR/$TRAIN_RUN/exit-code.txt")" != 0 ]]; then
  echo "Training is not complete: $RUNS_DIR/$TRAIN_RUN" >&2
  exit 20
fi

"$PYTHON" "$DRIVER" prepare "${common[@]}" > "$EVAL_DIR/logs/prepare-prompts.log" 2>&1

generation_complete=1
for rank in "${!gpu_array[@]}"; do
  [[ -s "$EVAL_DIR/manifests/full_generate_rank${rank}.json" ]] || generation_complete=0
done
if (( generation_complete )); then
  echo "Reusing completed generation manifests"
else
  pids=()
  for rank in "${!gpu_array[@]}"; do
    gpu="${gpu_array[$rank]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$DRIVER" generate-worker "${common[@]}" \
      --scope full --selected-checkpoint "$FINAL_STEP" --rank "$rank" \
      --world-size "${#gpu_array[@]}" > "$EVAL_DIR/logs/generate-gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
fi

metrics=(pickscore hpsv2 aesthetics_clip imagereward)
for ((start=0; start<${#metrics[@]}; start+=${#gpu_array[@]})); do
  pids=()
  for slot in "${!gpu_array[@]}"; do
    index=$((start + slot)); (( index < ${#metrics[@]} )) || continue
    metric="${metrics[$index]}"; gpu="${gpu_array[$slot]}"
    if [[ -s "$EVAL_DIR/scores/full_${metric}.json" ]]; then
      echo "Reusing completed metric: $metric"
      continue
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$DRIVER" score-metric "${common[@]}" \
      --scope full --selected-checkpoint "$FINAL_STEP" --metric "$metric" \
      > "$EVAL_DIR/logs/score-${metric}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done

"$PYTHON" "$DRIVER" merge "${common[@]}" --scope full --selected-checkpoint "$FINAL_STEP" \
  > "$EVAL_DIR/logs/merge.log" 2>&1
"$PYTHON" "$DRIVER" report "${common[@]}" --selected-checkpoint "$FINAL_STEP" \
  --selection-policy final > "$EVAL_DIR/logs/report.log" 2>&1
