#!/usr/bin/env bash
# Train OPD + IMPACT on 8k samples, then evaluate its checkpoint.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export RUN_NAME=llama2-tinyllama-opd-8k
export OPD_PROFILE=llama2-tinyllama
export TRAIN_SAMPLES=8000
export TRAIN_SUBSET_SEED=42
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
export METHODS=opd_impact
export ONLY_METHOD=opd_impact
export SEEDS=10
export GRAD_ACCUM=32
export STEPS=0
export MAX_MINUTES=0
export DOLLY_EVAL=1
export MATH_EVAL=0
export EVAL_BENCHMARKS="dolly self-inst vicuna sinst"
export DOLLY_EVAL_SEEDS="10 20 30 40 50"
export TEACHER_FROM_RUN="${TEACHER_FROM_RUN:-llama2-tinyllama-opd}"
# Old overnight deadlines must not stop this full-budget run.
unset STOP_AT

bash project_commands.sh setup-eval
bash project_commands.sh download-eval
bash project_commands.sh fast
