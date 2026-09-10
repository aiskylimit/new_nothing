#!/usr/bin/env bash
# The only server-side entry point: offline setup -> train -> eval -> summary.
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT
# shellcheck source=env.sh
source "$PROJECT_ROOT/env.sh"
mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$RUNTIME_ROOT"
PROJECT_LOG="$LOG_DIR/project.log"
exec > >(tee -a "$PROJECT_LOG") 2>&1

log() { printf '[%s] [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S%z')" "$1" "${2:-}"; }
fail() { log FAILED "$1"; exit "${2:-1}"; }
trap 'code=$?; log FAILED "line=$LINENO command=$BASH_COMMAND exit_code=$code"; exit "$code"' ERR

lock_file="$RUNTIME_ROOT/project-command.lock"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$lock_file"
  flock -n 9 || fail "another project_command.sh is already running" 9
fi

IFS=',' read -r -a configured_gpus <<< "$GPU_IDS"
(( ${#configured_gpus[@]} == NUM_GPUS )) || fail \
  "GPU_IDS has ${#configured_gpus[@]} entries but NUM_GPUS=$NUM_GPUS" 10
(( EFFECTIVE_BATCH % (NUM_GPUS * TRAIN_BATCH_SIZE) == 0 )) || fail \
  "effective batch is not divisible by NUM_GPUS*TRAIN_BATCH_SIZE" 11

log ENV "mode=$PIPELINE_MODE target=$TARGET_GPU_FAMILY gpu_ids=$GPU_IDS num_gpus=$NUM_GPUS"
log ENV "offline=1 effective_batch=$EFFECTIVE_BATCH model=$MODEL_DIR data=$DATA_DIR"

log SETUP "preparing uv environment at $VENV_DIR"
if [[ "${SKIP_ENV_SETUP:-0}" != 1 ]]; then
  [[ -d "$WHEELHOUSE_DIR" ]] || fail "offline wheelhouse missing: $WHEELHOUSE_DIR; follow download.txt" 12
  UV_BIN="${UV_BIN:-$(command -v uv || true)}"
  [[ -n "$UV_BIN" ]] || UV_BIN="$ASSET_ROOT/bin/uv"
  if [[ -f "$UV_BIN" && ! -x "$UV_BIN" ]]; then
    chmod +x "$UV_BIN"
  fi
  [[ -n "$UV_BIN" && -x "$UV_BIN" ]] || fail "uv not found; copy the uv binary as described in download.txt" 13
  export UV_OFFLINE=1 UV_PYTHON_DOWNLOADS=never
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$UV_BIN" venv --python "${PYTHON_BIN:-python3.10}" "$VENV_DIR"
  fi
  requirements_hash="$(sha256sum "$PROJECT_ROOT/requirements-offline-b200.txt" | awk '{print $1}')"
  marker="$VENV_DIR/.requirements-$requirements_hash"
  if [[ ! -f "$marker" ]]; then
    clip_wheels=("$WHEELHOUSE_DIR"/clip-*.whl)
    [[ -e "${clip_wheels[0]}" ]] || fail "OpenAI CLIP wheel missing from $WHEELHOUSE_DIR" 14
    "$UV_BIN" pip install --offline --find-links "$WHEELHOUSE_DIR" \
      --python "$VENV_DIR/bin/python" -r "$PROJECT_ROOT/requirements-offline-b200.txt" \
      "${clip_wheels[0]}"
    touch "$marker"
  else
    log SETUP "dependency marker found; installation reused"
  fi
else
  [[ -x "$VENV_DIR/bin/python" ]] || fail "SKIP_ENV_SETUP=1 but venv is missing: $VENV_DIR" 15
  log SETUP "reusing caller-provided environment"
fi

# hpsv2 1.2.0 expects this vocabulary beside its package; supply it locally.
[[ -s "$RATIO_HPSV2_BPE" ]] || fail "HPSv2 BPE asset missing: $RATIO_HPSV2_BPE" 16
hps_pkg_dir="$($VENV_DIR/bin/python - <<'PY'
import pathlib
import hpsv2
print(pathlib.Path(hpsv2.__file__).resolve().parent / "src" / "open_clip")
PY
)"
mkdir -p "$hps_pkg_dir"
cp -f "$RATIO_HPSV2_BPE" "$hps_pkg_dir/bpe_simple_vocab_16e6.txt.gz"
log SETUP "environment ready"

log 'CHECK ASSETS' "validating local model, data, prompts, and reward weights"
asset_args=()
[[ "${SKIP_ENV_SETUP:-0}" == 1 ]] && asset_args+=(--skip-wheelhouse)
"$VENV_DIR/bin/python" "$PROJECT_ROOT/hessian/check_offline_assets.py" "${asset_args[@]}" \
  | tee "$LOG_DIR/assets.json"
log 'CHECK ASSETS' "all runtime assets are local and complete"

log ENV "GPU/BF16 preflight start"
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$VENV_DIR/bin/python" \
  "$PROJECT_ROOT/hessian/preflight_b300_sdxl.py" | tee "$LOG_DIR/preflight.json"
log ENV "GPU/BF16 preflight passed"

log SETUP "targeted unit tests start"
(cd "$PROJECT_ROOT" && "$VENV_DIR/bin/python" -m pytest -q \
  tests/test_ratio_diffusion.py \
  -k 'not mixed_precision_inputs_promote_to_finite_float32_with_gradients') \
  | tee "$LOG_DIR/unit-tests.log"
log SETUP "targeted unit tests passed"

RUN_NAME="${RUN_NAME:-q3_dspo_sdxl_${PIPELINE_MODE}_eb${EFFECTIVE_BATCH}_${NUM_GPUS}gpu}"
export RUN_NAME
log 'TRAIN START' "run=$RUN_NAME"
bash "$PROJECT_ROOT/hessian/run_offline_train.sh" "$PIPELINE_MODE"
RUN_DIR="$RUNS_DIR/$RUN_NAME"
[[ -f "$RUN_DIR/final-step.txt" ]] || fail "training ended without final-step.txt" 30
FINAL_STEP="$(<"$RUN_DIR/final-step.txt")"
export FINAL_STEP TRAIN_RUN="$RUN_NAME"
log 'TRAIN END' "run=$RUN_NAME final_step=$FINAL_STEP"

if [[ "$PIPELINE_MODE" == pilot && "$OFFLINE_EVAL_LIMIT" == 0 ]]; then
  export OFFLINE_EVAL_LIMIT=2
fi
EVAL_NAME="${EVAL_NAME:-${RUN_NAME}_eval}"
export EVAL_NAME EVAL_GPUS="${EVAL_GPUS:-$GPU_IDS}"
EVAL_DIR="$EVAL_ROOT_OVERRIDE/$EVAL_NAME"
log 'EVAL START' "eval=$EVAL_NAME gpus=$EVAL_GPUS limit=$OFFLINE_EVAL_LIMIT"
if [[ ! -f "$EVAL_DIR/report/report_manifest.json" ]]; then
  bash "$PROJECT_ROOT/hessian/run_offline_eval.sh" &
  eval_pid=$!
  while kill -0 "$eval_pid" 2>/dev/null; do
    log EVAL "still running; detailed logs: $EVAL_DIR/logs"
    sleep 30
  done
  wait "$eval_pid"
else
  log EVAL "existing completed report reused"
fi
log 'EVAL END' "report=$EVAL_DIR/report/REPORT.md"

log SUMMARY "building compact result"
"$VENV_DIR/bin/python" "$PROJECT_ROOT/hessian/make_offline_summary.py" \
  --run-dir "$RUN_DIR" --eval-dir "$EVAL_DIR" --output "$RESULTS_DIR/summary.json" \
  --mode "$PIPELINE_MODE" --checkpoint-step "$FINAL_STEP" --seed "$SEED" \
  --effective-batch "$EFFECTIVE_BATCH" --num-gpus "$NUM_GPUS"
log DONE "summary=$RESULTS_DIR/summary.json log=$PROJECT_LOG"
