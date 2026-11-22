#!/bin/bash
# SELOT (selective OT activation steering): the stages of ONE run (one RUN_ID) on one or more GPUs.
# (project_commands.sh is the new_nothing server job; it calls this script once per run through scripts/pipeline.py.)
# Order: label (this model's D_b / D_r / D_j from the pools) -> extract activations -> fit the affine map per layer ->
# generate (alpha sweep) -> judge -> analyze -> summary.
# Every stage resumes: cached activations are reused, generation skips finished (alpha, prompt) pairs.
# Settings are env vars with the defaults below (GPU_IDS, MODEL, METHOD, DATA_CONFIG, RUN_ID, STAGES, JUDGE,
# JUDGE_EXTRA, STRONGREJECT, REWARD, OVERREFUSAL, FIT_BEHAVIOUR, FIT_WORKFLOW, EXTRA_SETS, and the paths), e.g.
#   MODEL=qwen2.5-7b STAGES=generate,judge bash run_stages.sh
# GPU_IDS="2,3": every vLLM step (label responses, Llama Guard, HarmBench, the over-refusal judge) splits its prompts
# over these GPUs; HF steps (activations, steered generation, StrongREJECT, reward model) use the first one.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
export PYTHONPATH=.

# VENV: a virtualenv to activate; empty = the current python.
VENV=${VENV:-}
if [ -n "${VENV}" ] && [ "${VENV}" != "none" ]; then
  [ -f "${VENV}/bin/activate" ] || { echo "ERROR: ${VENV}/bin/activate not found" >&2; exit 1; }
  source "${VENV}/bin/activate"
fi

# Paths. MODELS_DIR holds one folder per model (see README "Models"); WORK holds the large files.
BASE_DIR=${BASE_DIR:-${ROOT}}
MODELS_DIR=${MODELS_DIR:-${BASE_DIR}/models}
export HF_HOME=${HF_HOME:-${BASE_DIR}/.cache/huggingface}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
# no outbound calls: no usage stats / telemetry; vLLM's 1-process torch.distributed group stays on loopback
export HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_HOST_IP=127.0.0.1

# Runs on DATA_CONFIG. Every judge is a local model (no API): Llama Guard 3, HarmBench, StrongREJECT, and the
# Skywork-Reward-V2 reward model for response quality on AlpacaEval.
GPU_IDS=${GPU_IDS:-${GPU_ID:-0}}           # comma-separated, e.g. 2,3
MODEL=${MODEL:-llama3.1-8b}            # configs/models/<MODEL>.yaml
METHOD=${METHOD:-selot}                # configs/methods/<METHOD>.yaml
DATA_CONFIG=${DATA_CONFIG:-configs/data/v1.yaml}   # data.fit + data.eval (built by scripts/build_fit_sets.py, prepare_eval.py)
JUDGE=${JUDGE:-llamaguard}             # main ASR judge (local): llamaguard (AlphaSteer's Llama Guard protocol) | harmbench | keyword
JUDGE_EXTRA=${JUDGE_EXTRA:-[harmbench]}  # also reported, as asr_<name>
STRONGREJECT=${STRONGREJECT:-1}        # 1 = StrongREJECT score on the asr sets
REWARD=${REWARD:-1}                    # 1 = reward model (win rate vs alpha 0 / text-davinci-003) on the alpaca_eval sets
OVERREFUSAL=${OVERREFUSAL:-1}          # 1 = XSTest 3-class judge (Qwen2.5-14B-Instruct) on the over-refusal sets
# Fit sets: llamaguard = keep what the unsteered model refuses (D_r) / answers (D_j, D_b), judged by Llama Guard;
# none = every pooled prompt (ablation); keyword = refusal prefix instead of Llama Guard.
FIT_BEHAVIOUR=${FIT_BEHAVIOUR:-llamaguard}
# Fit-set workflow (selot/fit_data.py): 2 = fixed train/val split by behaviour groups shared by the three pools;
# 1 = the first Colab run (per-pool near-duplicate groups, per-model split after the topic cap).
FIT_WORKFLOW=${FIT_WORKFLOW:-2}
[[ "${FIT_WORKFLOW}" == "1" || "${FIT_WORKFLOW}" == "2" ]] || { echo "ERROR: FIT_WORKFLOW must be 1 or 2" >&2; exit 1; }
STAGES=${STAGES:-label,extract,fit,generate,judge,analyze,summary}
EXTRA_SETS=${EXTRA_SETS:-}             # space-separated config overrides, e.g. "method.beta=1000 steer.alphas=[0,1]"
MIN_FREE_MB=${MIN_FREE_MB:-40000}

RUN_ID=${RUN_ID:-${MODEL}_${METHOD}_wf${FIT_WORKFLOW}$([ "${FIT_BEHAVIOUR}" != "llamaguard" ] && echo "_${FIT_BEHAVIOUR}" || true)}
[ -n "${DATA_CONFIG}" ] || { echo "ERROR: DATA_CONFIG is empty (a YAML with data.fit and data.eval paths)"; exit 1; }
CONFIGS=(--config configs/base.yaml --config "configs/models/${MODEL}.yaml" --config "configs/methods/${METHOD}.yaml"
         --config "${DATA_CONFIG}")
SETS=()
WORK=${WORK:-${BASE_DIR}/work}           # acts/, maps/, generations/ (large)
MODEL_DIRNAME=$(sed -n 's/^ *path: *//p' "configs/models/${MODEL}.yaml" | head -1 | xargs basename)
RESULTS="${RESULTS_ROOT:-${ROOT}}/results_${RUN_ID}"   # small files only: logs, fit report, metrics, analysis, summary
SETS+=(--set "run_id=${RUN_ID}" --set "work_dir=${WORK}" --set "results_dir=${RESULTS}" --set device=cuda
       --set "judge.name=${JUDGE}" --set "judge.extra=${JUDGE_EXTRA}" --set "judge.path=${MODELS_DIR}/HarmBench-Llama-2-13b-cls"
       --set "judge.llamaguard.path=${MODELS_DIR}/Llama-Guard-3-8B"
       --set "judge.strongreject.enabled=$([ "${STRONGREJECT}" == "1" ] && echo true || echo false)"
       --set "judge.strongreject.base_path=${MODELS_DIR}/gemma-2b"
       --set "judge.strongreject.adapter_path=${MODELS_DIR}/strongreject-15k-v1"
       --set "judge.reward.enabled=$([ "${REWARD}" == "1" ] && echo true || echo false)"
       --set "judge.reward.path=${MODELS_DIR}/Skywork-Reward-V2-Llama-3.1-8B"
       --set "judge.overrefusal.enabled=$([ "${OVERREFUSAL}" == "1" ] && echo true || echo false)"
       --set "judge.overrefusal.path=${MODELS_DIR}/Qwen2.5-14B-Instruct"
       --set "fit_data.behaviour=${FIT_BEHAVIOUR}" --set "fit_data.workflow=${FIT_WORKFLOW}" --set "fit_data.guard.path=${MODELS_DIR}/Llama-Guard-3-8B"
       --set "model.path=${MODELS_DIR}/${MODEL_DIRNAME}")
for kv in ${EXTRA_SETS}; do SETS+=(--set "${kv}"); done
ARGS=("${CONFIGS[@]}" "${SETS[@]}")

mkdir -p "${RESULTS}/logs" "${WORK}"
exec > >(tee -a "${RESULTS}/logs/full_run_$(date +%Y%m%d_%H%M%S).log") 2>&1
echo "RUN_ID=${RUN_ID} MODEL=${MODEL} METHOD=${METHOD} FIT_BEHAVIOUR=${FIT_BEHAVIOUR} FIT_WORKFLOW=${FIT_WORKFLOW} GPU_IDS=${GPU_IDS} STAGES=${STAGES}"

for g in ${GPU_IDS//,/ }; do
  FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${g}" | tr -d ' ')
  echo "GPU ${g}: ${FREE_MB} MiB free"
  if [ "${FREE_MB}" -lt "${MIN_FREE_MB}" ]; then
    echo "ERROR: GPU ${g} has only ${FREE_MB} MiB free (need ${MIN_FREE_MB})" >&2
    nvidia-smi
    exit 1
  fi
done
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

# Inputs are checked now, not after extraction: model folder, every prompt file, the judge.
STAGES="${STAGES}" python - "${ARGS[@]}" <<'EOF'
import os, sys
from pathlib import Path
from selot.config import add_config_args, load_config, resolve_path
import argparse
args = add_config_args(argparse.ArgumentParser()).parse_args(sys.argv[1:])
cfg = load_config(args.config, args.overrides)
missing = []
if not (Path(cfg["model"]["path"]) / "config.json").exists():
    missing.append(f"model: {cfg['model']['path']}/config.json (download it: README \"Models\")")
for name, p in cfg["data"]["fit"].items():
    if not p or not resolve_path(p).exists():
        missing.append(f"data.fit.{name}: {p}")
for spec in cfg["data"]["eval"] + cfg.get("analysis", {}).get("sets", []):
    if not resolve_path(spec["path"]).exists():
        missing.append(f"{spec['name']}: {spec['path']}")
fd = cfg.get("fit_data") or {}
stages = os.environ.get("STAGES", "label").split(",")
if "label" in stages and fd.get("behaviour", "llamaguard") == "llamaguard" and \
        not (Path(fd["guard"]["path"]) / "config.json").exists():
    missing.append(f"Llama Guard: {fd['guard']['path']}/config.json (README \"Models\", or FIT_BEHAVIOUR=none)")
judges = [cfg["judge"]["name"]] + list(cfg["judge"].get("extra") or [])
if "llamaguard" in judges and not (Path(cfg["judge"]["llamaguard"]["path"]) / "config.json").exists():
    missing.append(f"judge llamaguard: {cfg['judge']['llamaguard']['path']}/config.json (README \"Models\")")
if "harmbench" in judges and not (Path(cfg["judge"]["path"]) / "config.json").exists():
    missing.append(f"judge: {cfg['judge']['path']}/config.json (README \"Models\", or JUDGE=keyword)")
ov = cfg["judge"].get("overrefusal") or {}
if "judge" in stages and ov.get("enabled") and not (Path(ov["path"]) / "config.json").exists():
    missing.append(f"over-refusal judge: {ov['path']}/config.json (README \"Models\", or OVERREFUSAL=0)")
rw = cfg["judge"].get("reward") or {}
if "judge" in stages and rw.get("enabled") and not (Path(rw["path"]) / "config.json").exists():
    missing.append(f"reward model: {rw['path']}/config.json (README \"Models\", or REWARD=0)")
# chat templates (tokenizer_config.json or chat_template.jinja): the generated model, Llama Guard, the over-refusal
# judge and the reward model format their inputs with it
import json as _json
def has_template(p):
    p = Path(p)
    if (p / "chat_template.jinja").exists() or (p / "chat_template.json").exists():
        return True
    tc = p / "tokenizer_config.json"
    return tc.exists() and bool(_json.loads(tc.read_text()).get("chat_template"))
need_tpl = [("model", cfg["model"]["path"])]
if "label" in stages and fd.get("behaviour", "llamaguard") == "llamaguard":
    need_tpl.append(("Llama Guard (label)", fd["guard"]["path"]))
if "judge" in stages:
    if "llamaguard" in judges:
        need_tpl.append(("judge llamaguard", cfg["judge"]["llamaguard"]["path"]))
    if ov.get("enabled"):
        need_tpl.append(("over-refusal judge", ov["path"]))
    if rw.get("enabled"):
        need_tpl.append(("reward model", rw["path"]))
for what, p in need_tpl:
    if (Path(p) / "config.json").exists() and not has_template(p):
        missing.append(f"{what}: no chat template in {p} (tokenizer_config.json / chat_template.jinja): "
                       "download the whole model repo")
sr = cfg["judge"].get("strongreject") or {}
if sr.get("enabled"):
    if not (Path(sr["base_path"]) / "config.json").exists():
        missing.append(f"strongreject base: {sr['base_path']}/config.json (README \"Models\", or STRONGREJECT=0)")
    if not (Path(sr["adapter_path"]) / "adapter_config.json").exists():
        missing.append(f"strongreject adapter: {sr['adapter_path']}/adapter_config.json (README \"Models\", or STRONGREJECT=0)")
    try:
        import peft  # noqa: F401
    except ImportError:
        missing.append("python package peft (StrongREJECT LoRA): pip install -r requirements.txt")
# Pinned versions: requirements.lock.txt (full lock of the reference run) if present, else requirements.txt.
import re, platform
from importlib.metadata import version, PackageNotFoundError
lock = Path("requirements.lock.txt") if Path("requirements.lock.txt").exists() else Path("requirements.txt")
bad = []
for line in lock.read_text().splitlines():
    m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*([^\s#;]+)", line)
    if not m:
        continue
    name, want = m.groups()
    try:
        have = version(name)
    except PackageNotFoundError:
        if name in ("wandb",):          # optional: only the wandb stage needs it (absent on offline servers)
            continue
        have = "not installed"
    if have.split("+")[0] != want.split("+")[0]:
        bad.append(f"{name}: installed {have}, pinned {want}")
mpy = re.search(r"Python (\d+\.\d+)", lock.read_text())
if mpy and platform.python_version().rsplit(".", 1)[0] != mpy.group(1):
    bad.append(f"python: running {platform.python_version()}, reference {mpy.group(1)}")
if bad:
    msg = f"versions differ from {lock}:\n  " + "\n  ".join(bad)
    if os.environ.get("ALLOW_VERSION_MISMATCH") == "1":
        print("WARNING: " + msg + "\n  (ALLOW_VERSION_MISMATCH=1: running anyway, results may not reproduce)")
    else:
        missing.append(msg + "\n  pip install -r " + str(lock) + "  (or ALLOW_VERSION_MISMATCH=1)")
if missing:
    sys.exit("ERROR: missing inputs:\n  " + "\n  ".join(missing))
print(f"inputs OK, versions match {lock}")
EOF

# Exact environment of this run, for reproduction.
{ echo "python $(python -c 'import platform; print(platform.python_version())')"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i "${GPU_IDS}" 2>/dev/null || true
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>/dev/null || true
  python -m pip freeze 2>/dev/null || uv pip freeze 2>/dev/null || true; } > "${RESULTS}/logs/env_$(date +%Y%m%d_%H%M%S).txt"

T0=$(date +%s)
on_exit() {  # always show where things stand, also after a failure or a timeout kill
  local code=$?
  if [ "${code}" != "0" ]; then
    echo; echo "=== FAILED/STOPPED (exit ${code}) after $(( $(date +%s) - T0 ))s - partial summary follows; logs: ${RESULTS}/logs ==="
    python scripts/summarize.py "${ARGS[@]}" || true
  fi
}
trap on_exit EXIT
trap 'exit 143' TERM INT
stage() { echo; echo "=== [$(( $(date +%s) - T0 ))s] $* ==="; }
want() { [[ ",${STAGES}," == *",$1,"* ]]; }
run() { local name=$1; shift; python "scripts/${name}.py" "${ARGS[@]}" "$@" 2>&1 | tee -a "${RESULTS}/logs/${name}.log"; }

if want label;    then stage "0. label: this model's fit sets (${FIT_BEHAVIOUR})"; run label_behaviour; fi
if want extract;  then stage "1. extract activations";       run extract_activations; fi
if want fit;      then stage "2. fit ${METHOD} + checks";    run fit_steering; fi
if want generate; then stage "3. generate (alpha sweep)";     run generate; fi
if want judge;    then stage "4. judge (${JUDGE} + ${JUDGE_EXTRA}, strongreject=${STRONGREJECT}, reward=${REWARD}, overrefusal=${OVERREFUSAL})"; run judge_responses; fi
if want analyze;  then stage "5. activation analysis";        run analyze_activations; fi
if want summary;  then stage "6. summary";                    run summarize; fi
if want wandb;    then stage "7. W&B (scripts/log_wandb.py)";  run log_wandb; fi

echo "ALL DONE - $(( $(date +%s) - T0 ))s. Results: ${RESULTS}/summary.md"
