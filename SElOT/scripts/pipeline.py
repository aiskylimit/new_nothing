#!/usr/bin/env python3
"""The new_nothing server job (called by project_commands.sh): SElOT layer sweep -> SElOT full eval at the best layer
-> AlphaSteer full eval, all on fit-data workflow WF (SELOT_WF: 2 = train/val split by behaviour group, 1 = the first
Colab run), every step on all GPUs of GPU_IDS. SELOT_UNTIL=select stops after step 5 (layer choice only). Each run goes through
run_stages.sh (one RUN_ID each); every step resumes, so a stopped job continues where it stopped when resubmitted.

  1. label     this model's fit sets (workflow WF, Llama Guard); vLLM splits the prompts over the GPUs
  2. extract   last-token activations at layers LAYERS + AlphaSteer's layers (configs/methods/alphasteer.yaml), one
               fit set group per GPU
  3. val sets  val_jailbreak / val_refused / val_benign from the val split of the fit sets (make_val_sets.py)
  4. sweep     SElOT (configs/methods/selot.yaml: k 5, m 256) fitted at each layer alone, activation analysis
               (analyze: displacement per val set, transport match), generated at alpha SWEEP_ALPHA on the val
               sets, judged by Llama Guard; one layer per GPU at a time
  5. select    best layer = highest mean DSR over val_jailbreak and val_refused (select_layer.py)
  6. selot     full eval (test sets) at the best layer, alphas [0, 1], with the activation analysis on test prompts
               (configs/data/v1.yaml analysis.sets); generation split over the GPUs (every n-th
               prompt of every set), then every judge (Llama Guard, HarmBench, StrongREJECT, reward model, XSTest
               3-class over-refusal judge)
  7. alphasteer full eval of AlphaSteer (configs/methods/alphasteer.yaml) on the same activations; its alpha 0
               responses (the unsteered model) and their labels are copied from step 6, not generated again;
               no activation analysis (the baseline is not tuned)
  8. report    results/compare_wf<WF>.md: the final tables (safety: DSR per attack by Llama Guard and HarmBench,
               StrongREJECT; utility: over-refusal, AlpacaEval, MMLU, GSM8K), one row per method;
               results/tune_log_wf<WF>.md: the layer sweep + every run's summary, for tuning

Settings come from the environment (project_commands.sh sets them): WORK, RESULTS_ROOT, MODELS_DIR, GPU_IDS, MODEL,
LAYERS (sweep layers, e.g. "10-22"; empty = model.layer_range of configs/base.yaml, 30-70% of the model's decoder
layers: 10-22 for Llama-3.1-8B), SWEEP_ALPHA, SMOKE, SELOT_WF, SELOT_UNTIL (select | all), SELOT_METHOD (selot |
selot_gated | selot_gated_logistic | selot_gated_single: the steering method of the sweep and the SElOT full eval; its
runs and report files carry its name, e.g. sweep_selot_gated_wf2_L12, compare_wf2_selot_gated.md). Steps finished are
recorded in <WORK>/pipeline/<step>.done for workflow 1, <WORK>/pipeline_wf<WF>/ for the others (label and extract
build per-workflow fit sets and activations, so their markers must not be shared between workflows).
SMOKE=1 (test run of the whole flow): SMOKE_SAMPLES prompts per eval / val set (default 16), runs and the result
files end in _smoke, markers in <WORK>/pipeline[_wf<WF>]_smoke; the label and extract steps are the real ones (their
outputs and markers are shared with the real run).
"""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import read_jsonl  # noqa: E402  (splits on \n only: responses may hold \u2028, \x0b, ...)
from select_layer import parse_layers  # noqa: E402

ENV = os.environ
WORK = Path(ENV["WORK"])
RESULTS = Path(ENV["RESULTS_ROOT"])
GPUS = [g.strip() for g in ENV.get("GPU_IDS", "0").split(",") if g.strip()]
MODEL = ENV.get("MODEL", "llama3.1-8b")
WF = ENV.get("SELOT_WF", "1").strip()
if WF not in ("1", "2"):
    raise SystemExit(f"SELOT_WF must be 1 or 2, not {WF!r}")
UNTIL = ENV.get("SELOT_UNTIL", "all").strip()        # select: stop after the layer choice | all: full evals + report
if UNTIL not in ("select", "all"):
    raise SystemExit(f"SELOT_UNTIL must be select or all, not {UNTIL!r}")


def default_layers():
    """model.layer_range (configs/base.yaml, the model YAML may override it) over the model's decoder layers, as
    selot.activations.resolve_layers; [] when the model folder is not there"""
    mcfg = {**yaml.safe_load(open(ROOT / "configs" / "base.yaml"))["model"],
            **yaml.safe_load(open(ROOT / "configs" / "models" / f"{MODEL}.yaml"))["model"]}
    f = Path(ENV.get("MODELS_DIR", "models")) / Path(mcfg["path"]).name / "config.json"
    if not f.exists():
        return []
    n = json.loads(f.read_text())["num_hidden_layers"]
    lo, hi = mcfg.get("layer_range", [0.3, 0.7])
    return list(range(math.ceil(lo * n), math.floor(hi * n) + 1))


LAYERS = parse_layers(ENV["LAYERS"]) if ENV.get("LAYERS", "").strip() else default_layers()
SWEEP_ALPHA = float(ENV.get("SWEEP_ALPHA", "1.0"))
SMOKE = ENV.get("SMOKE", "0") == "1"
SFX = "_smoke" if SMOKE else ""
DATA = ROOT / "configs" / "data" / "v1.yaml"
# workflow 1 keeps the marker folder of the first runs (they resume); other workflows get their own
MARK_BASE = WORK / ("pipeline" if WF == "1" else f"pipeline_wf{WF}")
MARK = MARK_BASE.with_name(MARK_BASE.name + "_smoke") if SMOKE else MARK_BASE
# label and extract run on the full data in a smoke run too: their markers are shared with the real run
SHARED_STEPS = ("label", "extract")
TAG = yaml.safe_load(open(ROOT / "configs" / "models" / f"{MODEL}.yaml"))["model"]["tag"]
# the steering method of the sweep and the full eval: configs/methods/<SM>.yaml (selot, or selot_gated: a gate + one
# Gaussian OT per jailbreak mode); its runs are named after it (sweep_<SM>_wf<WF>_L<l>, <SM>_wf<WF>_L<l>)
SM = ENV.get("SELOT_METHOD", "selot").strip() or "selot"
if not SM.startswith("selot") or not (ROOT / "configs" / "methods" / f"{SM}.yaml").exists():
    raise SystemExit(f"SELOT_METHOD={SM!r}: needs configs/methods/<name>.yaml with a name starting with selot")
RSFX = SFX if SM == "selot" else f"{SFX}_{SM}"     # report files: compare_wf<WF><RSFX>.md, tune_log_wf<WF><RSFX>.md
AS_LAYERS = yaml.safe_load(open(ROOT / "configs" / "methods" / "alphasteer.yaml"))["model"]["layers"]
EXTRACT_LAYERS = sorted(set(LAYERS) | set(AS_LAYERS))      # the sweep's layers and AlphaSteer's 10
ACTS = WORK / "acts" / TAG / f"wf{WF}" / "llamaguard"
FITSETS = WORK / "fitsets" / TAG / f"wf{WF}" / "llamaguard"
VAL = WORK / "valsets" / TAG / f"wf{WF}"
# B200-sized batches (HF generation / activations / HF judges); vLLM takes its share of each GPU
BATCH = ["generate.batch_size=64", "extract.batch_size=32", "judge.strongreject.batch_size=16",
         "judge.reward.batch_size=16", "fit_data.vllm_gpu_mem=0.85", "judge.vllm_gpu_mem=0.85"]
if SMOKE:
    BATCH += [f"generate.max_samples={int(ENV.get('SMOKE_SAMPLES', '16'))}",
              f"analysis.max_samples={int(ENV.get('SMOKE_SAMPLES', '16'))}"]
FULL_JUDGES = dict(JUDGE="llamaguard", JUDGE_EXTRA="[harmbench]", STRONGREJECT="1", REWARD="1", OVERREFUSAL="1")
VAL_JUDGES = dict(JUDGE="llamaguard", JUDGE_EXTRA="[]", STRONGREJECT="0", REWARD="0", OVERREFUSAL="0")


def log(msg):
    print(f"\n[pipeline {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mark_dir(step):
    return MARK_BASE if step in SHARED_STEPS else MARK


def done(step):
    return (mark_dir(step) / f"{step}.done").exists() or (MARK / f"{step}.done").exists()


def mark(step):
    mark_dir(step).mkdir(parents=True, exist_ok=True)
    (mark_dir(step) / f"{step}.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))


def stages(run_id, method, stage_list, gpus, sets=(), data=DATA, judges=FULL_JUDGES):
    """start run_stages.sh for one run; returns the process"""
    env = {**ENV, "RUN_ID": run_id, "METHOD": method, "STAGES": stage_list, "GPU_IDS": ",".join(gpus),
           "MODEL": MODEL, "FIT_WORKFLOW": WF, "FIT_BEHAVIOUR": "llamaguard", "DATA_CONFIG": str(data),
           "VENV": "none", "EXTRA_SETS": " ".join(BATCH + list(sets)), **judges}
    log(f"{run_id}: {stage_list} on GPU {','.join(gpus)} {' '.join(sets)}")
    return subprocess.Popen(["bash", str(ROOT / "run_stages.sh")], env=env, cwd=ROOT)


def wait(procs):
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"a run failed (exit codes {codes}); see the logs under {RESULTS}")


def metrics(run_id):
    f = RESULTS / f"results_{run_id}" / "eval" / "metrics.json"
    return json.loads(f.read_text()) if f.exists() else None


# ---------------------------------------------------------------- steps
def step_label():
    if done("label"):
        return log("label: done")
    wait([stages(f"prep_wf{WF}", "selot", "label", GPUS)])
    mark("label")


def extracted_layers():
    f = ACTS / "layers.json"
    return set(json.loads(f.read_text())["layers"]) if f.exists() else set()


def step_extract():
    if done("extract") and set(EXTRACT_LAYERS) <= extracted_layers():
        return log("extract: done")
    layers = "model.layers=[" + ",".join(map(str, EXTRACT_LAYERS)) + "]"
    groups = [["benign"], ["refused", "jailbreak"]] if len(GPUS) > 1 else [["benign", "refused", "jailbreak"]]
    wait([stages(f"prep_wf{WF}_extract{k}", "selot", "extract", [g], [layers, "extract.only_sets=[" + ",".join(sets) + "]"])
          for k, (g, sets) in enumerate(zip(GPUS, groups))])
    mark("extract")


def step_valsets():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_val_sets.py"), "--fitsets", str(FITSETS),
                    "--out", str(VAL), "--data-config", str(DATA)], check=True, cwd=ROOT)
    return VAL / "data.yaml"


def plot_shift(run_id, layer, alphas="1"):
    """scripts/plot_shift.py for a fitted SElOT run: the three groups before / after steering, and the gate of a gated
    map (PNG in results_<run>/plots); skipped without matplotlib; a failure only warns"""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return log("plot_shift: matplotlib not installed, no activation plots")
    code = subprocess.run([sys.executable, str(ROOT / "scripts" / "plot_shift.py"), "--acts", str(ACTS), "--map",
                           str(WORK / "maps" / run_id / "map.pt"), "--layer", str(layer), "--alphas", alphas,
                           "--out", str(RESULTS / f"results_{run_id}" / "plots"), "--title", f"{TAG} {run_id}"],
                          cwd=ROOT).returncode
    if code:
        log(f"WARNING: plot_shift {run_id} failed (exit {code}); continuing")


def sweep_run(layer):
    return f"sweep_{SM}_wf{WF}{SFX}_L{layer}"


def sweep_done(layer):
    m = metrics(sweep_run(layer))
    alphas = [0.0, SWEEP_ALPHA] if layer == LAYERS[0] else [SWEEP_ALPHA]
    analysed = (RESULTS / f"results_{sweep_run(layer)}" / "analysis.json").exists()
    return bool(m) and analysed and all(str(a) in m["sets"].get(n, {}).get("per_alpha", {})
                           for n in ("val_jailbreak", "val_refused", "val_benign") for a in alphas)


def step_sweep(val_cfg):
    todo = [l for l in LAYERS if not sweep_done(l)]
    log(f"sweep: {len(LAYERS) - len(todo)} of {len(LAYERS)} layers done; to do {todo}")
    running = {}                                     # gpu -> (layer, process)
    while todo or running:
        for g in GPUS:
            if g not in running and todo:
                l = todo.pop(0)
                alphas = f"[0.0,{SWEEP_ALPHA}]" if l == LAYERS[0] else f"[{SWEEP_ALPHA}]"   # alpha 0 once: reference
                running[g] = (l, stages(sweep_run(l), SM, "fit,analyze,generate,judge,summary", [g],
                                        [f"model.layers=[{l}]", f"steer.alphas={alphas}"], val_cfg, VAL_JUDGES))
        time.sleep(10)
        for g, (l, p) in list(running.items()):
            if p.poll() is not None:
                del running[g]
                if p.returncode:
                    for _, (_, q) in running.items():
                        q.wait()
                    raise SystemExit(f"sweep layer {l} failed (exit {p.returncode})")
                log(f"sweep layer {l}: done")


def step_select():
    out = RESULTS / f"sweep_{SM}_wf{WF}{SFX}"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "select_layer.py"), "--results-root", str(RESULTS),
                    "--prefix", f"sweep_{SM}_wf{WF}{SFX}_L", "--layers", ",".join(map(str, LAYERS)),
                    "--alpha", str(SWEEP_ALPHA), "--out", str(out)], check=True, cwd=ROOT)
    return int((out / "best_layer.txt").read_text())


def eval_set_names():
    return [s["name"] for s in yaml.safe_load(open(DATA))["data"]["eval"]]


def merge(run_id):
    from generate import merge_shards
    gen = WORK / "generations" / run_id
    added = {n: merge_shards(gen, n) for n in eval_set_names()}
    log(f"{run_id}: merged generation shards {added}")


def full_eval(run_id, method, sets=()):
    """fit (+ activation analysis for SElOT) -> generation split over the GPUs -> merge -> judge (vLLM judges over
    every GPU) -> summary"""
    if done(run_id):
        return log(f"{run_id}: done")
    wait([stages(run_id, method, "fit,analyze" if method.startswith("selot") else "fit", GPUS[:1], sets)])
    n = len(GPUS)
    wait([stages(run_id, method, "generate", [g], list(sets) + [f"generate.shard=[{k},{n}]"])
          for k, g in enumerate(GPUS)])
    merge(run_id)
    wait([stages(run_id, method, "judge,summary", GPUS, sets)])
    mark(run_id)


def seed_alpha0(src_run, dst_run):
    """copy the alpha 0 rows (unsteered model: the same for every method) of src_run's generations and labels into
    dst_run, where they are missing"""
    src, dst = WORK / "generations" / src_run, WORK / "generations" / dst_run
    dst.mkdir(parents=True, exist_ok=True)
    copied = {}
    for n in eval_set_names():
        for suffix in (".jsonl", ".judged.jsonl"):
            s, d = src / f"{n}{suffix}", dst / f"{n}{suffix}"
            if not s.exists():
                continue
            have = set()
            if d.exists():
                have = {(float(r["alpha"]), str(r["id"])) for r in read_jsonl(d)}
            rows = [r for r in read_jsonl(s)
                    if r and float(r["alpha"]) == 0.0 and (0.0, str(r["id"])) not in have]
            if rows:
                with open(d, "a") as f:
                    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            copied[n + suffix] = len(rows)
    log(f"{dst_run}: alpha 0 rows copied from {src_run}: {sum(copied.values())}")


def main():
    if not LAYERS:
        raise SystemExit(f"no sweep layers: set LAYERS, or put the model under MODELS_DIR ({ENV.get('MODELS_DIR')})")
    log(f"GPUs {GPUS}, model {MODEL}, method {SM}, fit-data workflow {WF}, sweep layers {LAYERS[0]}-{LAYERS[-1]} at alpha "
        f"{SWEEP_ALPHA:g}" + ("; layer choice only (SELOT_UNTIL=select)" if UNTIL == "select" else "")
        + (f"; SMOKE run: {ENV.get('SMOKE_SAMPLES', '16')} prompts per set" if SMOKE else ""))
    step_label()
    step_extract()
    val_cfg = step_valsets()
    step_sweep(val_cfg)
    best = step_select()
    if UNTIL == "select":
        plot_shift(sweep_run(best), best, "1,2")            # the best layer's sweep map (fitted on train)
        return log(f"ALL DONE: best layer {best} (SELOT_UNTIL=select: no full eval); "
                   f"{RESULTS / f'sweep_{SM}_wf{WF}{SFX}' / 'sweep.md'}")
    selot_run = f"{SM}_wf{WF}{SFX}_L{best}"
    log(f"best layer {best} -> full eval {selot_run}")
    full_eval(selot_run, SM, [f"model.layers=[{best}]", "steer.alphas=[0.0,1.0]"])
    plot_shift(selot_run, best, "1,2")
    as_run = f"alphasteer_wf{WF}{SFX}"
    if not done(as_run):
        seed_alpha0(selot_run, as_run)
    full_eval(as_run, "alphasteer")
    report([selot_run, as_run])


def report(runs):
    """the final tables (results/compare_wf*.md: safety, utility) and the tuning log (results/tune_log_wf*.md)"""
    out, tune = RESULTS / f"compare_wf{WF}{RSFX}.md", RESULTS / f"tune_log_wf{WF}{RSFX}.md"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_runs.py"), "--results-root", str(RESULTS),
                    "--runs", *runs, "--alpha", "1.0", "--out", str(out), "--log-out", str(tune),
                    "--sweep-md", str(RESULTS / f"sweep_{SM}_wf{WF}{SFX}" / "sweep.md"), "--work", str(WORK)],
                   check=True, cwd=ROOT)
    log(f"ALL DONE: final tables {out}; tuning log {tune}")


if __name__ == "__main__":
    main()
