#!/usr/bin/env python3
"""The new_nothing server job (called by project_commands.sh): SElOT layer sweep -> SElOT full eval at the best layer
-> AlphaSteer full eval, all on fit-data workflow 1, every step on all GPUs of GPU_IDS. Each run goes through
run_stages.sh (one RUN_ID each); every step resumes, so a stopped job continues where it stopped when resubmitted.

  1. label     this model's fit sets (workflow 1, Llama Guard); vLLM splits the prompts over the GPUs
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
  8. compare   results/compare_wf1.md: the two runs side by side

Settings come from the environment (project_commands.sh sets them): WORK, RESULTS_ROOT, MODELS_DIR, GPU_IDS, MODEL,
LAYERS, SWEEP_ALPHA, SMOKE. Steps finished are recorded in <WORK>/pipeline/<step>.done.
SMOKE=1 (test run of the whole flow): SMOKE_SAMPLES prompts per eval / val set (default 16), runs and the result
files end in _smoke, markers in <WORK>/pipeline_smoke; the label and extract steps are the real ones (their outputs
are shared with the real run).
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from select_layer import parse_layers  # noqa: E402

ENV = os.environ
WORK = Path(ENV["WORK"])
RESULTS = Path(ENV["RESULTS_ROOT"])
GPUS = [g.strip() for g in ENV.get("GPU_IDS", "0").split(",") if g.strip()]
MODEL = ENV.get("MODEL", "llama3.1-8b")
WF = "1"
LAYERS = parse_layers(ENV.get("LAYERS", "8-22"))
SWEEP_ALPHA = float(ENV.get("SWEEP_ALPHA", "1.0"))
SMOKE = ENV.get("SMOKE", "0") == "1"
SFX = "_smoke" if SMOKE else ""
DATA = ROOT / "configs" / "data" / "v1.yaml"
MARK = WORK / ("pipeline_smoke" if SMOKE else "pipeline")
TAG = yaml.safe_load(open(ROOT / "configs" / "models" / f"{MODEL}.yaml"))["model"]["tag"]
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


def done(step):
    return (MARK / f"{step}.done").exists()


def mark(step):
    MARK.mkdir(parents=True, exist_ok=True)
    (MARK / f"{step}.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))


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


def sweep_run(layer):
    return f"sweep_selot_wf{WF}{SFX}_L{layer}"


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
                running[g] = (l, stages(sweep_run(l), "selot", "fit,analyze,generate,judge,summary", [g],
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
    out = RESULTS / f"sweep_selot_wf{WF}{SFX}"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "select_layer.py"), "--results-root", str(RESULTS),
                    "--prefix", f"sweep_selot_wf{WF}{SFX}_L", "--layers", ENV.get("LAYERS", "8-22"),
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
    wait([stages(run_id, method, "fit,analyze" if method == "selot" else "fit", GPUS[:1], sets)])
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
                have = {(float(r["alpha"]), str(r["id"])) for r in map(json.loads, d.read_text().splitlines()) if r}
            rows = [r for r in map(json.loads, s.read_text().splitlines())
                    if r and float(r["alpha"]) == 0.0 and (0.0, str(r["id"])) not in have]
            if rows:
                with open(d, "a") as f:
                    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            copied[n + suffix] = len(rows)
    log(f"{dst_run}: alpha 0 rows copied from {src_run}: {sum(copied.values())}")


def main():
    log(f"GPUs {GPUS}, model {MODEL}, fit-data workflow {WF}, sweep layers {LAYERS[0]}-{LAYERS[-1]} at alpha "
        f"{SWEEP_ALPHA:g}" + (f"; SMOKE run: {ENV.get('SMOKE_SAMPLES', '16')} prompts per set" if SMOKE else ""))
    step_label()
    step_extract()
    val_cfg = step_valsets()
    step_sweep(val_cfg)
    best = step_select()
    selot_run = f"selot_wf{WF}{SFX}_L{best}"
    log(f"best layer {best} -> full eval {selot_run}")
    full_eval(selot_run, "selot", [f"model.layers=[{best}]", "steer.alphas=[0.0,1.0]"])
    as_run = f"alphasteer_wf{WF}{SFX}"
    if not done(as_run):
        seed_alpha0(selot_run, as_run)
    full_eval(as_run, "alphasteer")
    out = RESULTS / f"compare_wf{WF}{SFX}.md"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_runs.py"), "--results-root", str(RESULTS),
                    "--runs", selot_run, as_run, "--alpha", "1.0", "--out", str(out)], check=True, cwd=ROOT)
    log(f"ALL DONE: {RESULTS / f'sweep_selot_wf{WF}{SFX}' / 'sweep.md'}, {out}")


if __name__ == "__main__":
    main()
