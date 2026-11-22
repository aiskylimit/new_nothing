# SElOT on the new_nothing server

Jobs are submitted by editing the header (first 3 lines) of the root `commands.sh`. The rest of `commands.sh` stays
as it is, except the run block (Job 3). The server is offline during runs: models and source data come only from
the download job, nothing is uploaded, no W&B.

## Job 1: environment (once)

```
#i selot.txt
#selot
#v1
```

Builds `/mnt/local/uvenvs/selot` from the root `selot.txt` (Python 3.13, the pins of `SElOT/requirements.txt`).

## Job 2: models and source data (once)

Replace the header with the whole content of [`SElOT/download.txt`](download.txt) (it starts with `#d`). It fetches
into `/mnt/local/aiskylimit_new_nothing/selot/`:
- `models/`: Llama-3.1-8B-Instruct, Llama-Guard-3-8B, HarmBench-Llama-2-13b-cls, gemma-2b + strongreject-15k-v1,
  Skywork-Reward-V2-Llama-3.1-8B, Qwen2.5-14B-Instruct (over-refusal judge), all-MiniLM-L6-v2 (data build);
- `raw_src/`: every source file of the fit and eval sets (Hugging Face datasets, XSTest / OKTest / AlphaSteer's
  attack prompts from GitHub), one folder per file. Gated repos (Llama 3.1, Llama Guard 3, gemma-2b,
  allenai/wildguardmix, allenai/coconot, JailbreakV-28K) need their terms accepted by the downloading account.

## Job 3: run (resubmit until it prints `ALL DONE`)

```
#1 +120
#selot
#v1
```

In the body, keep only this block active (comment out any other project):

```bash
cd ./SElOT
bash ./project_commands.sh
```

**When a job stops (timeout or error), submit it again with `#v1` flipped to `#v2` (or back).** Every step resumes:
finished steps are skipped (`work/pipeline/*.done`), generation and judging continue per prompt.

What one run does (GPUs 2 and 3, both on every step), in order:
1. rebuild the data from `raw_src/` (`fetch_data.py --src`, `prepare_eval.py`, `build_fit_sets.py`); the log says
   whether it is identical to the reference build (`data_manifest.json`);
2. `label`: this model's fit sets, fit-data **workflow 1** (the first Colab run's pipeline), Llama Guard 3;
3. activations at layers 8-22 (the sweep's 10-22 and AlphaSteer's 8-19);
4. **SElOT layer sweep** over layers 10-22 (30-70% of the 32 layers, `model.layer_range`): k = 5, m = 256, alpha = 1, one layer at a time per GPU, on the val split of the fit sets
   (val_jailbreak, val_refused: DSR by Llama Guard; val_benign: refusal prefix);
5. best layer = highest mean DSR over val_jailbreak and val_refused;
6. SElOT full eval at that layer (alpha 0 and 1) on the test sets, every judge;
7. **AlphaSteer** full eval (10 layers 8-19, our D_r + D_j, r = mean(D_b) - mean(D_r), lambda = 0.5), same
   activations, same alpha 0 responses;
8. `results/compare_wf1.md`.

## Job 4: fetch results

```
#2 -f-/mnt/local/aiskylimit_new_nothing/SElOT/results/ +a
#selot
#v1
```

| File | Content |
|---|---|
| `results/sweep_selot_wf1/sweep.md` | layer sweep table and the best layer |
| `results/compare_wf1.md` | SElOT (best layer) vs AlphaSteer, alpha 0 vs 1, every set |
| `results/results_<run>/summary.md` | per run: fit sets, fit checks, every metric per set and alpha |
| `results/results_<run>/eval/metrics.json` | every number |
| `results/logs/job_*.log`, `results/results_<run>/logs/` | console output of each job / stage |

Runs: `prep_wf1` (label), `sweep_selot_wf1_L<layer>`, `selot_wf1_L<best>`, `alphasteer_wf1`.

## Settings

Edit the defaults at the top of `SElOT/project_commands.sh` (commands.sh passes nothing): `GPU_IDS` (default
`2,3`), `LAYERS` (empty = 30-70% of depth: `10-22`), `SWEEP_ALPHA` (`1.0`), `MIN_FREE_MB` (free memory needed per GPU).

## If something goes wrong

| Message | Meaning / action |
|---|---|
| `ERROR: /mnt/local/uvenvs/selot does not exist` | Job 1 has not run (or failed). |
| `ERROR: missing ...` | Job 2 has not run or is incomplete. |
| `offline: expected one <file> under ...` | a source file is missing in `raw_src/`: rerun Job 2. |
| `ERROR: GPU <id> has only ... MiB free` | another job uses the GPU; retry later or change `GPU_IDS`. |
| `versions differ from requirements.txt` | the env does not match `selot.txt`; rebuild it (Job 1). |
| `DATA DIFFERS from the reference build` | not an error: the rebuilt data differs from the local build (listed files). |
| `FAILED/STOPPED (exit 143)` | timeout: resubmit Job 3. |
