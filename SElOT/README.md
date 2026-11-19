# SelOT: Selective Optimal Transport for Jailbreak Defense

Selective optimal-transport steering for jailbreak defense (the ACL 2027 proposal). At each steered layer it
fits, in closed form and in float32, an affine map of the residual stream

```
h <- A h + b,   A = I + alpha * U V^T,   b = alpha * b0        (U = P, V = Q W^T, b0 = P (c - W Q^T mu_all))
```

which is `h + alpha * g(h)` with `g(h) = P M psi(h)` (proposal Eq. 8 and 11). M is the benign-cost weighted
ridge fit (Eq. 10, or its hard null-space variant) towards the Gaussian OT target delta* (Eq. 6). Every
baseline is stored as the same per-layer map and goes through the same hook:

| method (`configs/methods/`) | map |
|---|---|
| `selot`, `selot_hard`, `selot_const_target` | rank-k U V^T + b0 |
| `pca_ot` (Prop. 1a) | rank-k U V^T + b0, applied to every input |
| `dim` (difference in means; `subspace_k` > 0 gives Prop. 1b) | b0 only (A = I) |
| `linear_act` | diagonal + b0 |
| `vector_file` (Surgical, Jailbreak Antidote, ... precomputed) | b0 only |
| `alphasteer_form` (AlphaSteer's Delta, fitted by AlphaSteer's code) | dense D = Delta^T |
| `alphasteer` (AlphaSteer fitted here on our fit sets: null space of D_b, ridge to r = mean(D_b) - mean(D_r), its 10 layers / ratios, lambda 0.5) | dense D = strength Delta^T |

**Where it steers.** The map is applied at the input of decoder layer l, the same tensor the activations
are extracted from. `steer.positions` sets which positions:

- `prompt` (default): every prompt token gets its own A h_t + b, at prefill only. Generated tokens are not
  steered.
- `alphasteer`: AlphaSteer's rule, for that baseline only. One displacement is computed from the last prompt
  token and added to every prompt position.
- `all`: every prompt and generated token, as in Eq. 11 (ablation).

## Setup

**Hardware.** Suggested: one NVIDIA A100 80GB or larger (an 8B backbone in bf16, the 13B HarmBench judge, long
generations with batching). Smaller GPUs work with smaller batch sizes (`--set generate.batch_size=...`).

```bash
pip install -r requirements.txt      # or requirements.lock.txt (every package pinned) when present
```

Core versions are pinned (Python 3.13, torch 2.14.0, transformers 5.22.1, vLLM 0.34.0). The launcher stops if the
installed versions differ (`ALLOW_VERSION_MISMATCH=1` to run anyway) and saves `pip freeze` of every run to
`results_<RUN_ID>/logs/env_*.txt`.

**Models.** One folder per model under `models/` (or set `MODELS_DIR`). Llama 3.1, Gemma 2, gemma-2b and Llama Guard 3
are gated on Hugging Face: accept their licences and `huggingface-cli login` first.

```bash
for repo in meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct google/gemma-2-9b-it Qwen/Qwen2.5-14B-Instruct \
            meta-llama/Llama-Guard-3-8B cais/HarmBench-Llama-2-13b-cls google/gemma-2b qylu4156/strongreject-15k-v1 \
            Skywork/Skywork-Reward-V2-Llama-3.1-8B; do
  huggingface-cli download "$repo" --local-dir "models/${repo#*/}" --exclude "original/*"
done
```

Backbones: download only the ones you run. Llama Guard 3 labels the fit sets (stage `label`) and is the main ASR
judge; HarmBench, gemma-2b + StrongREJECT LoRA and Skywork-Reward-V2 (`REWARD=0` to skip) are judges (stage `judge`).

**Data.** `data/fit/` (candidate pools) and `data/eval/` (eval sets) are not in git: build them with the commands in
"Fitting data" below.

**Keys.** Only in the environment, never in a file: `HF_TOKEN` (model downloads), `WANDB_API_KEY` (optional stage
`wandb`). No judge calls an API: every judge is a local model.

## Run

```bash
MODEL=llama3.1-8b METHOD=selot bash run_stages.sh
```

`run_stages.sh` runs the stages of one run (one `RUN_ID`). `project_commands.sh` is the job of the offline
new_nothing server (see [RUN.md](RUN.md)): it rebuilds the data from downloaded source files (`fetch_data.py --src`),
then `scripts/pipeline.py` runs the SElOT layer sweep on the val split, the full eval at the best layer and the
AlphaSteer baseline, on several GPUs (`GPU_IDS`).

- Stages, in order: `label` (per-model fit sets) → `extract` → `fit` → `generate` (alpha sweep) → `judge` →
  `analyze` → `summary` (+ `wandb`). `STAGES=judge,summary` runs a subset; every stage resumes where it stopped.
- `MODEL` = `configs/models/<MODEL>.yaml`, `METHOD` = `configs/methods/<METHOD>.yaml`, `DATA_CONFIG` defaults to
  `configs/data/v1.yaml`.
- Judges (all local): `JUDGE=llamaguard` (default; AlphaSteer's Llama Guard 3 protocol), `JUDGE_EXTRA=[harmbench]`,
  `STRONGREJECT=1`, `REWARD=1`. Engines are fixed: Llama Guard and HarmBench run on vLLM (as AlphaSteer's and
  HarmBench's own evaluations do; `judge.vllm_gpu_mem`), StrongREJECT and Skywork-Reward-V2 on HF (they read logits,
  no generation). Each label stores its engine (`<label>_engine`) and `metrics.json` lists them (`engines`); a stored
  label is reused only when its engine matches. Eval responses (alpha = 0 and steered) are always generated with HF.
- Engine of the label stage's responses: `fit_data.engine=vllm` (default) | `hf`; its Llama Guard is always vLLM.
- `FIT_BEHAVIOUR`: `llamaguard` (default) | `none` (ablation: every pooled prompt) | `keyword`.
- `FIT_WORKFLOW` (`fit_data.workflow`): `2` (default) = fixed train/val split by behaviour groups shared by the
  three pools (`split` in `data/fit`); `1` = the first Colab run's pipeline (near-duplicate groups within each pool,
  `group_v1`, split per model after the topic cap; the vanilla and jailbreak versions of one behaviour can land on
  different sides). One `build_fit_sets.py` run writes the columns of both. The default `RUN_ID` ends in `_wf<k>`.
- `EXTRA_SETS` passes config overrides, e.g. `"method.beta=1000 steer.alphas=[0,0.5,1]"`.
- Paths: `MODELS_DIR` (default `models/`), `WORK` (large files, default `work/`), `RESULTS_ROOT` (default: here),
  `VENV` (a virtualenv to activate; default: the current python).

The launcher checks the free GPU memory, the model, every prompt file and the judges before it starts. All console
output also goes to `results_<RUN_ID>/logs/`.

## Run by hand

Configs are merged left to right, then the `--set` overrides are applied:

```bash
CFG="--config configs/base.yaml --config configs/models/llama3.1-8b.yaml --config configs/methods/selot.yaml --config configs/data/v1.yaml"
python scripts/label_behaviour.py $CFG              # this model's D_b / D_r / D_j (Llama Guard), train / val
python scripts/extract_activations.py $CFG          # train-split activations + topic weights, cached per model
python scripts/fit_steering.py $CFG                 # (A, b) per layer + numerical checks
python scripts/generate.py $CFG                     # steered generation for every alpha in steer.alphas
python scripts/judge_responses.py $CFG              # ASR / refusal / compliance per (set, alpha)
python scripts/analyze_activations.py $CFG          # RQ3 displacement norms, transport match
python scripts/summarize.py $CFG                    # summary.md
```

To load a fitted map yourself:

```python
from selot.affine_map import AffineMap
from selot.steering import Steerer
amap = AffineMap.load("…/maps/<run_id>/map.pt")
with Steerer(model, amap, positions="prompt") as s:
    s.set_alpha(0.5)
    with s.prompt(input_ids.shape[1]):          # left-padded prompt length
        out = model.generate(input_ids=input_ids, attention_mask=mask, do_sample=False)
```

## Eval metrics

Each eval set in the data config names its `metric` (`selot/metrics.py`); `configs/data/v1.yaml` has them all.

| metric | sets | numbers in `eval/metrics.json` |
|---|---|---|
| `asr` | AdvBench-100, attack_{gcg, autodan, pair, renellm, jailbroken, cipher, aim} | `asr` and `dsr` = 1 - asr from the main judge (`JUDGE=llamaguard`: Llama Guard 3 8B on (attack prompt, response), as in AlphaSteer's Llama Guard table), `asr_harmbench` (`JUDGE_EXTRA=[harmbench]`: HarmBench classifier vs `behavior`, response clipped to 512 tokens), `strongreject` (mean fine-tuned StrongREJECT score), `refusal_rate`; Cipher: `decoded_rate` (see below) |
| `refusal` | XSTest safe, OR-Bench-Hard-1K, OKTest; XSTest unsafe (`kind: harmful`) | `compliance_rate` = share of `1_full_compliance` from the over-refusal judge (`OVERREFUSAL=1`: AlphaSteer's XSTest 3-class prompt, verbatim, on local Qwen2.5-14B-Instruct; a partial refusal is not compliant; AlphaSteer uses gpt-4o), `refusal_rate` = 1 - compliance; `refusal_rate_prefix` (refusal-prefix matcher, also the fallback without the judge) |
| `gsm8k` | GSM8K test | `accuracy`: last `\boxed{}` (else last number) equals the gold number; AlphaSteer's prompt template |
| `mmlu` | MMLU test | `accuracy`, `macro_accuracy` over subjects: argmax of the A/B/C/D logits at the first answer token, zero-shot, steering on |
| `alpaca_eval` | AlpacaEval 805 | `avg_words`; with the reward model (`REWARD=1`): `reward`, `reward_delta`, `win_rate_vs_base`, `lc_win_rate_vs_base`, `win_rate_vs_ref`, `lc_win_rate_vs_ref` (see below) |

Cipher (CipherChat: Caesar, SelfDefine, ASCII, Morse; field `cipher`) is judged on the decoded response, against the
plain question, as CipherChat does: `selot/cipher.py` holds CipherChat's own decoders (MIT). One rule is added:
a response that reads more like English than its decoding (a plain-English refusal or answer) is judged as it is.
Judging the raw cipher text would score nearly every response safe (Llama Guard, HarmBench) or unsafe (GPT judge).

AlpacaEval quality is scored by a local reward model, Skywork-Reward-V2-Llama-3.1-8B (`selot/reward.py`, used as its
model card shows: [user, assistant] through the chat template, score = the single logit), in the judge stage:
- `win_rate_vs_base`: share of prompts where the steered response gets a higher reward than the unsteered (alpha 0)
  response to the same prompt (tie = 0.5). 0.5 = steering does not change quality; this is the number for tuning.
- `win_rate_vs_ref`: the same against the text-davinci-003 output (`reference`), AlphaSteer's AlpacaEval opponent.
- `lc_*`: length controlled, after AlpacaEval 2: logistic fit of win ~ tanh(word-count difference / its std), read
  at difference 0.
- `reward` (mean), `reward_delta` (mean change vs alpha 0). Rewards are stored per response (`reward`, `reward_ref`).
AlphaSteer's gpt-4o-mini AlpacaEval judge is not used (no API judges), so these win rates are not comparable to its
table, only between configurations.

`max_samples` on an eval set takes a fixed random subset (seed 0), e.g. for MMLU's 14k questions.

## Fitting data

Steps (i)-(iii) are model-independent and run once, locally. They write the candidate pools to `data/fit/`:

```bash
python scripts/fetch_data.py                    # every fit + eval source -> data/raw/ (WildGuardMix is gated: hf auth login)
python scripts/build_fit_sets.py --device cpu   # (i) pools, (ii) dedup vs eval, (iii) topics -> data/fit/, data/reports/
python scripts/prepare_eval.py                  # eval sets with their metric fields -> data/eval/
```

1. **Candidate pools.**
   - Benign, about 16.5k: Alpaca, CoCoNot, MATH train, and WildGuardMix benign (150 per topic).
   - Vanilla harmful, about 3.6k: AdvBench minus the 100 test prompts, MaliciousInstruct, TDC23, JBB and
     Do-Not-Answer, all kept. WildGuardMix vanilla fills the topics that have the fewest prompts.
   - Jailbreak, about 3k: 1k JailbreakV text prompts, spread over topics with ≤ 2 variants per question.
     WildGuardMix adversarial fills the rest, fewest-first.
2. **Dedup.** Run on the whole raw pool against every eval set. A prompt is dropped on an exact match,
   containment, MiniLM cosine ≥ 0.85 or sentence-BLEU ≥ 0.5.
3. **Topics.** 11 harm topics (`scripts/topics.py`, `taxonomy.yaml`). Each prompt also gets a near-duplicate
   `group`.

Reports: `data/reports/dedup_report.md`, `bleu.json` (AlphaSteer-style BLEU), `composition.md`, `removed.jsonl`.

Steps (iv)-(v) run per model, in the `label` stage (`scripts/label_behaviour.py`, `selot/fit_data.py`):

4. **Behaviour labelling.** The unsteered model answers every pooled prompt. Llama Guard (`Llama-Guard-3-8B`)
   labels the harmful-pool responses.
   - A vanilla prompt with a safe response goes to D_r.
   - A jailbreak with an unsafe response goes to D_j.
   - A benign prompt the model does not refuse (refusal-prefix matcher) goes to D_b.
5. **Balancing and split.**
   - D_r and D_j are capped at 150 prompts per topic. Within a topic the curated sources share the cap, and
     WildGuardMix fills the rest.
   - 1/7 of each set is held out (`split: val`), stratified by topic, with whole groups, fixed seed.
   - The train part of D_r and D_j gets inverse-topic-frequency weights (mean 1, clipped at 5). The fit uses them in
     every mean, covariance, PCA and ridge weight over D_r and D_j.

Labels are cached in `<work_dir>/behavior/<model>/` and sets go to `<work_dir>/fitsets/<model>/wf<workflow>/<behaviour>/`. Sizes
are in `results_<RUN_ID>/fit_sets.json` and `summary.md`. `FIT_BEHAVIOUR=none` keeps every pooled prompt (ablation).
`keyword` uses the refusal prefix instead of Llama Guard.

## Dataset files

Each set is one file whose path goes in the config. The format is chosen by the extension:

- `.txt`: one prompt per line.
- `.json`: a list of strings, or a list of `{"prompt": …}`.
- `.jsonl`: one `{"prompt": …}` per line.

Extra fields are kept in the eval outputs. `id` keys the resume logic. `behavior` is what the HarmBench
judge compares the response against, and it falls back to the prompt.

A data config looks like this:

```yaml
data:
  fit:
    benign: /path/benign.jsonl        # D_b: prompts the model answers (drop ones it refuses)
    refused: /path/refused.jsonl      # D_r: harmful prompts it refuses
    jailbreak: /path/jailbreak.jsonl  # D_j: jailbreaks it answers
  eval:
    - {name: harmbench_gcg, path: /path/hb_gcg.jsonl, kind: harmful}   # ASR
    - {name: xstest_safe, path: /path/xstest.jsonl, kind: benign}      # compliance rate
analysis:
  sets:                               # held-out prompts for RQ3 displacement norms
    - {name: orbench_hard, path: /path/orbench.jsonl, group: borderline}
```

## Outputs

| what | where |
|---|---|
| activations `{layer: n x d float32}` | `<work_dir>/acts/<model.tag>/wf<workflow>/<behaviour>/{benign,refused,jailbreak}.pt` |
| fitted map | `<work_dir>/maps/<run_id>/map.pt` |
| generations, judge labels | `<work_dir>/generations/<run_id>/<set>.jsonl`, `<set>.judged.jsonl` |
| fit report and checks | `results_<run_id>/fit_report.json` |
| eval numbers | `results_<run_id>/eval/metrics.json` |
| RQ3 and transport analysis | `results_<run_id>/analysis.json` |
| tables | `results_<run_id>/summary.md` |

## Numerical checks (the fit stage fails if any is violated)

- **Exact OT identities:** `A_k m_j + b_k = m_r` and `A_k S_j A_k = S_r`.
- **PCA-OT mean:** the lifted map sends the jailbreak mean onto the refusal mean inside the subspace.
- **Affine form:** `U V^T h + b0` equals `P M psi(h)`.
- **Proposition 2:** holds on the training benign set.

The final SELOT field is reported but not asserted, because with beta and gamma > 0 it only moves the means
approximately:

- `field_mean_gap_left`: the fraction of the in-subspace jailbreak-to-refusal mean gap left after steering.
- `mean_gap_outside_subspace`: the part of the gap outside the subspace, which is never moved.

## Code

| path | content |
|---|---|
| `selot/selot_fit.py` | SELOT fit (subspace, OT target, soft/hard field, diagnostics) |
| `selot/gaussian_ot.py`, `selot/linalg.py` | closed-form Gaussian OT, W2, float32 PCA / matrix roots / weighted ridge |
| `selot/baselines.py` | baselines as affine maps + `fit_map` registry |
| `selot/affine_map.py` | per-layer `(s, D, U, V, b0)` storage, `apply`, `dense`, save/load |
| `selot/steering.py` | forward pre-hooks, position modes, layer-input capture |
| `selot/activations.py` | model loading, chat formatting, last-token / all-token extraction |
| `selot/judge.py`, `selot/metrics.py` | refusal matcher, Llama Guard, HarmBench, StrongREJECT and XSTest 3-class over-refusal judges; GSM8K / MMLU scoring, set -> metric |
| `selot/reward.py`, `selot/cipher.py` | reward model and win rates (AlpacaEval); CipherChat decoders (Cipher judging) |
| `selot/vllm_engine.py`, `scripts/vllm_worker.py` | vLLM subprocess (label responses, Llama Guard, HarmBench) |
| `selot/analysis.py` | displacement and transport statistics |
| `selot/fit_data.py` | per-model fit sets: behaviour labels, topic cap, val split, topic weights |
| `scripts/` | one entry point per stage; `build_fit_sets.py` + `topics.py` build the data; `pipeline.py` (server job), `make_val_sets.py`, `select_layer.py`, `compare_runs.py`, `data_manifest.py` |
| `configs/` | `base.yaml`, `models/`, `methods/`, `data/` |
