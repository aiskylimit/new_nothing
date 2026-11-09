# Council-Weighted Reasoning Distillation (proposal_v2, svgd_func + key step)

Base: `Qwen/Qwen2.5-7B-Instruct`, data: s1K. Phase 1 trains M=3 LoRA experts (grad dropout + dual-ortho + svgd_func on key steps); Phase 2 merges them with TIES and distills a single LoRA student.

## Offline assets on B200

Install dependencies using root `cored.txt` and provision `cored/download.txt` on
the machine with internet before transferring to B200. The manifest includes
`models/Qwen2.5-7B-Instruct` and `models/DeepSeek-R1-Distill-Qwen-1.5B`.
All model, tokenizer, adapter and benchmark loaders use local files only;
missing assets raise an error without contacting the Hub.

Run commands from `cored/`. Relative model/dataset paths resolve against this
folder, or `CORED_ASSET_ROOT` when assets live elsewhere. Shared MATH500/AMC
snapshots resolve under `${CORED_ASSET_ROOT}/../palign/datasets/`, so preserve the
sibling folder layout. Generated training JSONL and output directories remain
relative to the working directory. Absolute local model/data paths also work.

Prepare the training data first:

```bash
python data/prepare_s1k.py --model_name models/Qwen2.5-7B-Instruct
```

To use DeepSeek, pass `--model_name models/DeepSeek-R1-Distill-Qwen-1.5B` to
preparation, key-step selection, Phase 1, council weights, distillation and
evaluation; prefix truncation uses `--model models/DeepSeek-R1-Distill-Qwen-1.5B`.
The shell launchers using a `MODEL` path accept an environment override, e.g.
`MODEL=models/DeepSeek-R1-Distill-Qwen-1.5B bash scripts/run_merge_distill_keystep.sh`.
Use separate processed data, key-step caches and output directories per backbone;
these artifacts depend on the tokenizer/model. The legacy ablation-metrics
launcher still uses its `MODEL=7b|8b` selector.

## Run on three B200 GPUs

The default entry point uses all three GPUs for preprocessing and both training
phases. Provision `cored.txt` and `download.txt`, then run from `cored/`:

```bash
bash project_commands.sh

# The second backbone gets its own data, caches, checkpoints and timings.
bash project_commands.sh --model models/DeepSeek-R1-Distill-Qwen-1.5B

# Optional final evaluation (n=3, configurable).
bash project_commands.sh --eval --eval_n 3
```

The launcher activates `/mnt/local/uvenvs/cored` when present; `CORED_VENV` or
`PY` can select a different provisioned environment. `--gpus 0,1,2` chooses the
three devices. This is a single-node launcher: run it once, not once per GPU.
`HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE` and `TRANSFORMERS_OFFLINE` stay enabled.

The accelerated execution preserves the main recipe's losses:

- Prefix selection, key-step entropy and council weighting run as three
  independent inference workers on disjoint rows. Prefix shards are merged in
  source order; exact cache ID coverage is checked before training continues.
- Phase 1 and distillation launch three data-parallel replicas using `torchrun`.
  Each Phase 1 replica has the whole council, preserving cross-expert loss terms.
  Only LoRA gradients are summed once per optimizer update; logits/activations
  are not transferred between GPUs. Orthogonal regularization is evaluated once
  per update, with the same coefficient as its per-example average.
- Global batch is **33**, epochs **3 per phase**, seed **42**. Each rank receives
  up to 11 examples per update, balanced by sequence length. Tokenization is
  cached in CPU RAM for all epochs. The final partial batch is used exactly
  once, normalized by its actual size; the LR schedule uses the real number of
  updates, `epochs * ceil(N / 33)`.
- Local examples are packed into microbatches with at most 11 sequences and
  **32,768 padded tokens**. Long sequences therefore use accumulation inside
  the global batch. If a microbatch runs out of memory, restore its LoRA
  gradients/RNG and retry smaller pieces, then lower that rank's token budget.
  There is no guarantee that 11 long CoTs fit in one forward pass.
- Exact NLL and full-vocabulary KL use chunked fp32 computation with analytical
  backward and bf16 logits. There is no top-k teacher approximation. Fused
  CUDA AdamW, bf16 autocast (with fp32 LoRA master weights/gradients), SDPA and
  non-reentrant gradient checkpointing are used by default.
  `--attn_implementation flash_attention_2` requires a compatible package already
  installed offline; SDPA requires no extra attention package or Hub kernel.
- Council weighting uses 2,048-token chunks and skips the extra expert-NLL cache:
  the student reads only cached `w`. Merge evaluation is omitted; `--eval` runs
  final-student evaluation after the timed training stages.

Tune memory/throughput using `--micro_batch_size`, `--max_batch_tokens` and
`--loss_chunk`; these can change when resuming the same run. For example, after
checking memory on the server, try `--max_batch_tokens 65536`. Disabling gradient
checkpointing with `--no_gradient_checkpointing` needs substantially more memory
and a new run ID. These defaults are not a measured B200 optimum.

Artifacts live in `outputs/b200/<model-name>_seed42/` by default. Select a new
`--run_id` for different method settings. Finished stages are reused only when
the model/tokenizer, source-data and implementation fingerprints match.

```bash
# Preflight the distributed Phase 1 loop for two optimizer updates.
# Prefix/key-step processing is still required; their results are reused later.
bash project_commands.sh --dry_run 2

# Continue the same run through both phases, using the saved optimizer/RNG state.
bash project_commands.sh

# Or select specific stages after their prerequisites exist.
bash project_commands.sh --stages weights merge distill
```

Training saves a resumable checkpoint after each epoch; `--checkpoint_every 5`
adds checkpoints every five updates at extra I/O cost. Interrupted stages rerun
with their valid cached rows; each rank restores its own RNG state for training.
Do not modify generated artifacts or copy caches across backbones/run IDs.

`timings.jsonl` records whole-stage wall time, including worker/model startup and
checkpoint I/O. Failed attempts and dry runs are labeled separately. Each
training phase also writes `timing-rank-*.json` with loop time, peak allocated
memory and OOM retries. `pipeline_config.json`, `invocations.jsonl`, and phase
`run_config.json` record settings and input provenance. A resumed invocation's
loop time covers that invocation, not earlier attempts.
`hardware.json` records the CUDA/PyTorch/NCCL versions and GPU names/capacities;
a short-lived probe checks three visible bf16-capable CUDA devices before GPU stages.

Local validation uses a tiny Qwen model on three CPU/Gloo workers: losses,
gradients, uneven final batches and resume are checked against single-process
execution. CUDA/NCCL kernels, memory fit and throughput still require B200 runs.

## Training method

The backbone stays frozen throughout training. Three LoRA experts share that
backbone, then one merged adapter acts as a fixed teacher for a single student.

1. **Prepare s1K.** Read local `simplescaling/s1K-1.1`, using
   `deepseek_thinking_trajectory` for reasoning and `deepseek_attempt` for the
   answer text. Split reasoning into paragraphs; retain `solution` as the gold
   answer for correctness checks. Write `data/train_s1k.jsonl`.
2. **Choose a reasoning prefix.** Try 15/30/45/60/80/100% of the reasoning steps.
   The frozen base generates four completions per candidate. Choose the shortest
   candidate in this grid with at least two correct completions; otherwise keep
   the full reasoning. Write `data/train_s1k_prefixed.jsonl`. This preprocessing
   uses max-new-tokens 1024, top-p 0.95 and generation seed 0.
3. **Cache key segments.** Compute base-model entropy per segment and select the
   highest 10% for the main recipe. `build_segments` includes reasoning steps and
   the final answer segment. This mask restricts only the diversity term in
   Phase 1. The standalone selector defaults to 7.5%; pass `--frac 0.1` explicitly.
4. **Phase 1: train three experts.** All experts learn response NLL, with an
   independent 10% segment dropout mask per expert/example. Add dual-orthogonal
   regularization on LoRA A/B subspaces (coefficient 0.01). Epoch 1 has these two
   terms; epochs 2–3 additionally repel expert predictions on cached key segments
   using the function-space RBF penalty (`svgd_func`, coefficient 1). Save
   `expert_0`, `expert_1`, `expert_2` under the Phase 1 output directory.
5. **Cache council weights.** Freeze the trained experts. For each reasoning
   step, compute mean expert entropy U and disagreement V = H(mean prediction)
   minus mean expert entropy, averaged over the step. The main launcher uses
   rho = V / max(U, epsilon), and weights
   `w = max(0.001, 1 + 0.5*tanh(zscore(U)) + 0.5*tanh(zscore(rho)))`,
   with z-scores within each example. The standalone weight script defaults to
   raw V instead, so pass `--disagreement_mode normalized` for this recipe.
   Cache U/V/w and optionally expert NLLs in one `.npz` per example. A diagnostic
   reports collapsed councils; stopping on failure requires `--fail_on_low_v`.
6. **Merge.** TIES trims each expert delta to its largest 20% entries, elects
   signs and combines agreeing entries. Refactor the result to rank-16 LoRA;
   save it under `merge_<tag>_ties/merged`. This step has no gradient training.
7. **Phase 2: distill one student.** Initialize both teacher and student from
   the merged adapter. Freeze the teacher and backbone; update only student
   LoRA. Minimize weighted step-mean NLL plus weighted step-mean
   `KL(teacher || student)` with coefficient 1. KL uses the full vocabulary.
   Reasoning steps use cached w; the final answer segment has weight 1.
   Save the final adapter under `distill_<tag>_ties/student`. Default inference
   needs only the backbone plus this student adapter.

| Setting | Phase 1 | Phase 2 |
|---|---|---|
| Epochs / seed | 3 / 42 | 3 / 42 |
| Learning rate | 5e-5 | 1e-4 |
| LoRA rank / alpha / dropout | 16 / 16 / 0.05 | Inherited from merged adapter |
| Cutoff length | 16,384 tokens | 16,384 tokens |
| Optimizer / scheduler | AdamW / cosine, 10% warmup | AdamW / cosine, 10% warmup |
| Global batch | 33 | 33 |
| GPU workers | 3 data-parallel replicas | 3 data-parallel replicas |
| Local microbatch | Up to 11, token-budget limited | Up to 11, token-budget limited |

For training-time reporting, record prefix selection, key-step selection,
Phase 1, council computation, merge and Phase 2 separately.

## Legacy single-process commands

The original modules below remain available for ablations. They default to
microbatch 1, accumulation 32 and one GPU. Use `project_commands.sh` above for
the accelerated three-GPU pipeline. The following commands use the same losses:

```bash
python data/prepare_s1k.py --model_name models/Qwen2.5-7B-Instruct
python -m proposal_v2.prefix_truncate --model models/Qwen2.5-7B-Instruct
python -m proposal_v2.keystep_select \
  --model_name models/Qwen2.5-7B-Instruct \
  --train_file data/train_s1k_prefixed.jsonl --cutoff_len 16384 \
  --frac 0.1 --out_dir outputs/proposal_v2/keystep_s1k_prefixed_10pct
python -m proposal_v2.phase1 \
  --config configs/proposal_v2/phase1_svgd_func_keystep_k1.yaml
python -m train.compute_weights \
  --model_name models/Qwen2.5-7B-Instruct \
  --train_file data/train_s1k_prefixed.jsonl --cutoff_len 16384 \
  --phase1_dir outputs/proposal_v2/phase1_svgd_func_keystep_k1 --M 3 \
  --weights_dir outputs/proposal_v2/weights_svgd_func_keystep_k1 \
  --disagreement_mode normalized
bash scripts/run_merge_distill_keystep.sh
```

The last script merges, evaluates the merge, distills, then evaluates the student;
it currently selects physical GPU 1 via `CUDA_VISIBLE_DEVICES=1`.

Phase 1 ablations: `configs/proposal_v2/phase1_svgd_func_keystep_k1_{allOff,noDropout,noOrtho,noSVGD,only*}.yaml`, run with `scripts/run_ablation_row.sh`.

## Results (Pass@1 / Pass@3, %, k=3, temp 0.6, top_p 0.9, max_tokens 4096)
| Model | AIME25 | AIME24 | AMC | MATH500 | Avg |
|---|--:|--:|--:|--:|--:|
| Zero-shot base | 7.8/16.7 | 10.0/16.7 | 43.4/56.6 | 76.3/85.4 | 34.37/43.84 |
| Expert 0 | 8.9/16.7 | 12.2/20.0 | 42.6/55.4 | 75.4/86.4 | 34.77/44.62 |
| Expert 1 | 11.1/20.0 | 11.1/20.0 | 44.2/62.7 | 76.2/86.4 | 35.65/47.26 |
| Expert 2 | 7.8/16.7 | 12.2/20.0 | 45.8/61.4 | 75.8/85.4 | 35.40/45.88 |
| Merge TIES (phi_merge) | 11.1/20.0 | 12.2/20.0 | 46.2/61.4 | 76.5/86.6 | 36.50/47.01 |
| Student, NLL only (lambda_KL=0) | 10.0/16.7 | 15.6/30.0 | 41.8/59.0 | 74.2/85.2 | 35.38/47.73 |
| **Student, full (lambda_KL=1)** | 11.1/16.7 | 14.4/23.3 | 45.8/59.0 | 77.2/86.4 | 37.13/46.36 |

AMC and MATH500 are read from the shared local snapshots in `palign/datasets/`;
AIME24 is read from `cored/datasets/aime_2024`, and AIME25 combines the two local
OpenCompass files `aime2025-I.jsonl` and `aime2025-II.jsonl` (30 unique problems).
Optional GSM8K and CurioSFT test files are also listed in `download.txt`.
ARC-Challenge and LogiQA2 downloads are provisioned for the planned OOD evaluation;
the standalone ARC/LogiQA2 loader/grading integration is not part of this change.
Raw per-problem scores from the original runs are in `results/*.json`.
