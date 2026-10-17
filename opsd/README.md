# Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models


<p align="center">
<a href="https://arxiv.org/pdf/2601.18734v3"><img src="https://img.shields.io/badge/arXiv-2601.18734-b31b1b.svg"></a>
<a href="https://siyan-zhao.github.io/blog/2026/opsd/"><img src="https://img.shields.io/badge/Blog-Post-blue.svg"></a>
</p>

---
## Overview

**On-Policy Self-Distillation (OPSD)** trains a single model to act as both student and teacher by conditioning on different contexts — the student sees only the problem, while the teacher additionally sees the ground-truth solution — and performs token-level distribution matching along the student's own on-policy trajectories.


## Updates

- **Mar 18, 2026**: Released updated code. 

  (1) Fixed chat template and zero2 bugs (see [template issue](https://github.com/huggingface/trl/issues/5241)), we re-ran experiments with updated results (detailed results & ablations updated on arxiv/blog). The fixes yield improved OPSD performance, most notably on Qwen3-1.7B.

  (2) Added a new training stabilization strategy 🚀: per-token point-wise KL clipping. We find style tokens (such as 'wait', 'think') can exhibit 6–15× higher KL divergence than math-related tokens, and dominates the training signal. Clipping stablizes training and improves performance.


-  **Mar 3, 2026**: Initial code release.

## Installation


```bash
bash setup_env.sh
source /mnt/local/uvenvs/opsd/bin/activate
```
Training uses PyTorch SDPA for attention.

The code uses `trl`'s experimental GOLD trainer as a base.

## Repository Structure

```
├── opsd_trainer.py          # OPSDTrainer: core self-distillation trainer
├── data_collator.py         # Data collator for self-distillation
├── opsd_train.py            # OPSD training entry point
├── sft_train.py             # SFT baseline training entry point
├── grpo_train.py            # GRPO baseline training entry point
├── accelerate.yaml          # Accelerate/DeepSpeed ZeRO-2 config
├── requirements.txt         # Pinned Python packages
├── setup_env.sh             # Create the Python environment
├── experiment_settings.env  # Reproduction settings and explicit overrides
├── download.txt             # Models and datasets required by the offline server
├── project_commands.sh      # Prepare, train, evaluate, and aggregate
├── scripts/
│   ├── run_training.sh      # Shared SFT/GRPO/OPSD launcher
│   ├── run_sft.sh           # SFT wrapper
│   └── run_grpo.sh          # GRPO wrapper
└── eval/
    ├── evaluate_math.py     # Evaluation script (vLLM)
    ├── run_eval_matrix.sh   # Full checkpoint/benchmark matrix
    └── summarize_results.py # CSV/JSON aggregation
```

## Offline Reproduction

The training and evaluation workflow does not access hosted models, datasets, experiment trackers, or Hub uploads. Download the assets listed in `download.txt` before moving the project to the server. `project_commands.sh` creates `/mnt/local/uvenvs/opsd`, installs the pinned dependencies, and then runs the full workflow. The initial environment setup requires package-index access or a populated Python wheel cache.

```bash
bash project_commands.sh
```

This prepares the datasets, runs each SFT, GRPO, and OPSD job for Qwen3-4B and Qwen3-8B, and evaluates that job before starting the next one. The evaluation command writes updated result summaries after each job. `math-verify==0.8.0` is pinned in `environment.yml` and used to grade all benchmarks including HMMT25.

## Experiment Settings

The authoritative executable configuration is `experiment_settings.env`.

| Setting | Qwen3-4B | Qwen3-8B |
|---|---:|---:|
| GPUs | 2 | 2 |
| Per-GPU batch | 8 | 4 |
| Gradient accumulation | 2 | 4 |
| Effective batch | 32 | 32 |
| Learning rate | 5e-6 | 5e-6 |
| LoRA rank / alpha | 64 / 128 | 64 / 128 |

GRPO uses a smaller per-GPU microbatch of 2 with 8 gradient-accumulation steps on two GPUs (effective batch 32). This bounds the memory used to score its 8 long generations per prompt while retaining the 16,000-token completion limit. SFT and OPSD keep the per-GPU batch sizes above.

Paper-aligned method settings:

- SFT: 100 steps and maximum sequence length 16,000.
- GRPO: 500 steps, 8 generations per prompt, maximum completion length 16,000, and reference KL beta 0.
- OPSD: 100 steps, one generation per prompt, maximum completion length 1,024, forward KL, full vocabulary, fixed initial-policy teacher, student thinking off, and teacher thinking on.
- Checkpoint evaluation: SFT at step 100; GRPO at steps 400, 425, 450, and 500; OPSD at steps 25, 50, 75, and 100. GRPO checkpoints are saved every 25 steps so the requested checkpoints exist. The paper evaluates OPSD every 20 steps, does not state the GRPO interval, and reports GRPO peak performance within 500 steps.
- Evaluation: thinking mode, 38,912 maximum new tokens, and 12 solutions per problem.
- Active benchmarks: AIME25, HMMT25, plus the requested MathArena AIME26 extension. AIME24 is retained as a commented option in `download.txt` and `eval/run_eval_matrix.sh`.

Explicit user overrides relative to the paper are training/evaluation `temperature=1.0`, `top_p=1.0`, and disabled top-k (`top_k=0`). The custom OPSD and evaluation paths convert disabled top-k to vLLM's `-1`; TRL GRPO accepts `0` directly as disabled. The paper reports OPSD temperature 1.1, GRPO temperature 1.2, and evaluation top-p 0.95. The paper does not disclose the pointwise clipping threshold; this workflow preserves the released main-run values of 0.05 for 4B and 0.06 for 8B.

Results are written locally under `results/`. `results_table.md` contains the complete model/method/checkpoint table and is also printed after evaluation. `best_results.csv` selects checkpoints by `selected_paper_average` over the currently enabled paper tasks AIME25 and HMMT25; `extended_average` additionally includes AIME26.

## Historical Upstream Results

The following tables are retained from the upstream repository for reference. The executable offline workflow in this fork supports Qwen3-4B and Qwen3-8B.

### Evaluation Results across Tasks on Qwen3-1.7B

## Thinking Mode Eval:

<div align="center">
<table>
<tr>
<th align="center">AIME24</th>
<th align="center">AIME25</th>
<th align="center">HMMT25</th>
</tr>
<tr>
<td>

| Step | Avg@12 |
|---|---|
| Base | 51.5% |
| 25 | 51.4% |
| 50 | 52.8% |
| 75 | 54.4% |
| 100 | 57.2% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 36.7% |
| 25 | 42.5% |
| 50 | 43.9% |
| 75 | 40.6% |
| 100 | 41.1% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 23.1% |
| 25 | 24.7% |
| 50 | 27.8% |
| 75 | 26.9% |
| 100 | 29.2% |

</td>
</tr>
</table>
</div>

> **Evaluation settings:** temperature=1.0, thinking mode enabled, max new tokens=38912, top-p=none, top-k disabled, min-p=0, presence penalty=0, num samples=12

**Reproducibility note:** The results above report Avg@12 using a single seed run, so some variation across runs is expected. We acknowledge that multi-seed evaluation should be adopted and more reliable. For reference, the authors of [OP²SD](https://github.com/MBZUAI-reasoninglab/OP2SD#results-snapshot) have independently evaluated OPSD across 4 decoding seeds; their results may serve as a helpful reference.


## Non-Thinking Mode

OPSD can also run in non-thinking setting where both the Qwen student and teacher are enabled_thinking=False during training (`--student_thinking False --teacher_thinking False`) and evaluated with non-thinking inference (`--no_thinking`), with faster evaluation time than thinking mode.

The non-thinking numbers below are historical upstream ablations and are not part of `project_commands.sh`.

### Evaluation Results with Non-Thinking Mode across Models

#### Qwen3-8B (`--jsd_token_clip 1e-7`)

<div align="center">
<table>
<tr>
<th align="center">AIME24</th>
<th align="center">AIME25</th>
<th align="center">HMMT25</th>
</tr>
<tr>
<td>

| Step | Avg@12 |
|---|---|
| Base | 26.4% |
| 50 | 49.7% |
| 75 | 45.3% |
| 100 | 38.3% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 19.7% |
| 50 | 35.0% |
| 75 | 26.9% |
| 100 | 27.5% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 10.8% |
| 50 | 18.3% |
| 75 | 17.5% |
| 100 | 15.3% |

</td>
</tr>
</table>
</div>

#### Qwen3-4B (`--jsd_token_clip 1e-6`)

<div align="center">
<table>
<tr>
<th align="center">AIME24</th>
<th align="center">AIME25</th>
<th align="center">HMMT25</th>
</tr>
<tr>
<td>

| Step | Avg@12 |
|---|---|
| Base | 23.1% |
| 50 | 20.3% |
| 75 | 27.5% |
| 100 | 31.1% |
| 150 | 32.8% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 21.4% |
| 50 | 21.4% |
| 75 | 20.8% |
| 100 | 21.1% |
| 150 | 21.9% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 10.8% |
| 50 | 11.1% |
| 75 | 13.1% |
| 100 | 16.4% |
| 150 | 14.4% |

</td>
</tr>
</table>
</div>

#### Qwen3-1.7B (`--jsd_token_clip 1e-6`)

<div align="center">
<table>
<tr>
<th align="center">AIME24</th>
<th align="center">AIME25</th>
<th align="center">HMMT25</th>
</tr>
<tr>
<td>

| Step | Avg@12 |
|---|---|
| Base | 11.9% |
| 50 | 15.0% |
| 75 | 13.9% |
| 100 | 12.5% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 9.2% |
| 50 | 6.2% |
| 75 | 8.3% |
| 100 | 8.1% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 5.0% |
| 25 | 7.2% |
| 50 | 5.8% |
| 75 | 5.0% |

</td>
</tr>
</table>
</div>

> **Evaluation settings:** temperature=1.0, non-thinking mode, num samples=12.



## Key OPSD arguments

| Argument | Default | Description |
|---|---|---|
| `--fixed_teacher` | `False` | Fix the teacher to the initial policy (step 0). Requires --use_peft. Note ❗ If you disable PEFT, the teacher will keep updating at every training step, which may make training unstable. Our main results use the fixed teacher, which is currently implemented with LoRA adapter weights. |
| `--use_tinker_loss` | `False` | Use sampled-token policy-gradient objective instead of full-vocabulary JSD. More memory efficient. Currently no clipped implemented for this variant, could be unstable. |
| `--max_completion_length` | — | Student generation length for distillation. We use 1024 in our main experiments. |
| `--beta` | — | Interpolation weight for the JSD mixture distribution. Beta=0 means forward KL and 1 means reverse KL. |
| `--jsd_token_clip` | 0.05 | Clip the point-wise JSD loss contributions to a maximum value before summing over vocabulary (i.e., clipping is applied point-wise, not to the vocabulary-summed loss). This can improve stability by preventing stylistic tokens from dominating the training signal. Note when clipping is applied, the loss can be negative due to positive KL summand being capped. | 
| `--reason_first` | `False` | Prepend an explicit rationalization to the teacher context before distillation. |
| `--run_config` | `None` | Custom local name suffix for the output directory. |

### SFT Baseline

See [`scripts/run_sft.sh`](scripts/run_sft.sh).

### GRPO Baseline

See [`scripts/run_grpo.sh`](scripts/run_grpo.sh).

### Acknowledgements
Our implementation builds on [TRL GOLD Trainer](https://huggingface.co/docs/trl/gold_trainer). We sincerely thank [@simran135](https://github.com/simran135) and [@beanie00](https://github.com/beanie00) for identifying the prompt template bugs and the zero-2 issue, respectively!

## Citation
If you find this useful, please consider citing:
```bibtex
@article{zhao2026self,
  title={Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models},
  author={Zhao, Siyan and Xie, Zhihui and Liu, Mengchen and Huang, Jing and Pang, Guan and Chen, Feiyu and Grover, Aditya},
  journal={arXiv preprint arXiv:2601.18734},
  year={2026}
}
```
