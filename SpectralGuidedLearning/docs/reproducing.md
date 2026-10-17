# Reproducing the published numbers

All runs evaluate with P-ALIGN's protocol (`configs/common/eval/palign-protocol.yaml`): thinking off,
AIME24 / AIME25 / AMC12 / MATH500, n = 3, T = 0.6, top-p 0.9, repetition penalty 1.05, 4096 tokens,
math_verify grading. Results go to `results/<run>/summary.json`; `--stages compare` rebuilds
`results/comparison-table.md`.

## Main track: DeepSeek-R1-Distill-Qwen-1.5B on P-ALIGN's data (full fine-tuning)

| Row | Command |
|---|---|
| P-ALIGN (paper's own trainer) | `palign run r1-qwen-1.5b --stages reference_data,train_reference` then `palign run r1-qwen-1.5b --stages eval run_suffix=-llamafactory` |
| P-ALIGN, native trainer, P-ALIGN's exact tokens and schedule | `palign run r1-qwen-1.5b` |
| P-ALIGN data through the SGL recipe (NLL; eval-style prompt, lr to 1e-5) | `sgl run r1-qwen-1.5b-palign/sft-nll` |
| Segment-Selective SFT | `ssft run r1-qwen-1.5b` |
| SGL (spectral selection) | `sgl run r1-qwen-1.5b-palign/sgl-spectral` |
| IWC (gate + entropy) | `sgl run r1-qwen-1.5b-palign/iwc-stable` |
| Entropy allocation | `sgl run r1-qwen-1.5b-palign/iwc-nogate-l05` |
| Shuffled control | `sgl run r1-qwen-1.5b-palign/iwc-shuf-l05` |
| Answer-gain allocation | `sgl run r1-qwen-1.5b-palign/iwc-gain-l05` |
| Provenance arms | `sgl run r1-qwen-1.5b-palign/prov-nll-dft` (`-j8`, `prov-dft-nll`), `sft-dft` |

Seeds: training seeds 43/44 are `name=<arm>-s43 train_seed=43` (the published runs are named
`<arm>-s<seed>`); sampling seeds 43/44 of a trained checkpoint are
`--stages eval eval_seed=43 eval_suffix=-e43 results_dir=results_evalseed`.

## LoRA tracks: Qwen2.5-7B-Instruct and Qwen3-8B on P-ALIGN's data

`sgl run qwen25-7b-palign/{sft-nll-lora,iwc-nogate-l05-lora,iwc-gain-l05-lora}`, same for
`qwen3-8b-palign/`. P-ALIGN's own LoRA runs: `palign run qwen25-7b-lora`, `palign run qwen3-8b-lora`.

## Rebuilding P-ALIGN's training data

`palign run qwen25-7b-from-scratch` regenerates the prefix-aligned traces with Qwen2.5-7B-Instruct
as judge and continuation writer (`data/palign-build/qwen25-7b/`), then trains and evaluates on
them. Sampling makes the rebuilt file differ from the released one row by row.

## Multi-GPU

`run.gpus=[0,1,2,3]` gives torchrun stages (training) one process per GPU, with gradient accumulation
recomputed to keep the effective batch at 32; single-process stages use the first GPU. Spectral
capture can be sharded by hand: `--stages capture stages.capture.args.num-shards=4
stages.capture.args.shard-index=<i> stages.capture.gpus=[<i>] --force`, then one plain
`--stages capture --force` to merge.

## Legacy drivers

The shell drivers that produced the published numbers (`project_commands*.sh`, `scripts/`) still
work and now call the package (`python -m sgl.…`). Tracks not covered by configs (the s1K-1.1
Qwen2.5/Qwen3 runs, R1-Distill-Qwen-7B on LIMO, the Qwen3-8B L_trans sweep, the Unsloth spectral
runs) remain reproducible through them.
