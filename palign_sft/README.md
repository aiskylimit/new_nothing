# SFT baselines for P-ALIGN

This folder reproduces the two supervised fine-tuning baselines around
[NEUIR/P-ALIGN](https://github.com/NEUIR/P-ALIGN) on both supported student backbones:

| Run target | Backbone | Supervision from [`simplescaling/s1K-1.1`](https://huggingface.co/datasets/simplescaling/s1K-1.1) |
|---|---|---|
| `qwen25_label` | `Qwen/Qwen2.5-7B-Instruct` | `solution` |
| `qwen25_longcot` | `Qwen/Qwen2.5-7B-Instruct` | `deepseek_thinking_trajectory` |
| `qwen3_label` | `Qwen/Qwen3-8B` | `solution` |
| `qwen3_longcot` | `Qwen/Qwen3-8B` | `deepseek_thinking_trajectory` |


## Run

First provision every entry in `download.txt` and the Python environment from
the repository-level `p-align.txt`. The latter already contains all required
packages, so it does not need a baseline-specific dependency.

Run all four experiments serially from data preparation through LoRA training,
merge, inference, scoring, and report generation:

```bash
bash project_commands.sh
```

Run one experiment:

```bash
bash project_commands.sh all qwen25_label
bash project_commands.sh train qwen3_longcot
bash project_commands.sh eval qwen3_longcot
```

Available stages are `all`, `env`, `data`, `train`, and `eval`. Available
targets are `all`, `qwen25_label`, `qwen25_longcot`, `qwen3_label`, and
`qwen3_longcot`. `env` and `data` ignore the target selector.

Useful overrides include `ASSET_ROOT`, `DATA_DIR`, `PALIGN_VENV`,
`CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`, `EFFECTIVE_BATCH`,
`TENSOR_PARALLEL_SIZE`, `MAX_MODEL_LEN`, and `MODEL` (the last one is valid
only when evaluating a single target).

For multi-GPU training, set the visible devices and process count together:

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash project_commands.sh train qwen3_longcot
```

The default exposes only GPU 0.
The end-to-end orchestrator is intentionally single-node (`NNODES=1`) so that
only one process performs adapter export and evaluation.
Training configs intentionally use `overwrite_output_dir: true`, matching
`P-ALIGN`, so rerunning `train` starts that selected run again rather than
resuming an interrupted checkpoint.

Every run writes into its own directory below `output/`; for example,
`output/qwen3-8b-longcot/{lora,merged,result,eval_results.txt}`.

## Reproducibility notes

- `project_commands.sh data` requires exactly 1,000 valid examples in both
  target fields by default. Override `EXPECTED_TRAIN_ROWS` only when
  deliberately using a different dataset revision.
- Full training and four merged checkpoints require substantial GPU time, CPU
  RAM, and disk. Runs are serial by default so they do not contend for a GPU.
