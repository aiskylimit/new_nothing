# Optional full fine-tuning

The default KD method folders keep LoRA student/teacher training (their SFT
baseline is already full). Full fine-tuning for every method is available as a
separate, non-normalized setting in:

- `configs/llama3_full_finetune`
- `configs/qwen2.5_full_finetune`
- `configs/qwen3_full_finetune`

Each folder contains the same 12 student methods as its default model-family
folder. These presets use `finetuning_type: full`, a learning rate of `2e-5`,
and write to:

```text
results/full_finetune/<model-family>/<method>
```

Train the matching full teacher before a KD method. For example:

```bash
RUN_GPUS=0,1 bash scripts/train.sh \
  configs/distillation/teacher_full_qwen3.yaml
RUN_GPUS=0,1 bash scripts/train.sh configs/qwen3_full_finetune/fkl.yaml
```

To train the teacher and all 12 Qwen3 methods in sequence:

```bash
RUN_GPUS=0,1 bash scripts/train_all_qwen3_full_finetune.sh
```

The KD presets load the local teacher checkpoint through `ref_model`; they do
not use `ref_model_adapters`. LoRA and normalized-loss presets remain unchanged.

Student inference uses the existing method names with a different checkpoint
root:

```bash
bash scripts/infer_all_qwen3_full_finetune.sh
```

This wrapper includes `teacher_full` and all 12 student methods. Pass
`--methods fkl` to run only one method.
The Qwen 2.5 config directory omits `_coder` for consistency with the normalized
presets, while its checkpoint model-family remains `qwen2.5_coder`.
