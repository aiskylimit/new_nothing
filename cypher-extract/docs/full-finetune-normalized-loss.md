# Full fine-tuning with task-normalized loss

These presets combine full-parameter fine-tuning with the optional multitask
loss normalization:

- `configs/llama3_full_finetune_normalized_loss`
- `configs/qwen2.5_full_finetune_normalized_loss`
- `configs/qwen3_full_finetune_normalized_loss`

Each folder mirrors all 12 student methods and sets both:

```yaml
finetuning_type: full
selector_loss_weight: 0.5
```

Generator and selector losses are normalized separately and then combined with
equal weight. Checkpoints are isolated from the other settings under:

```text
results/full_finetune_normalized_loss/<model-family>/<method>
```

Train the matching normalized full teacher before running a KD method:

```bash
RUN_GPUS=0,1 bash scripts/train.sh \
  configs/distillation/teacher_full_qwen3_normalized_loss.yaml
RUN_GPUS=0,1 bash scripts/train.sh \
  configs/qwen3_full_finetune_normalized_loss/fkl.yaml
```

To train the teacher and all 12 Qwen3 methods in sequence:

```bash
RUN_GPUS=0,1 bash scripts/train_all_qwen3_full_finetune_normalized.sh
```

Inference uses the same method name and the combined setting's checkpoint root:

```bash
bash scripts/infer_all_qwen3_full_finetune_normalized.sh
```

This wrapper includes `teacher_full` and all 12 student methods. Pass
`--methods fkl` to run only one method.
The Qwen 2.5 checkpoint family remains `qwen2.5_coder`.
