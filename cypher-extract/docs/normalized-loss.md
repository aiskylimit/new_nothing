# Optional task-normalized loss

The default configs keep the original token-mean objective. Task-normalized
loss is enabled only when a config sets `selector_loss_weight`.

The normalized presets are grouped by model family:

- `configs/llama3_normalized_loss`
- `configs/qwen2.5_normalized_loss`
- `configs/qwen3_normalized_loss`

Each folder mirrors every method in its base config folder and sets
`selector_loss_weight: 0.5`. Generator and selector losses are normalized
independently, then combined as:

```text
(1 - selector_loss_weight) * generator_loss
    + selector_loss_weight * selector_loss
```

This applies consistently to LM, KD, FDD, and HPD losses. If a local batch
contains only one task, that task keeps the full loss scale.

Normalized runs use a separate checkpoint tree so they cannot overwrite the
default baselines:

```text
results/normalized_loss/<model-family>/<method>
```

Train the matching normalized teacher first, then a student method. For
example:

```bash
RUN_GPUS=0,1 bash scripts/train.sh \
  configs/distillation/teacher_lora_qwen3_normalized_loss.yaml
RUN_GPUS=0,1 bash scripts/train.sh configs/qwen3_normalized_loss/fkl.yaml
```

To train the teacher and all 12 Qwen3 methods in sequence:

```bash
RUN_GPUS=0,1 bash scripts/train_all_qwen3_lora_normalized.sh
```

Inference can reuse the existing model-family and method names by changing
only the checkpoint root:

```bash
bash scripts/infer_all_qwen3_lora_normalized.sh
```

Pass `--methods fkl` to run only one method.

The Qwen 2.5 config directory is named `qwen2.5_normalized_loss`, while its
checkpoint model-family remains `qwen2.5_coder` for inference compatibility.
