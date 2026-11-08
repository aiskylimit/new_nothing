# CSD Quickstart

Run all commands from the repository root.

## 1. Setup dependencies and data

```bash
bash runs/setup.sh .
```

This installs dependencies, downloads the required data, preprocesses OpenWebText and Dolly, and downloads the velocity-field checkpoints.

## 2. Set GPU devices

Each CSD wrapper uses one GPU through its `DEVICE` variable:

```bash
DEVICE=0 # set gpu id here, 1 gpu only
```

Before running jobs in parallel, edit each script under `runs/csd/` so every script uses a different GPU, for example:

- `runs/csd/gpt2.sh`: `DEVICE=0`
- `runs/csd/openllama2.sh`: `DEVICE=1`
- `runs/csd/openllama2_contra.sh`: `DEVICE=2`
- `runs/csd/llama2.sh`: `DEVICE=3`
- `runs/csd/llama2_contra.sh`: `DEVICE=4`

## 3. Run CSD jobs

GPT-2 CSD:

```bash
bash runs/csd/gpt2.sh .
```

OpenLLaMA-2 CSD:

```bash
bash runs/csd/openllama2.sh .
```

OpenLLaMA-2 CSD + Contra-KD:

```bash
bash runs/csd/openllama2_contra.sh .
```

LLaMA-2 CSD:

```bash
bash runs/csd/llama2.sh .
```

LLaMA-2 CSD + Contra-KD:

```bash
bash runs/csd/llama2_contra.sh .
```

To launch all five in parallel:

```bash
bash runs/csd/gpt2.sh . &
bash runs/csd/openllama2.sh . &
bash runs/csd/openllama2_contra.sh . &
bash runs/csd/llama2.sh . &
bash runs/csd/llama2_contra.sh . &
```

Outputs are written under `results/{model}/train/...` and `results/{model}/eval_main/...`.
