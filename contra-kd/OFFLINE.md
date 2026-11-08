# Offline GPT-2 +Contra t=0 — GPU 6 only

The original contra-kd checkout is unchanged. This copy contains source code,
not the original `data/`, Git metadata, environments, caches or logs.
Remote artifact publication and external experiment reporting have been removed
from the entire copy, including legacy scripts, dependencies, credentials and
configuration. Existing console and filesystem logs remain available.

## Provision before entering the offline machine

1. Use `../contra.txt` with the environment provisioner, like `opsd.txt`.
   Python 3.11 environment: `contra`, default `/mnt/local/uvenvs/contra`.
2. Use `download.txt` with the external downloader, like `opsd/download.txt`.
   It downloads complete snapshots of:
   - `Elriggs/openwebtext-100k`
   - `MiniLLM/dolly` (raw training records and held-out evaluation)
   - `MiniLLM/self-inst`
   - `MiniLLM/Vicuna`
   - `MiniLLM/sinst` (evaluation uses `11_/valid.jsonl`)
   - Teacher: `bachthetrollface/gpt2-1.5B-teacher-dolly`
   - Student init: `bachthetrollface/gpt2-120M-init-dolly`

No datasets from the original checkout are copied or used. The pipeline never
downloads resources or installs packages on the offline machine.

## Run

```bash
bash project_commands.sh
```

**Only physical GPU 6 is used**, including generation, both field pretraining
jobs, all four student methods and their evaluations. Jobs run **sequentially**:
generation → offline-Dolly field → DistiLLM2 field → DistiLLM + eval → FDD + eval
→ DistiLLM2 + eval → CSD + eval. CPU preprocessing hides all GPUs.
Every torchrun call starts one process. Logical CUDA device 0 inside each job
maps to physical GPU 6. Old per-stage GPU override variables are ignored.

## Path contract

`@PROJECT@` in `download.txt` must resolve to `PROJECT_NAME` in the launcher,
default `aiskylimit_new_nothing`. `BASE_DIR` defaults to
`/mnt/local/aiskylimit_new_nothing/contra-kd`.

| Data | Path relative to BASE_DIR |
| --- | --- |
| Downloaded OpenWebText snapshot | `data/raw/openwebtext-100k/data/train-*.parquet` |
| Downloaded Dolly training records | `data/raw/dolly/raw.jsonl` |
| Downloaded Dolly evaluation | `data/raw/dolly/valid.jsonl` |
| Downloaded Self-Instruct evaluation | `data/raw/self-inst/valid.jsonl` |
| Downloaded Vicuna evaluation | `data/raw/vicuna/valid.jsonl` |
| Downloaded SInst evaluation | `data/raw/sinst/11_/valid.jsonl` |
| Generated Dolly generation prompts | `data/prepared/dolly/{train,dev}.jsonl` |
| Converted OpenWebText text | `data/prepared/openwebtext/data.txt` |
| Tokenized Dolly | `processed_data/dolly/full/gpt2/` |
| Tokenized OpenWebText | `processed_data/openwebtext/gpt2/512/22.87K/` |
| Generated DistiLLM2 paired data | `data/generated/distillm2/gpt2/t0/` |
| Downloaded teacher checkpoint | `results/gpt2/train/sft/gpt2-xlarge/` |
| Downloaded student initialization | `results/gpt2/train/init/gpt2-base/` |

All active preprocessing, generation, field, training and evaluation scripts
use the exported absolute paths. There is no fallback to the old checkout's
`data/`. The code directory may be separate from BASE_DIR. Set `PROJECT_NAME`,
`BASE_DIR` or `ENV_DIR` before launching if your provisioned root differs.
TEACHER_PATH and STUDENT_PATH may override model locations explicitly.

MiniLLM/dolly has raw.jsonl and valid.jsonl but no train/dev files. The launcher
creates prompts from the downloaded raw.jsonl: first 1,000 records for dev,
remaining records for train, matching process_data_dolly.py. Evaluation uses
the independently downloaded valid.jsonl, never these training/dev files.
OpenWebText conversion preserves the original newline replacement and 250,000
record cap (the selected snapshot contains 100,000 records).

Model generation uses the existing Transformers fallback; vLLM is omitted to
avoid conflicting with the training Transformers pin. Hyperparameters and the
four-method t=0 experiment matrix are otherwise unchanged. Offline HF flags
are enabled. Metrics and checkpoints are stored locally. No NLTK download is needed.

Generated field/student checkpoints, evaluation results and logs are written
under the **code directory** in `results/` and `logs/gpt2_t0/`. Re-running repeats
generation and training using the same output paths; this is not auto-resume.
`runs/GPT2_T0.md` describes the original multi-GPU connected version; use this
file and project_commands.sh for this single-GPU offline copy.

Validation is static plus mocked end-to-end shell execution. No full dependency
installation, actual model/data provisioning or CUDA training is performed here.
