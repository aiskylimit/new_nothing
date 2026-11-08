# GPT-2 +Contra at t=0

These entrypoints run DistiLLM, FDD, DistiLLM2, and CSD with +Contra. Velocity-field pretraining and periodic updates both use `t=0` through `distillm/losses.py`. Standalone distillation baselines are disabled in these entrypoints.

Use the centralized launcher from the repository root:

```bash
TEACHER_REPO=your-org/your-teacher-sft \
STUDENT_REPO=your-org/your-student-init \
bash runs/gpt2/run_t0.sh .
```

Alternatively, replace the two dummy repository names at the top of `runs/gpt2/run_t0.sh`. The launcher stops before installation if either dummy name remains. It installs the repository's training dependencies with the existing Transformers/PEFT versions, then downloads both public checkpoints using `hf download ... --local-dir ...` ([HF CLI documentation](https://huggingface.co/docs/huggingface_hub/guides/cli#download-to-a-local-folder)). The installed CUDA PyTorch build is retained when it satisfies the dependencies. vLLM is optional: the existing generator uses it when available and otherwise falls back to Transformers.

The launcher runs these phases in order:

1. Install dependencies.
2. Download the shared teacher SFT and student initialization checkpoints.
3. Download raw OpenWebText if missing, then preprocess OpenWebText and full Dolly exactly once using direct `python3 -u tools/process_data_*.py` commands.
4. Generate and format DistiLLM2 response pairs on one GPU.
5. Pretrain the shared offline-Dolly field and the separate DistiLLM2 field concurrently, then wait for both to succeed.
6. Train the four methods concurrently. Each method then evaluates its own student on the same GPU.

| Job | Default GPU | GPU variable | Training port | Evaluation port |
| --- | --- | --- | --- | --- |
| DistiLLM2 response generation | 0 | `GENERATE_GPU` | — | — |
| Shared offline-Dolly field | 0 | `OFFLINE_FIELD_GPU` | 20120 | — |
| DistiLLM2 field | 1 | `DISTILLM2_FIELD_GPU` | 20121 | — |
| DistiLLM +Contra | 0 | `DISTILLM_GPU` | 20130 | 20140 |
| FDD +Contra | 1 | `FDD_GPU` | 20131 | 20141 |
| DistiLLM2 +Contra | 2 | `DISTILLM2_GPU` | 20132 | 20142 |
| CSD +Contra | 3 | `CSD_GPU` | 20133 | 20143 |

Set these GPU variables to any available GPU IDs on the training machine, keeping simultaneous jobs on distinct GPUs. For example, `DISTILLM_GPU=4 FDD_GPU=5 DISTILLM2_GPU=6 CSD_GPU=7` moves student training to GPUs 4–7. `BASE_PORT` defaults to `20120`; field, training, and evaluation jobs use distinct offsets from it. All jobs use one GPU each.

Preprocessing uses the downloaded student tokenizer locally, disables tokenizer parallelism, defaults CPU thread counts to one, and uses `DATA_PROCESS_WORKERS=4` instead of 32. Set `DATA_PROCESS_WORKERS=1` for fewer worker processes. The prompt-only Dolly pass is omitted because these experiments only consume the full dataset. OpenWebText processing terminates and joins its encoding pool after collecting enough instances, discarding queued work; Dolly processing joins its pool after finishing. These changes address worker teardown and excess parallelism without claiming to eliminate every possible stall.

Logs go to `logs/gpt2_t0/` by default (`LOG_DIR` overrides it). Preprocessing and generation logs also stream to the terminal. The two field jobs write `field_offline.log` and `field_distillm2.log`; the four experiments write `distillm.log`, `fdd.log`, `distillm2.log`, and `csd.log`. The launcher checks each background job's exit status and terminates remaining background process groups if a job fails or the launcher is interrupted.

The four individual experiment entrypoints now perform only student training and evaluation. They require the downloaded checkpoints, preprocessed data, generated DistiLLM2 data, and pretrained fields. They do not repeat installation, preprocessing, generation, or field pretraining. After a successful setup phase, they can also be launched individually with `CUDA_VISIBLE_DEVICES` and a separate `EVAL_PORT` for each concurrent job.

The shared field/projector files are initial checkpoints. Periodic updates happen independently in each experiment's memory and do not overwrite the shared pretrained pair.

## Inputs

All paths below are relative to `BASE_PATH`.

| Input | Path |
| --- | --- |
| Downloaded student initialization | `results/gpt2/train/init/gpt2-base/` |
| Downloaded teacher SFT | `results/gpt2/train/sft/gpt2-xlarge/` |
| Raw Dolly for preprocessing | `data/dolly/raw.jsonl` |
| Dolly prompts for DistiLLM2 generation | `data/dolly/train.jsonl`, `data/dolly/dev.jsonl` |
| Raw OpenWebText, downloaded if missing | `data/openwebtext/data.txt` |
| Processed Dolly, produced before training | `processed_data/dolly/full/gpt2/` |
| Processed OpenWebText, produced before training | `processed_data/openwebtext/gpt2/512/22.87K/` |
| Dolly evaluation | `data/dolly/valid.jsonl` |
| Self-Instruct evaluation | `data/self-inst/valid.jsonl` |
| Vicuna evaluation | `data/vicuna/valid.jsonl` |
| SInst evaluation | `data/sinst/11_/valid.jsonl` |

Download the complete Hugging Face checkpoints, including `config.json`, model weights, and tokenizer files, into the two checkpoint directories above. The launcher and experiment entrypoints check for each `config.json` before training. To use different absolute locations, set `STUDENT_PATH` and `TEACHER_PATH` in the environment for every run, for example:

```bash
STUDENT_PATH=/path/to/downloaded/student \
TEACHER_PATH=/path/to/downloaded/teacher \
CUDA_VISIBLE_DEVICES=0 bash runs/gpt2/distillm_contra.sh .
```

All four methods use these same downloaded checkpoints, including DistiLLM2 response generation. DistiLLM, FDD, and CSD load the same pretrained offline-Dolly field/projector. DistiLLM2 field pretraining and student training both consume `data/distillm2/gpt2/t0/formatted/`; its validation data comes from processed Dolly.

## Outputs

The downloaded teacher SFT and student initialization checkpoints are inputs. Generated field/projector checkpoints and distilled students use the t=0 output directories below.

| Pretrained field/projector | Directory |
| --- | --- |
| DistiLLM, FDD, CSD (shared offline Dolly) | `results/gpt2/train/velocity_field_t0/` |
| DistiLLM2 (generated response pairs) | `results/gpt2/train/velocity_field_t0/distillm2/` |

Both directories contain `velocity_field.pth` and `projector.pth`. The centralized launcher pretrains both pairs on every invocation; individual experiment entrypoints only load them. Re-running the launcher reuses its output paths.

| Method | Final student checkpoint | Evaluation directory |
| --- | --- | --- |
| DistiLLM | `results/gpt2/train/contra_t0/distillm/0.1B_1.5B/` | `results/gpt2/eval_main/contra_t0_distillm_0.1B_1.5B/` |
| FDD | `results/gpt2/train/contra_t0/fdd/base/` | `results/gpt2/eval_main/contra_t0_fdd_base/` |
| DistiLLM2 | `results/gpt2/train/contra_t0/distillm2/0.1B_1.5B/` | `results/gpt2/eval_main/contra_t0_distillm2_0.1B_1.5B/` |
| CSD | `results/gpt2/train/contra_t0/csd/0.1B_1.5B/` | `results/gpt2/eval_main/contra_t0_csd_0.1B_1.5B/` |

Evaluation uses only Dolly, Self-Instruct, Vicuna, and SInst, with seeds `10 20 30 40 50`. Results are nested under `{data_name}-512/{seed}/` within each evaluation directory (`data_name` is `dolly`, `self_inst`, `vicuna`, or `sinst_11_`). All results stay on the local filesystem. Re-running an entrypoint reuses its t=0 output paths.

## Verification

Raw Dolly and the four evaluation datasets were present in this checkout. The centralized launcher has been checked with mocked dependency installation, downloads, preprocessing, generation, field training, and student training/evaluation to verify stage order, GPU/port assignments, paths, parallelism, one preprocessing pass per dataset, and failure propagation. Preprocessor worker cleanup was checked with real multiprocessing workers and lightweight encoder/builder substitutes. Shell and Python syntax checks passed. Dependencies, public model downloads, actual dataset tokenization, and GPU training have not been run here.
