# P-ALIGN hyperparameters vs paper (Qwen2.5-7B-Instruct)

Paper: [Long-Chain Reasoning Distillation via Adaptive Prefix Alignment](https://arxiv.org/pdf/2601.10064).
Values marked **paper** are stated in the paper. Values marked **assumed** are not published and are set in `configs/qwen2.5_7b_palign_sft.yaml` / `project_commands.sh`.

## Match check

| Item | Paper | This repo | Match |
|---|---|---|---|
| Student | Qwen2.5-7B-Instruct (also Qwen3-8B) | `Qwen/Qwen2.5-7B-Instruct` | yes |
| Teacher (Long-CoT) | DeepSeek-R1 | data already in `data/palign_sft_qwen2.5-7b.json` | n/a (offline data) |
| Method | SFT + LoRA | `finetuning_type: lora` | yes |
| Framework | TRL + LLaMA-Factory | LLaMA-Factory `src/train.py` | yes |
| Epochs | 3 | `num_train_epochs: 3.0` | yes |
| Learning rate | \(5 \times 10^{-5}\) | `5.0e-5` | yes |
| Train set | 1,000 from s1K-1.1, Eq.9-filtered → 966 | local JSON, 966 rows | yes (file on disk) |
| Loss | response only (Eq. 2) | `train_on_prompt: false` | yes |
| Logging | — | `report_to: none` | no external logger |
| Eval sets | AIME24, AIME25, AMC12/AMC23, MATH-500 | `data/raw/{aime24,aime25,amc12,math500}.jsonl` | yes |
| Eval metrics | pass@1, pass@3 | `n=3` all sets; Pass@1 = mean of k samples; Pass@3 = any-correct; Avg = equal-weight mean of 4 sets | yes (right-column eval) |
| Sampling T | 0.6 | `temperature=0.6` | yes |
| Sampling top-p | 1.0 | `top_p=0.9` | eval uses 0.9 |
| `repetition_penalty` | not stated | `1.05` | eval default |
| Samples / problem | AIME/AMC 32, MATH500 8 | `k=3` all benchmarks | eval uses k=3 |
| Max response | 32768 | `--max_tokens 4096` | eval uses 4096 |
| LoRA rank | 16 | `lora_rank: 16` | yes |
| LoRA alpha | 16 | `lora_alpha: 16` | yes |
| LoRA dropout | 0.05 | `lora_dropout: 0.05` | yes |
| LoRA targets | q/k/v/o + gate/up/down | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | yes |
| Effective batch | 32 samples/step | `per_device=1 × grad_accum=32` | yes |
| Optimizer | AdamW, β=(0.9, 0.999), eps default, wd=0 | `adamw_torch`, same β/eps/wd | yes |
| Scheduler | cosine + warmup, warmup_ratio 0.1 (LambdaLR) | `lr_scheduler_type: cosine`, `warmup_steps: 0.1` | yes |
| Max sequence length | 32768 | `cutoff_len: 32768` | yes |
| Precision | not stated | bf16 | assumed |
| Grad checkpoint | not stated | `true` | assumed |

## Paper Table 1 (P-ALIGN, Qwen2.5-7B-Instruct) — reference only

| Metric | AIME25 | AIME24 | AMC12 | MATH500 | Avg. |
|---|---|---|---|---|---|
| Pass@1 | 16.67 | 16.67 | 49.40 | 75.80 | 39.64 |
| Pass@3 | 26.67 | 26.67 | 63.86 | 85.20 | 50.60 |

AMC: Table 3 says AMC23; Table 1 / `data/raw` use AMC12 (`AI-MO/aimo-validation-amc`, 83 problems).

## Commands file

All install / data / train / eval commands: `project_commands.sh`.
Eval table is written to `output/eval_results.txt` (pass@1, pass@3, average).
