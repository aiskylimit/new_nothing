# Reference and declared differences

Reference repository: [songmzhang/DSKDv2](https://github.com/songmzhang/DSKDv2), inspected at commit `ae4993e8b857ca001742de651498dc627138dd32`.
Reference paper: [A Dual-Space Framework for General Knowledge Distillation of Large Language Models](https://arxiv.org/html/2504.11426v1), Table I and section IV-A.

The public commit supplies GPT2 Dolly launchers, not a complete TinyLLaMA OPD launcher or the exact fine-tuned LLaMA2 teacher used for its result table. This project implements the requested **ordinary OPD vs ordinary OPD + IMPACT** experiment using the closest documented TinyLLaMA training setup. It is not an exact reproduction of the paper's reported scores.

| Setting | Evidence / decision |
|---|---|
| LLaMA2-7B → TinyLLaMA-1.1B | Paper's same-vocabulary pair; canonical base HF IDs selected here. The exact Tiny checkpoint revision is not specified by the released launcher. |
| 10 epochs, LR 1e-3, effective batch 32, cosine, LoRA 256/8/0.1 | Paper Table I. The same declared recipe is also used for local teacher SFT; its exact original SFT settings/checkpoint cannot be recovered from the released Tiny launcher. |
| Frozen, fine-tuned teacher | Paper fine-tunes teachers with LoRA; this pipeline prepares one local adapter, then merges and freezes it for BI and both branches. Existing local teacher adapters are supported. |
| q_proj/v_proj | Inferred from upstream PEFT Llama defaults: upstream creates LoraConfig without explicit targets. We set these explicitly. |
| Prompt truncation 256, total 512, rollout cap 256 | Released GPT2 Dolly launcher/data loader; adopted here because no Tiny launcher is published. Raw processed prompt IDs, add_special_tokens=False; prompt is truncated on the right before all forwards. |
| KD rate 0.5 and temperature 2 | Paper IV-A and released launcher. Reference CE uses the original response; KL uses the sampled response. |
| Forward KL and no T² rescaling | Released launcher's divergence choice and ordinary upstream KL convention. This is a declared ordinary OPD objective; reverse KL is also available through opd.train. |
| AdamW, decay 0.01, clip 1, rollout T=1/top_p=1 | Released launcher. One B200 with microbatch 1/accumulation 32 replaces eight GPUs while keeping effective batch size. This does not reproduce distributed numerical behavior. |
| OPD from the first update | Deliberate choice for the paired OPD comparison. Public OPD flags are commented examples, including after-one-epoch rollout; parser defaults differ (3 epochs, ratio 0.5). No exact Tiny OPD schedule can be inferred. We use fresh student responses on every microbatch with no off-policy warmup. |
| Dataset | Bundled processed MiniLLM-format Dolly, not a new random 4k subset. Audited split hashes/counts are saved at prepare time. Fourteen train rows overlap test prompts and are removed. Other train duplicates remain. BI uses disjoint dev samples. |
| IMPACT parameters | Local experiment choices: lambda 0.25, Top-6, tau 2, embedding-overlap ridge initializer capped at 4096 tokens. Not claimed to be published DSKDv2 hyperparameters. |
| Evaluation | User disabled MATH-500/GSM8K. No benchmark eval jobs are scheduled. The paper's sampled instruction-following ROUGE-L protocol is not run in this configuration. Reports describe training and BI only. |

Primary code references at the inspected revision:

- [Dolly launcher](https://github.com/songmzhang/DSKDv2/blob/ae4993e8b857ca001742de651498dc627138dd32/scripts/dolly/gpt2/run_dskdv2.sh)
- [Arguments / OPD defaults](https://github.com/songmzhang/DSKDv2/blob/ae4993e8b857ca001742de651498dc627138dd32/code/arguments.py)
- [Processed Dolly prompts](https://github.com/songmzhang/DSKDv2/blob/ae4993e8b857ca001742de651498dc627138dd32/code/data_utils/distill_datasets.py)
- [Training loop / on-policy switch](https://github.com/songmzhang/DSKDv2/blob/ae4993e8b857ca001742de651498dc627138dd32/code/distillation.py)
- [LoRA and teacher loading](https://github.com/songmzhang/DSKDv2/blob/ae4993e8b857ca001742de651498dc627138dd32/code/distiller.py)

Model cards: [LLaMA2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf), [TinyLLaMA base](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T). The TinyLLaMA card describes its LLaMA2-compatible tokenizer. The runner still verifies actual local assets instead of relying on model names. The pair tests application of IMPACT in a shared-tokenizer OPD setting; it does not establish a cross-tokenizer result.
