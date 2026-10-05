# Methods and how faithfully they are ported

Each family has its own command, config tree and run names (`<arm>-<track>` for `sgl`,
`palign-<model>`, `ssft-<model>`), so results from different methods never overwrite each other and
always land in the shared `results/` table side by side.

## `sgl` — spectral-guided learning and step allocation

Code: `sgl.data.prepare`, `sgl.signals`, `sgl.selection`, `sgl.allocation`, `sgl.transforms`,
`sgl.training`. Configs: `configs/sgl/` (`methods/` = what to compute, `tracks/` = model + data +
recipe, `<track>/<arm>.yaml` = one published arm).

| Method config | Arm | Supervision |
|---|---|---|
| `sft` / `sft-dft` | sft-nll / sft-dft | every response token, NLL / DFT |
| `sgl-spectral` | spectral | steps holding p = 0.95 of the spectral energy (Eq. 8), uniform |
| `iwc-gated` | iwc-stable | spectral gate p = 0.95 + IWC-Stable entropy weights, lambda 1 |
| `iwc-entropy` | iwc-nogate-l05 | every token, IWC-Stable entropy weights, lambda 0.5, tau 2, clip 2 |
| `iwc-shuffled` | iwc-shuf-l05 | as above with the entropies permuted within each trace (control) |
| `iwc-gain` | iwc-gain-l05 | as above with per-step answer-information gain |
| `provenance*` | prov-nll-dft(-j8), prov-dft-nll | NLL on the teacher prefix, DFT on the continuation (+ junction) |
| `sft-trans` | trans | NLL + lambda * L_trans (next-step representation) |

Fidelity: the code is the pre-refactor code moved into a package. Every dataset the CPU stages
write is byte-identical to `paper-v1` (`tests/regression/golden.json`), and every config launches
exactly the command of the shell driver that produced its published run: the drivers are executed
with recording fake `python`/`torchrun` and compared flag by flag (`tests/regression/legacy_drivers.py`,
30 prepare / capture / weights / train / eval commands).

Note on "P-ALIGN through this pipeline" (`sft-nll`): the SGL prompt is the evaluation prompt
(`...\boxed{}.<question>`), whereas P-ALIGN's own LLaMA-Factory training inserts a newline
(`...\boxed{}.\n<question>`) and, for Qwen students, also supervises the newline after
`<|im_end|>`. `sft-nll` therefore matches P-ALIGN's data and objective but not its exact training
tokens; `palign run` reproduces those (see below). Two intentional differences from the old drivers:
the IWC arms write their dataset under its final name directly (`allocation.build --output-name`)
instead of building in a scratch dir and renaming it, and seed-43/44 evaluations are tagged
`<arm>-<track>-e43` for every track (the 1.5B driver used `<arm>-e43`).

## `palign` — P-ALIGN (adaptive prefix alignment)

Code: `sgl.palign` (`truncate`, `align`, `build`), `sgl.data.s1k`. Configs: `configs/palign/`.

| Step | Upstream | Port | Checked by |
|---|---|---|---|
| Teacher traces | s1K-1.1 `deepseek_thinking_trajectory` | `sgl.data.s1k --segment-mode none` | `test_ssft.py` (shared converter) |
| Truncation | `src/binary_search.py` | `sgl.palign.truncate` | judge prompt byte-identical; verdict rule; 300 searches give upstream's exact result |
| Continuation | `src/prefix-alignment.py` | `sgl.palign.align` | prompt byte-identical (upstream `process_data` run with stubs) |
| Training file | not released as code | `sgl.palign.build` | marker layout reproduces all 966 released rows |
| Training tokens | LLaMA-Factory alpaca converter + `deepseekr1` / `qwen3` / `qwen` templates | `sgl.data.prepare --instruction-separator "\n" [--stop-suffix "\n"]` | ids identical to LLaMA-Factory's own `encode_oneturn` (`tests/test_palign_training_format.py`) |
| Training | LLaMA-Factory (`src/train.py` + YAML) | `sgl.training.train` | same recipe; `train_reference` stage runs the original |
| Evaluation | `src/test.py` + `src/evaluation.py` | `sgl.eval.evaluate --palign-prompt --grader math_verify` | prompt test; grader reproduces 429/429 published labels |

Notes:

* Truncation runs every search in lockstep, one batched judge call per round instead of one search
  after another; each trace still receives exactly the verdicts it would get alone. `--backend hf`
  decodes exactly as upstream (model generation defaults, 256 tokens, one prompt at a time);
  `--backend vllm` samples from the same generation config but is not bit-identical to HF.
* The assembly step (`<Begin_of_Prefix>prefix<End_of_Prefix>\ncontinuation`, kept when the
  continuation's last `\boxed{}` matches the gold answer under math_verify) is reconstructed from
  the released file, since upstream ships no script for it.
* P-ALIGN runs keep their own data dir (`data/palign-<model>/`) because their token layout differs
  from the SGL arms'. LoRA runs merge the adapter before saving and evaluate the merged model, as
  P-ALIGN exports before `test.py`.
* Native training uses P-ALIGN's own schedule, cosine to 0 (the SGL arms decay to 1e-5), and its
  precision: full fine-tuning keeps fp32 trainable weights under bf16 autocast (what LLaMA-Factory
  does for full tuning), LoRA keeps a bf16 base. No DeepSpeed, as in P-ALIGN's runs. Losses are the
  token mean over the optimizer step in both trainers.
* `palign run <cfg> --stages reference_data,train_reference stages.train_reference.python=<llamafactory env>/bin/python`
  trains with the unmodified LLaMA-Factory script; evaluate it with `--stages eval run_suffix=-llamafactory`
  (add `lora_adapter=true` for the LoRA configs, whose LLaMA-Factory output is an adapter).
* Not bit-identical by construction: data order and dropout RNG differ between trainers, the HF
  Trainer's default AdamW implementation may be the fused one (same update rule), and sdpa vs
  LLaMA-Factory's `flash_attn: auto` kernels. These change numbers at the level of seed noise only.

## `ssft` — Segment-Selective SFT

Code: `sgl.ssft` (`attribution`, `select`, `build`), `sgl.data.s1k`. Configs: `configs/ssft/`.

| Step | Upstream (SegmentSelectiveSFT fork) | Port | Checked by |
|---|---|---|---|
| Traces + segments | `prepare_s1k.py`, `Attribution/segment_split.py` | `sgl.data.s1k` | converter and `last_boxed` vs upstream; `paragraph` / `cue` splits identical |
| Attribution | `Attribution/grad_analyze.py` | `sgl.ssft.attribution` | per-token IG identical to upstream's class on a bf16 model |
| Selection | `Attribution/get_important_segments.py` | `sgl.ssft.select` | output file byte-identical (unit test + regression) |
| Labels | `SelectiveSFT/train_mask.py` `formatting_prompts_func` | `sgl.ssft.build` | identical ids and masks over 40 template / prompt / truncation modes |
| Training | Unsloth + TRL `SFTTrainer` | `sgl.training.train` | same hyperparameters; `train_reference` stage runs the original |
| Evaluation | `eval.sh` + `score_palign.py` (math_verify) | P-ALIGN protocol | as above |

Notes:

* To train on the fork's released selection instead of recomputing IG:
  `ssft run r1-qwen-1.5b --stages build,train,eval stages.build.args.data-path=<.../solutions_selected.jsonl>`.
* The 7B LoRA config merges the adapter before evaluation, as the fork does with `merge_lora.py`.
* `configs/ssft/r1-qwen-1.5b.yaml` is the fork's `commands.sh`: full fine-tuning, DeepSeek-R1
  template, P-ALIGN prompt, empty think block (`--think-prefix off`), paragraph segments, IG from
  DeepSeek-R1-Distill-Qwen-7B with 20 steps, top-70% strength + coherence <= 0.8, lr 5e-5 cosine to
  0, batch 1 x 32, seed 3407. `qwen25-7b-lora.yaml` is `train.sh` (LoRA r64 / alpha 64).
* Kept from upstream on purpose: no EOS token is appended to the target, the first / second-to-last
  / last segments are always learned, samples truncated to nothing are dropped, and the answer span
  for IG is located by scanning token strings for `boxed`, `{` and `}` (correct for Qwen's
  tokenizer, which keeps `{` and the closing `}` as separate tokens).
* Attribution runs in the model's dtype on any device (upstream hard-codes `.cuda()` and bf16).
* Not ported: `--think_prefix special` (adds tokens and resizes embeddings; use the reference
  trainer) and the Qwen-Math grader of the fork's own `Eval/` (the comparison uses math_verify).
* Whether Unsloth's full fine-tuning keeps fp32 master weights was not verified; the native 1.5B
  run uses fp32 trainable weights with bf16 autocast, like the other families.
