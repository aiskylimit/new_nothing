# Unified IE

One generative model trained on several IE datasets at once. For now it's event extraction only. NER data isn't picked yet, and the RE labels (TACRED, FewRel) aren't merged yet.

The training code is EventKD: a Qwen3-0.6B student learns from a LoRA-tuned Qwen3-4B teacher with CE + skewed forward KL + span-level representation distillation (`span_finetune.py`). No pseudo labels, no self-distillation.

## Data

Data lives on Hugging Face (private dataset `datht/unified-ie`), not in git. For training, `download.txt` fetches only
`processed_data.zip` (every tokenized folder under `processed_data/general_domain/` plus `data/general_domain/ee_label_map.tsv`
and `ner_label_map.tsv`, 58 MB) and the two Qwen3 models into `models/`; `run.sh` unpacks the zip on its first run.
The scripts use `models/Qwen3-*` when it's there, else the Hub ids. The untokenized rows (`general_domain/<folder>/*.jsonl`)
stay on HF for rebuilding:

```bash
huggingface-cli download datht/unified-ie --repo-type dataset --local-dir data --include "general_domain/*"
```

`data/general_domain/` holds ACE05, MAVEN, RAMS and GENEVA merged into one label set of 233 event types. The prompt doesn't name the dataset or list the labels.

| folder | task | datasets |
|---|---|---|
| `ee_v1/`, `ee_v2/` | event detection: `{"events": [[trigger, type], ...]}` | ACE, MAVEN, RAMS, GENEVA |
| `ee_arg_v1/`, `ee_arg_v2/` | event extraction with arguments: `[[trigger, type, [[argument, role], ...]], ...]` | ACE, GENEVA |
| `re_tacred/`, `re_fewrel/` | sentence-level relation extraction, no entity pair given: `{"relations": [[subject, relation, object], ...]}` | TACRED, FewRel (labels not merged, one folder each) |
| `ner_v1/`, `ner_v2/` | named entity recognition: `{"entities": [[span, type], ...]}` | CoNLL-2003, OntoNotes 5.0 (21 labels, `ner_label_map.tsv`; only PER/PERSON merged; sentences without entities kept) |

NER: CoNLL's PER and OntoNotes' PERSON are merged (93% agreement both ways on shared entity strings). CoNLL's LOC, ORG and MISC are broader than any OntoNotes type (LOC is 95% GPE there, ORG 49% ORG / 44% GPE, MISC 83% NORP), so they stay separate under their own names (`location`, `organization or team`, `miscellaneous` in v2). NER rows are tokenized with `--max-prompt-length 480` (one CoNLL test sentence needs 462 tokens), so NER training must pass 480 too. Built by `tools/build_ner_data.py`; scored per dataset as exact (span, type) matches, counting repeats (`entity`), and spans alone (`span`).

v1 writes LDC labels as `level1.level2` (`conflict.attack`) and FrameNet labels as MAVEN does (`Hostile_encounter`). v2 writes every label (and role) in lowercase words (`conflict attack`, `hostile encounter`). Same rows, same order, only the label strings differ.

Arguments are a separate training run because MAVEN and RAMS rows have none: mixed in, they'd teach "no arguments" for the types those datasets share with ACE and GENEVA.

Each row carries `dataset` and `gold_eval` (gold in the dataset's own labels). `dev.jsonl` keeps at most 300 rows per dataset for checkpoint selection, `dev_full.jsonl` has all of them. `ee_label_map.tsv` maps every original label to its merged v1/v2 name, and `label_list.md` lists the 233 labels.

## Training

```bash
pip install -r requirements.txt   # same pins as new_nothing/OpenED (transformers 4.57.3, torch 2.9.1)
# processed_data/ (already tokenized) comes with download.txt; process_data.sh is only for rebuilding it
bash scripts/general_domain/sft_teacher_4B.sh ee_v1                # LoRA SFT of the Qwen3-4B teacher
bash scripts/general_domain/eventkd_0.6B_4B.sh ee_v1 <teacher_lora_dir>
```

Swap `ee_v1` for `ee_v2`, `ee_arg_v1` or `ee_arg_v2`.

RE, one model per dataset, all four runs at once on GPUs 4-7 (one 80GB B200 each; effective batch 32 as before, from micro-batch 8 x 4 for the teacher and 16 x 2 for the student):

```bash
bash run.sh   # teacher TACRED (GPU 4), teacher FewRel (5), student TACRED (6), student FewRel (7)
```

Each run is a standalone script in `scripts/general_domain/re/` with its GPU, data and hyperparameters fixed. Dev/test are scored per dataset on exact (subject, relation, object) triples, plus subject-object pairs. Console output goes to `logs/<run>.log`; `grep -rh RESULT results/qwen3/general_domain --include=log.txt` collects the scores.

Student SFT baseline (Qwen3-0.6B, full fine-tuning, CE only):

```bash
bash scripts/general_domain/sft_student_0.6B.sh ee_v1          # every row once per epoch, 5 epochs
bash scripts/general_domain/sft_student_0.6B.sh ee_v1 0.5      # square-root balancing, 13 epochs
```

`--balance-alpha a` (BalancedSampler) gives dataset d a share of every epoch prop. to n_d^a, with no row twice in an epoch. With a = 0.5 on ee_v* an epoch is GENEVA 1,872 (all) + ACE 2,402 + RAMS 3,456 + MAVEN 7,008 rows. Each dataset is walked through a shuffled order across epochs, so every row is used before any row is used again (after 13 epochs: MAVEN rows 3-4 times, RAMS 7-8, ACE 10-11, GENEVA 13). The seed only fixes the order. Each epoch's row list is saved to `balanced_epochs/epoch_<e>_indices.json`.

Things that differ from the original EventKD scripts:

- `--label-map`: dev/test are also scored per dataset in its own labels. A predicted label the row's dataset doesn't use is dropped, and RAMS is scored on its 38 level-2 types. `grep RESULT log.txt` gives one line per dataset and split.
- `--eval-gen-mode best`: dev answers after every epoch, test answers only when the dev score (mean of per-dataset trigger F1) beats the best so far, marked by a `BEST` line. `every` keeps the old schedule.
- Decoding is greedy (no `--do-sample`). `do_sample` is passed straight to `generate()`, because transformers otherwise fills it from Qwen3's `generation_config.json`, which samples.
- `process_data.py` drops train/dev rows longer than `--max-prompt-length` / `--max-length` instead of cutting them (10 train rows, 1 dev row). Test rows are never dropped.

Speed (same losses, ported from `new_nothing/OpenED`):

- The student and teacher forwards compute logits only over the label window (`logits_to_keep`), the only positions CE and KD read. On ED rows that's about 33 of 768 columns, so the LM head and the KL work shrink by about 20x. Checked on CPU: CE and SFKL match the full computation to 6 decimals.
- `--dynamic-pad` pads each batch to its longest row (rounded up to 64) instead of 768. It's on for teacher SFT. It stays off while the span loss runs, because the span token weights average over every query row, padding included, so they'd change with the pad width.
- `--gen-backend vllm` generates dev/test answers in vLLM (`gen_backend.py`, `tools/vllm_generate.py`, from `new_nothing`). It needs its own environment (`uv venv .venv-vllm --python 3.12 && uv pip install --python .venv-vllm vllm==0.27.1`, or `VLLM_PY=...`) and a single process (world size 1). Answers are statistically equivalent to `generate()`, not token-identical.
