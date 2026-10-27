# Unified IE

One generative model trained on several IE datasets at once: event extraction, relation extraction and NER. There's one teacher per source dataset, an SFT student per task and one on all tasks together (`mt_v2`), and a multi-task student distilled from the per-dataset teachers (BAM). RE labels (TACRED, FewRel) aren't merged.

The training code is EventKD: a Qwen3-0.6B student learns from a LoRA-tuned Qwen3-4B teacher with CE + skewed forward KL + span-level representation distillation (`span_finetune.py`). No pseudo labels, no self-distillation.

## Data

Data lives on Hugging Face (private dataset `datht/unified-ie`), not in git. For training, `download.txt` fetches only
`processed_data.zip` (every tokenized folder under `processed_data/general_domain/` plus `data/general_domain/ee_label_map.tsv`
and `ner_label_map.tsv`, 93 MB) and the two Qwen3 models into `models/`; `run.sh` unpacks the zip on its first run, and again whenever the unpacked EE data predates the argument format (no `unknown_args`); an old zip stops it with a message to fetch the zip again.
The scripts use `models/Qwen3-*` when it's there, else the Hub ids. The untokenized rows (`general_domain/<folder>/*.jsonl`)
stay on HF for rebuilding:

```bash
huggingface-cli download datht/unified-ie --repo-type dataset --local-dir data --include "general_domain/*"
```

`data/general_domain/` holds ACE05, MAVEN, RAMS and GENEVA merged into one label set of 233 event types. The prompt doesn't name the dataset or list the labels.

| folder | task | datasets |
|---|---|---|
| `ee_v1/`, `ee_v2/` | event extraction with arguments: `{"events": [[trigger, type, [[argument, role], ...]], ...]}` | ACE, MAVEN, RAMS, GENEVA (arguments from ACE and GENEVA only) |
| `re_tacred/`, `re_fewrel/` | sentence-level relation extraction, no entity pair given: `{"relations": [[subject, relation, object], ...]}` | TACRED, FewRel (labels not merged, one folder each) |
| `ner_v1/`, `ner_v2/` | named entity recognition: `{"entities": [[span, type], ...]}` | CoNLL-2003, OntoNotes 5.0 (21 labels, `ner_label_map.tsv`; only PER/PERSON merged; sentences without entities kept) |
| `mt_v2/` | all three tasks, each row with its own task's prompt | `ee_v2` + `ner_v2` + `re_tacred` + `re_fewrel` (148.6k train rows, 8 datasets) |

NER: CoNLL's PER and OntoNotes' PERSON are merged (93% agreement both ways on shared entity strings). CoNLL's LOC, ORG and MISC are broader than any OntoNotes type (LOC is 95% GPE there, ORG 49% ORG / 44% GPE, MISC 83% NORP), so they stay separate under their own names (`location`, `organization or team`, `miscellaneous` in v2). NER rows are tokenized with `--max-prompt-length 480` (one CoNLL test sentence needs 462 tokens), so NER training must pass 480 too. Built by `tools/build_ner_data.py`; scored per dataset as exact (span, type) matches, counting repeats (`entity`), and spans alone (`span`).

v1 writes LDC labels as `level1.level2` (`conflict.attack`) and FrameNet labels as MAVEN does (`Hostile_encounter`). v2 writes every label (and role) in lowercase words (`conflict attack`, `hostile encounter`). Same rows, same order, only the label strings differ.

Detection and arguments are one task with one prompt. MAVEN and RAMS have no argument annotation, so their events are written `[trigger, type, []]` and the row carries `"unknown_args": true`. That `[]` means "not annotated", not "no arguments", so the trainer gives it no loss: `LMTrainDataset` finds the tokens overlapping each `[]` (with the Qwen3 tokenizer it's exactly one token, `' []'`) and sets their labels to -100, which takes them out of CE and of the KD loss too. Checked on the full train split: one masked token per MAVEN/RAMS event (66,286 + 6,725), every other response token keeps its label, ACE and GENEVA rows untouched. MAVEN/RAMS `gold_eval` has no arguments, so they're scored on triggers only; ACE and GENEVA also get argument P/R/F1. The checkpoint is still picked on the mean trigger F1.

`mt_v2` is the four v2 folders concatenated, rows unchanged (`tools/build_mt_data.py v2`), tokenized with `--max-prompt-length 480` for NER. Its tokenized rows are the single-task ones plus 3 RAMS train rows of 461-480 prompt tokens that the 460 limit of `ee_v2` drops. Dev and test are scored task by task (`eval_loop.score_responses`, `--label-map ee_label_map.tsv,ner_label_map.tsv`) and the checkpoint is picked on the mean of the EE, NER and RE scores, so each task weighs the same. Checked on CPU: each task's scores on `mt_v2` test are identical to scoring its own folder. Scripts made before `mt_v2` find it missing on a server that unpacked an older zip, so they unpack it from the zip (`tools/unpack_data.py`).

Each row carries `dataset` and `gold_eval` (gold in the dataset's own labels). `dev.jsonl` keeps at most 300 rows per dataset for checkpoint selection, `dev_full.jsonl` has all of them. `ee_label_map.tsv` maps every original label to its merged v1/v2 name, and `label_list.md` lists the 233 labels.

## Training

```bash
pip install -r requirements.txt   # same pins as new_nothing/OpenED (transformers 4.57.3, torch 2.9.1)
# processed_data/ (already tokenized) comes with download.txt; process_data.sh is only for rebuilding it
bash scripts/general_domain/sft_teacher_4B.sh ee_v1                # LoRA SFT of the Qwen3-4B teacher
bash scripts/general_domain/eventkd_0.6B_4B.sh ee_v1 <teacher_lora_dir>
```

Swap `ee_v1` for `ee_v2`.

Every SFT run at once on GPUs 4-7 (one 80GB B200 each; effective batch 32 as before, from micro-batch 8 x 4 for the teacher and 16 x 2 for the student):

```bash
bash run.sh   # every teacher/student SFT run (EE v1/v2, NER, TACRED, FewRel), one queue per GPU 4-7
```

`run.sh` runs 17 SFT jobs: a Qwen3-4B LoRA teacher per source dataset (`script:dataset` passes the dataset, and `--datasets` keeps only its rows in train, dev and test: EE teachers on ace, maven, rams, geneva for both ee_v1 and ee_v2, NER on conll03 and ontonotes, plus TACRED and FewRel) and a Qwen3-0.6B student per task (ee_v1, ee_v2, ner_v2, re_tacred, re_fewrel). One queue per GPU, so each GPU runs one job at a time, on torchrun port 29500 + 10 x GPU; the queues are at the end of `run.sh`, and `run2.sh` restarts the GPU 7 queue alone. NER scripts pass `--max-prompt-length 480` and `--label-map data/general_domain/ner_label_map.tsv`.

Each run is a standalone script in `scripts/general_domain/{ee,re,ner}/` with its GPU, data and hyperparameters fixed. Dev/test are scored per dataset on exact (subject, relation, object) triples, plus subject-object pairs. Console output goes to `logs/<run>.log`; `grep -rh RESULT results/qwen3/general_domain --include=log.txt` collects the scores.

One student on all three tasks (`mt_v2`), the SFT baseline and multi-teacher KD as in BAM (Clark et al., ACL 2019) once the per-dataset teachers are trained:

```bash
bash scripts/general_domain/mt/sft_student_0.6B_mt_v2.sh                 # CE only, every row once per epoch, 5 epochs
BALANCE=0.5 bash scripts/general_domain/mt/sft_student_0.6B_mt_v2.sh     # square-root balancing, 18 epochs
bash scripts/general_domain/bam_0.6B_4B.sh mt_v2                          # BAM, 8 teachers
KD_SCHEDULE=constant KD_TYPE=sfkl W_SPAN=1.0 bash scripts/general_domain/bam_0.6B_4B.sh mt_v2   # EventKD losses, same teachers
bash scripts/general_domain/bam_0.6B_4B.sh ee_v1                          # EE only, 4 teachers
```

With `BALANCE=0.5` an `mt_v2` epoch is 41,183 rows (GENEVA all 1,872, TACRED 2,934, ACE 2,402, RAMS 3,453, CoNLL 5,127, MAVEN 7,009, FewRel 7,795, OntoNotes 10,591), so 18 epochs give about the updates of 5 plain ones.

The BAM student (full fine-tuning, same data and hyperparameters as the SFT student of that folder) learns from Qwen3-4B LoRA teachers, one per source dataset, and each row is distilled only from the teacher of its own dataset (`--teacher-peft-paths ace=<dir>,conll03=<dir>,...`): on `mt_v2` the four `ee_v2` teachers, the two `ner_v2` ones, TACRED and FewRel. The base 4B is loaded once with one unmerged adapter per dataset. Each batch runs one teacher forward per dataset in it and puts the logits and hidden states back in row order. That's fine for the span loss, since it only compares spans inside a row. Without `TEACHERS=`, the script takes each teacher's best dev epoch from its results dir (`sft_4B_<data>_<dataset>`, `sft_4B_re_<dataset>`) (`tools/best_lora.py`; a `PEFT_PATH` continuation in `<settings>-from<step>` competes on its dev score too).

Every part is its own switch, so a rerun only changes a flag; the run dir `bam_0.6B_4B_<data>/<type>_<schedule>_span<w>[_balance<a>]` is named after them:

- `--kd-schedule anneal` (`KD_SCHEDULE`): BAM's teacher annealing, the KD weight goes linearly from 1 to 0 over training, so the student starts on the teacher and ends on the gold labels. `constant` uses `--kd-ratio`.
- `--type fkl` (`KD_TYPE`): forward KL, which is BAM's loss (CE on the teacher's distribution). `sfkl` is the skewed KL from EventKD.
- `--w-span-loss` (`W_SPAN`): 0 turns the span loss off and skips the teacher's hidden states; 1.0 is EventKD's. Spans come from events only, so NER and RE rows add nothing to it yet.
- `--balance-alpha` (`BALANCE`): as for the SFT student.

Checked on CPU with a tiny Qwen3 and random LoRAs: each row's routed logits and hidden states match its own merged LoRA to float error (relative 3e-5, while the two teachers differ by 1.22), the span loss and KL run on a real ee_v1 dev batch mixing all four datasets and on an `mt_v2` batch with one row of each of the 8 datasets (8 adapters, every row equal to its own adapter alone), and a LoRA student's layers are the same through `get_decoder()` as before.

Student SFT baseline (Qwen3-0.6B, full fine-tuning, CE only):

```bash
bash scripts/general_domain/sft_student_0.6B.sh ee_v1          # every row once per epoch, 5 epochs
bash scripts/general_domain/sft_student_0.6B.sh ee_v1 0.5      # square-root balancing, 13 epochs
```

`--balance-alpha a` (BalancedSampler) gives dataset d a share of every epoch prop. to n_d^a, with no row twice in an epoch. With a = 0.5 on ee_v* an epoch is GENEVA 1,872 (all) + ACE 2,402 + RAMS 3,452 + MAVEN 7,009 rows. Each dataset is walked through a shuffled order across epochs, so every row is used before any row is used again (after 13 epochs: MAVEN rows 3-4 times, RAMS 7-8, ACE 10-11, GENEVA 13). The seed only fixes the order. Each epoch's row list is saved to `balanced_epochs/epoch_<e>_indices.json`.

Things that differ from the original EventKD scripts:

- `--label-map`: dev/test are also scored per dataset in its own labels. A predicted label the row's dataset doesn't use is dropped, and RAMS is scored on its 38 level-2 types. `grep RESULT log.txt` gives one line per dataset and split.
- `--eval-gen-mode best`: dev answers after every epoch, test answers only when the dev score (mean of per-dataset trigger F1) beats the best so far, marked by a `BEST` line. `every` keeps the old schedule.
- Decoding is greedy (no `--do-sample`). `do_sample` is passed straight to `generate()`, because transformers otherwise fills it from Qwen3's `generation_config.json`, which samples.
- `process_data.py` drops train/dev rows longer than `--max-prompt-length` / `--max-length` instead of cutting them (10 train rows, 1 dev row). Test rows are never dropped.

Speed (same losses, ported from `new_nothing/OpenED`):

- The student and teacher forwards compute logits only over the label window (`logits_to_keep`), the only positions CE and KD read. On ED rows that's about 33 of 768 columns, so the LM head and the KL work shrink by about 20x. Checked on CPU: CE and SFKL match the full computation to 6 decimals.
- `--dynamic-pad` pads each batch to its longest row (rounded up to 64) instead of 768. It's on for every SFT script and also works with the span loss (see below). `--student-gen` still turns it off.
- `--gen-backend vllm` generates dev/test answers in vLLM (`gen_backend.py`, `tools/vllm_generate.py`, from `new_nothing`). It needs its own environment (`uv venv .venv-vllm --python 3.12 && uv pip install --python .venv-vllm vllm==0.27.1`, or `VLLM_PY=...`) and a single process (world size 1). Answers are statistically equivalent to `generate()`, not token-identical.

Span loss fix: `compute_token_weights` in `span_finetune.py` used to take `attn_weights.mean(dim=1)`, which averages the attention over every query row, padding rows included. With 768-wide rows about 2/3 of those rows are padding, so the token weights were mostly set by the padding and changed with the pad width. It now averages over the real query rows only, which is the definition in the EventKD paper (w_t = 1/N sum_s alpha_s->t). Checked on CPU: the old weights had correlation 0.16 with the paper's weights and 0 of the top 10 tokens in common; after the fix the weights at pad width 768 and 256 differ by at most 4e-8. The same bug is in the original EventKD code and in new_nothing (CED), which weren't changed.
