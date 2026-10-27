# Validation of the LLaMA2 → TinyLLaMA OPD recipe

Validation host: macOS CPU, Python 3.10, torch 2.7.0 and transformers 4.55.2. Target environment remains Python 3.11, torch 2.9.0 CUDA 12.8, transformers 4.56.1 on B200. These CPU checks validate code paths, not B200 performance or model quality.

- `python3 -m unittest discover -s tests_opd -v`: 27 tests, including the legacy regression tests, shared-vocabulary OPD, raw prompt/truncation consistency, explicit KD temperature scaling, teacher dependencies and reporting with math disabled.
- Python compilation, shell syntax (`bash -n`) and `git diff --check` pass.
- Generated the default B200 plan with local model/data/output paths. It schedules teacher SFT, three Dolly BI subsets and both student branches; no evaluation jobs or MATH/GSM datasets.
- Prepared all bundled Dolly splits: 11,421 train rows after removing 14 rows whose prompts overlap test; 500 original test rows; 972 distinct eligible dev prompts; 256 calibration examples per saved subset. Test/train/calibration intersections checked by prompt hash. Original prompt and response strings preserved.
- Tiny random LLaMA fixture models: teacher hidden size 24 / 3 layers, student hidden size 16 / 2 layers, shared WordLevel tokenizer without a chat template. Teacher LoRA SFT ran one update. Both OPD and OPD + IMPACT ran two updates using the frozen teacher adapter, reference CE, temperature-2 forward KL, cosine schedule and nonzero LoRA dropout. The fixture overrides rank/budget/lengths for CPU execution.
- Resumed OPD after one update and compared with the uninterrupted two-update run. Every saved student adapter tensor matched exactly (`rtol=0, atol=0`); scheduler states matched. Cumulative rollout token counts matched.
- Collected BI separately through the CLI using the teacher adapter, then trained OPD + IMPACT using that cached summary. Teacher adapter and prompt-input signatures validated successfully.
- Aggregated the two real CPU training outputs with math disabled: complete training report, two cost rows, no missing evaluations and no fabricated accuracy.
- Optional math wrapper dry-run with the base fixture omitted chat-template flags. This does not run or validate a math benchmark.

No pretrained LLaMA2/TinyLLaMA training, instruction benchmark, math benchmark or CUDA/B200 performance run was performed on this host. The public TinyLLaMA tokenizer was inspected (32,000 tokens, BOS 1, EOS 2, no chat template). The real teacher/student tokenizer compatibility is checked locally before teacher SFT on the target machine; access to gated LLaMA2 weights must already be approved for the user's Hugging Face account.

See [DSKDV2_REFERENCE.md](DSKDV2_REFERENCE.md) for the reference revision, adopted hyperparameters, inferred choices and differences from the published experiment.

After the dataset migration to Hugging Face, unit tests use synthetic data fixtures created in temporary directories; fresh clones no longer need any tracked dataset payload. The published full Dolly snapshot is validated separately through the download/prepare path.

Migration validation: 31 unit tests passed. A fresh staged export without any `data/` directory passed the code-only tests. Both the pinned Hub dataset downloader and the template-compatible `--url` download path were exercised from empty local roots; offline preparation reproduced all six normalized train/test/calibration JSONL files byte-for-byte against the published release.
