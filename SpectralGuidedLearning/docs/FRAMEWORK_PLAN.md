# Kế hoạch hợp nhất SGL + P-ALIGN + SegmentSelectiveSFT thành một framework

Trạng thái: **đang triển khai** — xem §8 cho các quyết định đã chốt và tiến độ. Tài liệu này dựa trên `main` @ `7990089`
(sau commit "Add P-ALIGN and SegmentSelectiveSFT reference code").

---

## 1. Hiện trạng

### 1.1 Ba codebase, ba cách tổ chức

| | SGL (`src/`) | P-ALIGN (`P-ALIGN/`) | SegmentSelectiveSFT (`SegmentSelectiveSFT/`) |
|---|---|---|---|
| Ý tưởng | Chọn/gán trọng số step theo tín hiệu (spectral, entropy, answer-gain), IWC, L_trans, provenance | Cắt teacher CoT còn prefix ngắn nhất đủ dùng (binary search, student tự đánh giá) → student viết tiếp | Integrated Gradients từ segment → token đáp án, chọn segment quan trọng, SFT có mask |
| Code Python | ~7.7k dòng, 28 file phẳng trong `src/` | ~0.8k dòng, 7 script | ~3k dòng (chưa kể `latex2sympy` vendored ~9k dòng) |
| Train | HF Trainer + DeepSpeed (`train_sft.py`), Unsloth (`train_sft_unsloth.py`) | LLaMA-Factory (`src/train.py` + YAML) | Unsloth + TRL (`SelectiveSFT/train_mask.py`) |
| Eval | `evaluate.py` (vLLM) + `palign_grader` / `answer_scoring` | `test.py` + `evaluation.py` (math_verify) | `Eval/math_eval.py` (Qwen-Math grader + latex2sympy) + `score_palign.py` |
| Môi trường | torch 2.13, transformers 5.5.3, vllm 0.27.1 | conda pin cứng (requirements dạng conda export) | **hai** env xung đột: `ssft_train` (torch 2.9, tf 4.57) và `ssft_eval` (torch 2.7.1, vllm 0.10) |
| Điều phối | ~60 shell script + 13 `project_commands*.sh` | `project_commands*.sh` | `run_pipeline.sh` (theo stage) + `train.sh`/`eval.sh` |
| Config | argparse + `--config` YAML ad-hoc trong từng script | YAML LLaMA-Factory | env var + flag shell |

### 1.2 Phần trùng lặp (cơ hội hợp nhất)

Nhìn kỹ, ba phương pháp là **các biến thể của cùng một pipeline**:

```
dữ liệu → (biến đổi trace) → phân đoạn step → tín hiệu/step → chọn (mask) → gán trọng số → objective → train → eval
```

| Thành phần | SGL | P-ALIGN | SSFT | Hợp nhất thành |
|---|---|---|---|---|
| Phân đoạn | `segmentation.py` (câu, regex `[.?!}\]]\s+[A-Z]`), `step_transitions.py` (`\n\n`) | `split_sentences` (`". "`) | `segment_utils.py` (`cue` / `paragraph`) | `sgl.segment` — một registry `sentence` / `paragraph` / `cue` / `palign_sentence`, cùng một hàm map char→token bằng offset |
| Map span ký tự → token | `encode_with_offsets` + `step_token_spans` | — | viết lại 2 lần (trong `grad_analyze.py` và `train_mask.py`) | 1 hàm duy nhất |
| Tín hiệu step | spectral strength, entropy, answer-gain (probe `\boxed{`) | sufficiency `[ENOUGH]` từ student | IG tới token đáp án (probe `</think> So, the final answer is \boxed{`) | `sgl.signals` — mỗi tín hiệu ghi cùng một schema parquet `{id, step_scores}` |
| Chính sách chọn | `select_steps_by_energy` (top theo strength đến ngưỡng p) | prefix ngắn nhất đủ dùng | top theo `Σ|IG|/√n` đến 70% + lọc coherence ≤ 0.8 + luôn giữ segment đầu/áp chót/cuối | `sgl.selection` — `cumulative_top`, `coherence_filter`, `always_keep`, `prefix` |
| Gán trọng số | IWC / IWC-Stable / shuffled / reverse / gain | uniform | uniform | `sgl.allocation` (giữ nguyên) |
| Objective | masked NLL Eq.9 (Z toàn bước), DFT, per-source, L_trans | NLL (LLaMA-Factory) | NLL (labels=-100) | `sgl.training.losses` (giữ nguyên, đã tổng quát nhất) |
| Prompt | `PROMPT_TEMPLATE` P-ALIGN-style, `palign_close_thinking` | `test.py` `apply_chat` | `prompt_style default|palign`, `think_prefix none|off|plain|special` | `sgl.data.prompts` — một nguồn duy nhất cho train và eval |
| Grader | `answer_scoring`, `palign_grader`, `oat_math_grader` | `math_verify` (+ oat) | Qwen-Math `grader.py` + latex2sympy | `sgl.eval.graders` registry: `palign`, `math_verify`, `qwen_math`, `oat` |

Tức là P-ALIGN và SSFT **không cần tồn tại như repo riêng**: P-ALIGN là một *data transform* (sinh ra trace mới có `<End_of_Prefix>`), SSFT là một *tín hiệu* (IG) + *chính sách chọn*. Cả hai cắm vào đúng các điểm mở rộng mà SGL đã có.

### 1.3 Vấn đề cần dọn khi refactor

- `src/` phẳng 28 file, import bằng `PYTHONPATH=src` / `sys.path.insert` trong `tests/conftest.py`; không cài được như package (`[tool.uv] package = false`).
- Mỗi script tự viết argparse + merge YAML; `train_sft.py` và `train_sft_unsloth.py` lặp ~40 flag.
- ~60 shell script khác nhau chủ yếu ở tên model / đường dẫn; có bản trùng (`scripts/eval_qwen25-7b.sh` vs `scripts/eval/eval_qwen25-7b.sh`).
- 30 file `logs_*.txt` ở root, `logs/` (99 file) được track; `results*` nằm rải 4 thư mục.
- Code chết / tham chiếu hỏng: `VolumeCommitCallback` (Modal) và docstring `modal_train.py` không tồn tại; `docs/iwc-baselines.md` được nhắc nhưng không có; `pyproject.toml` mô tả "Pru-CoT baseline" đã lỗi thời.
- Không có README cho SGL.
- P-ALIGN `binary_search.py`: HF `generate` từng mẫu một + `sleep`, tách câu bằng `". "` → rất chậm; LLaMA-Factory cần monkey-patch cho transformers 5.x.
- SSFT: vendored `latex2sympy` (antlr generated, ~9k dòng), hai env không cài chung được, comment không dấu.

---

## 2. Mục tiêu và nguyên tắc

1. **Một package cài được** (`pip install -e .[train,eval]`), một CLI, một hệ config.
2. **Tái lập được số liệu đã công bố**: mọi refactor phải cho output byte-identical (dataset) hoặc trong sai số seed (train/eval) so với code hiện tại. Đây là *gate* của từng phase, không phải việc làm sau.
3. **Phương pháp = cấu hình**, không phải script: SFT, SGL, IWC, P-ALIGN, SSFT, và các tổ hợp (vd. SSFT-IG + IWC allocation trên dữ liệu P-ALIGN) đều là một file YAML.
4. **Một env** (torch/transformers/vllm của SGL hiện tại). Unsloth và LLaMA-Factory thành *optional extras*, không phải điều kiện.
5. Không vendored code bên thứ ba nếu có trên PyPI; giữ license header khi port.

---

## 3. Kiến trúc đề xuất

Tên package tạm: `sgl` (đổi được — xem câu hỏi mở).

```
pyproject.toml              # package thật, extras: [train] [eval] [unsloth] [dev]
src/sgl/
  __init__.py
  cli.py                    # `sgl <command> --config ... [key=value ...]`
  config.py                 # dataclass config + load YAML + override dạng a.b=c (thay mọi argparse lặp)
  registry.py               # register("segmenter", "paragraph")(fn) ...
  io.py                     # jsonl / parquet / resume-safe writer (dùng chung cho mọi stage dài)

  data/
    sources.py              # s1K-1.1, LIMO, P-ALIGN alpaca json.gz, jsonl local -> Record(question, response, answer)
    prompts.py              # PROMPT_TEMPLATE, close-thinking, think_prefix; MỘT nguồn cho train + eval
    prepare.py              # (data_prep.py) tokenize + verify span
    schema.py               # TypedDict cho các trường: input_ids, loss_mask, loss_weights, token_source, step_*
  segment/
    base.py                 # Segmenter protocol + char->token span (encode_with_offsets, step_token_spans)
    sentence.py             # SGL regex hiện tại
    paragraph.py            # "\n\n" (SSFT paragraph, step_transitions)
    cue.py                  # SSFT backtracking-cue (cách của paper SSFT)
    palign.py               # ". " của P-ALIGN (để tái lập đúng)
  signals/                  # tín hiệu per-step -> parquet {id, scores: list[float], extra...}
    spectral.py             # gradient_utils + spectral_utils + gradient_capture
    entropy.py
    answer_gain.py
    integrated_gradients.py # SSFT grad_analyze.py (đã có batch IG, grad-ckpt, OOM fallback)
    answer_probe.py         # dùng chung: tìm span token đáp án sau probe \boxed{ (answer_gain + IG đều cần)
  selection/                # step -> mask nhị phân
    cumulative.py           # select_steps_by_energy ≡ SSFT cumulative_ratio (cùng 1 hàm)
    filters.py              # coherence_filter (SSFT), always_keep(first, second_last, last)
    build.py                # (build_masks.py) áp mask lên tokenized record
  allocation/
    iwc.py                  # iwc_weights.py
    build.py                # (build_iwc_datasets.py, build_gain_signal.py)
  transforms/               # biến đổi trace (sinh văn bản mới)
    palign/
      truncate.py           # binary search prefix — vLLM, batch đồng bộ theo vòng
      align.py              # student viết tiếp từ prefix (prefix-alignment.py)
      filter.py             # giữ mẫu đúng đáp án (Eq.9 P-ALIGN) + gắn <Begin/End_of_Prefix>
    provenance.py           # build_provenance_dataset.py (token_source)
    transitions.py          # build_trans_dataset.py + step_transitions.py
  training/
    losses.py               # masked_loss.py + transition_loss.py
    collator.py, dataset.py
    trainer.py              # MaskedSFTTrainer + TransitionLossMixin
    backends/
      hf.py                 # HF Trainer + DeepSpeed (mặc định)
      unsloth.py            # train_sft_unsloth.py + phần Unsloth của train_mask.py
      llamafactory.py       # tuỳ chọn: chỉ để đối chứng P-ALIGN gốc
    lora.py                 # merge_lora.py
  eval/
    benchmarks.py           # gộp loader của SGL + P-ALIGN data/raw + SSFT data_loader
    generate.py             # vLLM, LoRA adapter, prompt từ data.prompts
    graders/
      palign.py, answer_scoring.py, oat.py  # đã có
      qwen_math.py          # SSFT Eval/grader.py + parser.py (latex2sympy2 qua PyPI)
    metrics.py              # pass@k (SSFT pass_at_k.py + SGL), bootstrap / seed mean±std
    compare.py              # compare_results.py
  diagnostics/
    iwc.py                  # iwc_diagnostics.py
configs/
  models/      qwen25-7b.yaml, qwen3-8b.yaml, r1-qwen-1.5b.yaml, r1-qwen-7b.yaml
  data/        s1k.yaml, limo.yaml, palign-966.yaml
  methods/     sft.yaml, sgl.yaml, iwc-stable.yaml, iwc-gain.yaml, palign.yaml, ssft.yaml, provenance.yaml, trans.yaml
  train/       full-ds-z2.yaml, lora-r16.yaml, lora-r64-unsloth.yaml
  eval/        palign-protocol.yaml (n=3, T=0.6, top_p=0.9, rp=1.05, 4096 tok), ssft-protocol.yaml
  experiments/ <tên>.yaml = model + data + method + train + eval (compose bằng `defaults:`)
  deepspeed/
scripts/
  run.sh                    # wrapper mỏng: GPU/torchrun/env -> `sgl run configs/experiments/x.yaml`
  sweeps/                   # vài script sweep seed/arm thật sự cần
tests/                      # giữ toàn bộ test hiện có + test mới (xem §5)
references/                 # (tuỳ chọn) code gốc đông cứng để đối chứng, hoặc chỉ ghi commit hash
experiments/                # results/ + summary.json đã công bố (logs ra khỏi git)
docs/
  README.md, methods.md, reproducing.md, extending.md
```

### 3.1 CLI

```bash
sgl prepare     --config configs/experiments/ssft-qwen25-7b.yaml   # tokenize + segment
sgl palign      --config ...   # truncate -> align -> filter (chỉ khi method cần)
sgl signal      --config ...   # spectral | entropy | answer_gain | ig
sgl build       --config ...   # selection + allocation + transforms -> train-<arm>.jsonl
sgl train       --config ...   # backend hf | unsloth | llamafactory
sgl eval        --config ...   # + --rescore
sgl compare     --results experiments/
sgl run         --config ...   # chạy các stage còn thiếu theo thứ tự, skip stage đã có output (hash config)
```

Override không cần sửa YAML: `sgl train --config x.yaml train.lr=3e-5 seed=43`.

### 3.2 Một method = một YAML

```yaml
# configs/methods/ssft.yaml  — Segment-Level Attribution (Wang et al., ICLR 2026)
segmenter: paragraph            # paper dùng `cue`
signal:
  name: ig
  model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
  steps: 20
  probe: "</think> So, the final answer is \\boxed{"
selection:
  score: ig_strength            # sum|IG| / sqrt(n_tok)
  policy: cumulative
  ratio: 0.7
  filters: [{coherence_max: 0.8}]
  always_keep: [first, second_last, last]
allocation: uniform
objective: nll
```

```yaml
# configs/methods/iwc-gain.yaml — SGL paper chính
segmenter: sentence
signal: {name: answer_gain}
selection: {policy: none}                     # no-gate
allocation: {name: iwc_stable, interpolation: 0.5}
objective: nll
```

Tổ hợp mới (vd. IG làm tín hiệu allocation thay vì selection) chỉ là đổi vài dòng.

---

## 4. Lộ trình (mỗi phase = 1 PR, có gate rõ ràng)

### Phase 0 — Đóng băng & dựng lưới an toàn (≈1 ngày)
- Tag `paper-v1` trên commit hiện tại để luôn tái lập được bài báo.
- Sinh **golden fixtures** từ code cũ trên ~20 mẫu (CPU, tokenizer nhỏ): `train-segmented.jsonl`, `train-vanilla/spectral/iwc*.jsonl`, `train-provenance.jsonl`, trans fields; mask SSFT từ `train_mask.formatting_prompts_func`; kết quả `--rescore` trên các file `raw/` đã có.
- Viết `tests/regression/` so sánh byte-for-byte với fixtures.
- Gate: test cũ + regression xanh trên code cũ.

### Phase 1 — Package hoá SGL, không đổi logic (≈2 ngày)
- `git mv src/*.py` vào `src/sgl/...` theo §3, sửa import tương đối; bỏ `sys.path` hack; `pyproject` thành package thật với extras.
- `logs_*.txt`, `logs/` ra khỏi git (giữ ở release/artifact nếu cần); gom `results*` vào `experiments/`.
- Xoá code chết (Modal callback, tham chiếu `modal_train.py`).
- Gate: **toàn bộ regression byte-identical**.

### Phase 2 — Config + CLI thống nhất (≈2–3 ngày)
- `sgl.config` (dataclass + YAML + override), `sgl.cli`.
- Chuyển các `main()` thành hàm thuần nhận config; argparse cũ thành shim mỏng (deprecate) để script cũ vẫn chạy trong lúc chuyển.
- Viết `configs/` cho các arm đã công bố (sft-nll, sgl, iwc-stable, iwc-nogate-l05, iwc-gain-l05, shuffled, provenance, trans).
- Thay ~60 shell script bằng `scripts/run.sh` + `configs/experiments/*.yaml`.
- Gate: với mỗi experiment đã công bố, `sgl build` cho dataset byte-identical; `sgl train --smoke` chạy.

### Phase 3 — Hợp nhất phân đoạn & span (≈1–2 ngày)
- `sgl.segment` với 4 segmenter; một hàm char→token duy nhất.
- Port `segment_utils.split_segments` (giữ bất biến `"".join == text`, gộp segment rỗng).
- Gate: segmenter `sentence` byte-identical với SGL cũ; `paragraph`/`cue` cho cùng segment list với SSFT cũ trên fixtures.

### Phase 4 — Tích hợp SegmentSelectiveSFT (≈3–4 ngày)
1. `signals/integrated_gradients.py` từ `grad_analyze.py` (giữ batch IG, freeze weight, grad-ckpt + kiểm tra dropout, OOM→điểm 0, `--resume` đối chiếu `question`, file compact 3 số/segment). Ghi ra schema chung.
2. `signals/answer_probe.py`: tìm span đáp án — dùng chung cho IG và answer_gain (hiện là hai đoạn code riêng làm cùng việc).
3. `selection`: `cumulative` (đã có) + `coherence_filter` + `always_keep`.
4. Mask SSFT tạo bởi `selection/build.py` (dạng `loss_mask`), **không** còn tính trong lúc train → train SSFT bằng chính `MaskedSFTTrainer` (đúng Eq.9 normalization) hoặc backend Unsloth.
5. `eval/graders/qwen_math.py` từ `Eval/grader.py`+`parser.py`; `latex2sympy2` lấy từ PyPI thay vì vendored. Giữ làm grader tuỳ chọn để so với số SSFT gốc.
6. Gate:
   - IG trên 3–5 mẫu (GPU nhỏ / model 0.5B) khớp `grad_analyze.py` (allclose).
   - `selected_spans_ids` khớp `get_important_segments.py`.
   - `loss_mask` khớp `labels != -100` của `train_mask.py` (cùng tokenizer, cùng prompt_style/think_prefix).
   - Rescore các file eval SSFT cũ cho đúng số trong `paper/main.tex` (dòng Segment-Selective SFT).

### Phase 5 — Tích hợp P-ALIGN (≈3–4 ngày)
1. `transforms/palign/truncate.py`: viết lại binary search bằng **vLLM batch theo vòng** — mọi mẫu còn active được hỏi cùng lúc mỗi vòng → ~log2(n_câu) lần gọi `generate` cho cả dataset, thay vì HF từng mẫu + `sleep`. Giữ đúng prompt `[ENOUGH]/[NOT_ENOUGH]`, `enable_thinking=False`, `max_new_tokens=256`, và segmenter `palign_sentence` để tái lập.
2. `transforms/palign/align.py`: student viết tiếp (prompt "Please continue from the draft ...", T=0.6, top_p=0.9, rp=1.05), resume theo `question`.
3. `transforms/palign/filter.py`: chấm đáp án bằng grader `palign`, ghép `<Begin_of_Prefix>…<End_of_Prefix>` + continuation → record chuẩn → đi tiếp vào `sgl prepare` như mọi nguồn khác (hiện `data_prep.iter_samples` đã đọc được file alpaca của P-ALIGN).
4. Train P-ALIGN = method `palign` = uniform NLL trên data đó qua backend `hf`; backend `llamafactory` giữ tuỳ chọn để kiểm tra tương đương.
5. `eval`: `test.py` + `evaluation.py` đã được port (`--palign-prompt`, `palign_grader`) → chỉ cần đưa vào `configs/eval/palign-protocol.yaml`.
6. Gate:
   - Trên ~20 mẫu, `truncate` mới chọn cùng `sufficient_sentences` với `binary_search.py` gốc dưới greedy decoding (temperature 0) — so sánh trên cùng model nhỏ.
   - Loss bước đầu của backend `hf` vs LLaMA-Factory trên cùng batch P-ALIGN khớp (≤1e-3).
   - Rescore các eval P-ALIGN cũ cho đúng số trong `HYPERPARAMETERS.md` / paper.

### Phase 6 — Dọn dẹp & tài liệu (≈2 ngày)
- Xoá `P-ALIGN/` và `SegmentSelectiveSFT/` khỏi cây (đã port xong); ghi upstream commit + license trong `docs/methods.md` và header file port (P-ALIGN: MIT; SSFT upstream không kèm file LICENSE → cần hỏi tác giả/kiểm tra repo gốc trước khi phân phối lại; latex2sympy: MIT; oat grader: Apache-2.0).
- README: cài đặt, quickstart 3 lệnh, bảng method → YAML, cách thêm signal/segmenter/selection mới (`docs/extending.md`).
- CI: `ruff` + `pytest` CPU (unit + regression), không cần GPU.
- `docs/reproducing.md`: mỗi dòng bảng trong paper ↔ một `configs/experiments/*.yaml` + lệnh.

**Tổng ước lượng: ~2.5–3 tuần làm việc**, phần lớn thời gian nằm ở các gate tương đương (Phase 4–5 cần GPU).

---

## 5. Chiến lược kiểm thử

| Lớp | Nội dung | Chạy ở |
|---|---|---|
| Unit | Toàn bộ 15 file test hiện có (giữ nguyên ý nghĩa) + segmenter, coherence filter, always_keep, answer_probe, palign round-batched search (mock LLM) | CPU, CI |
| Regression | Golden fixtures Phase 0, byte-identical | CPU, CI |
| Equivalence | IG vs `grad_analyze.py`; mask vs `train_mask.py`; truncate vs `binary_search.py`; loss hf vs LLaMA-Factory | GPU nhỏ, chạy tay trước mỗi merge Phase 4/5 |
| Reproduction | Rescore raw generations cũ → đúng bảng paper; 1 run train ngắn mỗi method | GPU, trước khi xoá code gốc |

---

## 6. Rủi ro

- **Lệch số liệu do khác backend train** (LLaMA-Factory/Unsloth → HF Trainer): khác normalization loss, packing, gradient checkpointing. Giảm thiểu: gate loss-equivalence + giữ backend gốc làm tuỳ chọn.
- **Xung đột phụ thuộc**: Unsloth pin transformers cũ hơn SGL. Giải pháp: extra `[unsloth]` cài vào env riêng; core không import Unsloth ở top-level (như `masked_dataset.py` đã làm).
- **IG tốn VRAM** ở seq 32k: giữ nguyên các tối ưu đã có trong fork (freeze weight, grad-ckpt, `ig_batch_size`, OOM fallback).
- **Tái lập P-ALIGN truncate** phụ thuộc sampling của student evaluator: so sánh bằng greedy; với sampling chỉ so phân phối `prefix_ratio`.

---

## 7. Câu hỏi mở (cần bạn quyết)

1. **Tên package**: `sgl`, hay tên trung tính hơn vì giờ chứa 3 method (vd. `stepwise`, `cotdistill`)?
2. **Giữ backend LLaMA-Factory / Unsloth?** Đề xuất: HF là mặc định, Unsloth là extra, LLaMA-Factory chỉ dùng để kiểm tra tương đương rồi bỏ.
3. **Logs/kết quả cũ**: xoá khỏi git (giữ ở GitHub Release / HF dataset) hay giữ trong `experiments/`?
4. **Code gốc P-ALIGN/SSFT**: xoá hẳn sau khi port (đề xuất, ghi commit hash upstream) hay giữ trong `references/`?
5. **Ngôn ngữ comment**: SSFT fork dùng tiếng Việt không dấu; SGL dùng tiếng Anh. Đề xuất thống nhất **tiếng Anh** cho code/comment của framework.
6. Thứ tự: đề xuất làm Phase 0→3 trước (thuần refactor, an toàn), rồi Phase 4 (SSFT) và 5 (P-ALIGN) có thể làm song song.

---

## 8. Quyết định đã chốt và tiến độ

| # | Câu hỏi | Quyết định |
|---|---|---|
| 1 | Tên package | `sgl` |
| 2 | Backend train | HF mặc định; Unsloth là extra; LLaMA-Factory chỉ để đối chứng |
| 3 | Logs / kết quả cũ | Giữ trong git. `logs_*.txt` ở root → `experiments/logs/`; `results*/` và `logs/` giữ nguyên vì là thư mục output đang được script dùng |
| 4 | Code gốc P-ALIGN / SSFT | Giữ trong `references/` (không xoá ở Phase 6) |
| 5 | Ngôn ngữ comment | Không quan trọng → giữ tiếng Anh như SGL hiện tại |

- [x] **Phase 0** — tag `paper-v1` (= `7990089`); `tests/regression/`: tokenizer BPE cục bộ + 16 mẫu
  P-ALIGN + 30 bài AIME24 đã sinh của P-ALIGN; 21 stage CPU (data_prep, masks, IWC ×5 biến thể,
  gain, provenance, trans, diagnostics, rescore, compare, SSFT split/select) → 53 output được băm
  trong `golden.json`. `pytest` chạy cả bộ (~1 phút CPU).
- [x] **Phase 1** — `src/*.py` → `src/sgl/` (git mv, giữ lịch sử); import theo package;
  mọi shell script gọi `python -m sgl.…` / `torchrun -m sgl.training.train`; `pyproject` là
  package thật (hatchling, `uv` package = true, entry points `sgl-prepare/-evaluate/-compare`);
  code gốc → `references/`; bỏ Modal volume-commit. Gate: 141 test xanh, regression giống hệt
  (trừ một chuỗi ghi chú trong `correlations.json` nhắc tên module cũ); 12 entrypoint `--help` OK;
  train 1 epoch CPU trên model Qwen2 tí hon OK. **Việc của bạn:** chạy `uv lock` trên máy có mạng
  (sandbox không tới được index PyTorch cu130).
- [x] **Phase 2** — `sgl.config` + một engine CLI chạy dưới 3 lệnh `sgl` / `palign` / `ssft`
  (mỗi lệnh chỉ nhận config của họ mình, tên run riêng: `<arm>-<track>`, `palign-<model>`,
  `ssft-<model>`); `configs/{common,sgl,palign,ssft}`; mỗi module có `build_parser()`/`main(argv)`.
- [x] **Phase 3** — `sgl.segment` (sentence / paragraph / cue + palign_sentences), `prepare --segmenter`.
- [x] **Phase 4** — `sgl.ssft` (attribution, select, build) + `sgl.data.s1k`; khớp code gốc (IG, nhãn
  train_mask trên 40 tổ hợp, file select byte-identical).
- [x] **Phase 5** — `sgl.palign` (truncate theo vòng, align, build); prompt khớp từng byte, binary
  search khớp bản tuần tự, grader tái lập 429/429 nhãn công bố.
- [x] **Phase 6** — README, `docs/methods.md`, `docs/reproducing.md`, `docs/extending.md`, CI
  (ruff + pytest CPU).

Khác plan ban đầu: code gốc giữ trong `references/` (không xoá); grader Qwen-Math của SSFT không port
(so sánh dùng math_verify như `commands.sh` của fork); các shell driver cũ vẫn giữ và gọi package.
