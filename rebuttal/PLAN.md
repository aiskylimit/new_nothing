# ICLR 2027 Rebuttal — Main Experiments Plan

**Status:** Finalized experiment scope — Ready for implementation validation

**Hardware:** 3× NVIDIA B200 180GB; GPU 3, 4, 5&#x20;

**Methods:** Long-CoT SFT, SGL, Segment Selective SFT, P-ALIGN, CoReD.

**Training seeds:** 1 seed per configuration.

**Math evaluation:** 5 evaluation seeds, 64 responses/question tổng cộng.

**OOD evaluation:** Deterministic Pass\@1, 1 response/question.

---

# 1. TRAIN

## 1.1. Methods

Tích hợp 5 methods vào một project để thống nhất training, logging và evaluation.

| ID | Method                |
| -- | --------------------- |
| M1 | Long-CoT SFT          |
| M2 | SGL                   |
| M3 | Segment Selective SFT |
| M4 | P-ALIGN               |
| M5 | CoReD                 |

Các phương pháp được phép giữ implementation và hyperparameters riêng theo cấu hình gốc, nhưng phải thống nhất cách quản lý dữ liệu, checkpoints, logging và evaluation.

**Lưu ý:** CoReD vẫn phải được train lại trên server này, dù nhóm khác đã train CoReD trên server riêng. Mục đích là đo training cost của cả 5 methods trong cùng môi trường GPU.

## 1.2. Training matrix

| Setting   | Backbone | Training dataset   | Teacher          | Methods |   Runs |
| --------- | -------- | ------------------ | ---------------- | ------- | -----: |
| A         | 7B       | s1K-1.1 Original   | Original teacher | M1–M5   |      5 |
| B         | 7B       | s1K Gemini version | Gemini           | M1–M5   |      5 |
| C         | 1.5B     | s1K-1.1 Original   | Original teacher | M1–M5   |      5 |
| **Total** |          |                    |                  |         | **15** |

**Tổng: 15 method-setting runs, mỗi configuration chỉ sử dụng một training seed.**

Không train lại backbone 8B.

### Setting A — Original 7B

**Objective:** Reproduce original setting, compare training overhead và evaluate in-domain/OOD performance.

- Train lại 5 methods trên 7B.
- Sử dụng original s1K-1.1 dataset.
- Giữ nguyên training protocol của paper.
- Log đầy đủ training cost của tất cả methods.
- Lưu final checkpoints cho math và OOD evaluation.

Đây là **setting chính để so sánh computational overhead** theo yêu cầu của reviewers.

### Setting B — Different Teacher 7B

**Objective:** Evaluate robustness khi thay đổi teacher.

- Backbone: giống Setting A.
- Training dataset: public s1K Gemini version.
- Teacher: Gemini.
- Train lại 5 methods.
- Giữ cùng training questions với Setting A.
- Giữ nguyên method-specific training protocol.

Đã xác nhận hai phiên bản s1K tương ứng về training questions.

Cần ghi lại statistics của teacher traces như token length và correctness để diễn giải kết quả. Những khác biệt này không làm thay đổi lựa chọn dataset đã chốt.

Setting B chỉ cần evaluate trên 4 math benchmarks gốc.

### Setting C — Original 1.5B

**Objective:** Evaluate improvements trên smaller backbone.

- Backbone: 1.5B.
- Training dataset: original s1K-1.1.
- Teacher: original teacher.
- Train lại 5 methods.
- Log training cost.
- Lưu checkpoints để evaluate math và OOD.

## 1.3. Training configuration

Mỗi run cần có một configuration file riêng, bao gồm:

| Category        | Required information                                       |
| --------------- | ---------------------------------------------------------- |
| Model           | Backbone, model revision, tokenizer                        |
| Data            | Dataset, split, number of samples, data revision           |
| Teacher         | Teacher name, trace source                                 |
| Optimizer       | Optimizer, learning rate, weight decay                     |
| Schedule        | LR scheduler, warmup, total steps/epochs                   |
| Batch           | Micro batch size, gradient accumulation, global batch size |
| Sequence        | Maximum sequence length, truncation rules                  |
| Precision       | BF16/FP16/FP32 hoặc mixed precision                        |
| Seed            | Fixed training seed                                        |
| Method-specific | LoRA settings, loss coefficients, KD parameters            |
| Environment     | CUDA, PyTorch, Transformers, DeepSpeed/Accelerate versions |
| Code            | Git commit hash                                            |

Những hyperparameters khác nhau giữa các methods cần được giữ theo experimental setup tương ứng, không ép tất cả methods dùng cùng cấu hình nếu điều đó làm thay đổi bản chất baseline.

## 1.4. Fair comparison requirements

- Sử dụng cùng training data cho tất cả methods trong từng setting.
- Giữ cùng backbone initialization trong một setting.
- Không thay đổi dữ liệu hoặc preprocessing giữa các methods một cách không chủ đích.
- Log số training samples, processed tokens và optimization steps để nhận biết khác biệt training budget.
- Sử dụng cùng hardware type khi so sánh training cost.
- Cố định phương pháp đo GPU-hours, FLOPs và memory.

Nếu methods có training budget khác nhau theo thiết kế, báo cáo rõ thay vì giả định chúng có chi phí tương đương.

## 1.5. Training outputs

Mỗi run phải lưu:

- Final model checkpoint.
- Tokenizer và model configuration.
- Training configuration YAML/JSON.
- Training logs.
- Training cost summary.
- Training dataset/teacher identifiers.
- Git commit hash.
- Run status và lỗi nếu có.

Với CoReD, lưu cả các artifacts cần thiết cho expert training, merging và final student distillation.

**Checkpoint phải load được độc lập và phục vụ evaluation mà không cần train lại.**

---

# 2. TRAINING LOG

## 2.1. Objective

Báo cáo computational overhead của CoReD so với Long-CoT SFT, SGL, Segment Selective SFT và P-ALIGN trên cùng hardware.

**Primary cost comparison:** Setting A — Original 7B.

Training cost trên Setting B và Setting C được lưu đầy đủ để phân tích nếu cần.

## 2.2. Required training metrics

Log những metrics sau cho toàn bộ 15 training runs:

| Metric               | Description                               |
| -------------------- | ----------------------------------------- |
| Wall-clock time      | Thời gian thực hiện training job          |
| GPU-hours            | Tổng thời gian GPU sử dụng                |
| Training FLOPs       | Measured hoặc estimated FLOPs             |
| Peak GPU memory      | Peak allocated/reserved VRAM              |
| GPU hardware         | GPU type và GPU count                     |
| Optimization steps   | Tổng optimizer steps                      |
| Processed tokens     | Tổng training tokens                      |
| Training samples     | Tổng samples đã xử lý                     |
| Effective batch size | Global training batch size                |
| Throughput           | Tokens/sec và steps/sec                   |
| Precision            | BF16/FP16/FP32 và các thiết lập liên quan |

GPU-hours được tính theo tổng GPU time của các GPU tham gia một job.

Nếu một job sử dụng nhiều GPU, cần phân biệt wall-clock time và GPU-hours. Với nhiều stages chạy song song, wall-clock end-to-end không nhất thiết bằng tổng wall-clock time từng stage.

FLOPs phải ghi rõ là giá trị đo thực tế hay ước lượng.

## 2.3. Stage-wise training cost

### CoReD

Log chi phí riêng cho:

| Stage                          | Wall-clock | GPU-hours | FLOPs | Peak VRAM |
| ------------------------------ | ---------- | --------- | ----- | --------- |
| Phase 1 — Expert training      | ✓          | ✓         | ✓     | ✓         |
| Step-weight computation        | ✓          | ✓         | ✓     | ✓         |
| Expert merging                 | ✓          | ✓         | ✓     | ✓         |
| Phase 2 — Student distillation | ✓          | ✓         | ✓     | ✓         |
| Total CoReD                    | ✓          | ✓         | ✓     | ✓         |

Total CoReD cost bao gồm tất cả các bước cần thiết để tạo ra final student.

Không chỉ đo Phase 2.

### Baselines

Ghi stage-wise cost tương ứng với pipeline của từng baseline, bao gồm những preprocessing, additional optimization hoặc teacher inference stages nếu chúng phát sinh chi phí đáng kể.

## 2.4. Final training cost table

Bảng chính cho Setting A:

| Method                | GPU-hours | FLOPs | Peak VRAM | Training tokens |
| --------------------- | --------- | ----- | --------- | --------------- |
| Long-CoT SFT          | TBD       | TBD   | TBD       | TBD             |
| SGL                   | TBD       | TBD   | TBD       | TBD             |
| Segment Selective SFT | TBD       | TBD   | TBD       | TBD             |
| P-ALIGN               | TBD       | TBD   | TBD       | TBD             |
| CoReD                 | TBD       | TBD   | TBD       | TBD             |

Báo cáo thêm:

- Compute overhead ratio so với Long-CoT SFT.
- Breakdown chi phí từng stage của CoReD.
- Throughput và peak memory comparison.
- End-to-end training cost, không chỉ cost của riêng optimizer.

Có thể tạo các bảng bổ sung cho Setting B và C từ logs đã thu thập.

---

# 3. EVALUATE

## 3.1. Evaluation objectives

Evaluation phục vụ ba mục tiêu:

1. **Math performance and reliability:** Re-evaluate 4 benchmarks gốc với 64 responses/question và 5 evaluation seeds.
2. **Different-teacher robustness:** So sánh results giữa 7B Original và 7B Gemini.
3. **Non-math OOD generalization:** Evaluate 7B và 1.5B Original trên ARC-Challenge và LogiQA 2.0.

## 3.2. Evaluation matrix

| Setting           | Math benchmarks | OOD benchmarks | Math sampling       |
| ----------------- | --------------- | -------------- | ------------------- |
| A — 7B Original   | Yes             | Yes            | 64 samples, 5 seeds |
| B — 7B Gemini     | Yes             | No             | 64 samples, 5 seeds |
| C — 1.5B Original | Yes             | Yes            | 64 samples, 5 seeds |

Tất cả evaluations được thực hiện cho cả 5 methods.

**Total evaluation checkpoints: 15.**

---

## 3.3. Mathematical benchmarks

### Datasets

Evaluate trên 4 benchmarks gốc của paper:

| Benchmark | `max_new_tokens` |       Sampling budget |
| --------- | ---------------: | --------------------: |
| AIME 2024 |            4,096 | 64 responses/question |
| AIME 2025 |            4,096 | 64 responses/question |
| AMC12     |            4,096 | 64 responses/question |
| MATH500   |            4,096 | 64 responses/question |

Cố định dataset versions, prompt templates, answer extraction và evaluation rules theo paper.

### Multiple evaluation seeds

**Đã chốt:**

- 5 evaluation seeds.
- 64 responses/question tổng cộng.
- Không chạy multiple training seeds.
- Avg\@32 và Avg\@64 được tính từ cùng tập generated responses.

Đề xuất phân bổ 64 responses qua 5 seeds:

| Evaluation seed | Responses/question |
| --------------- | -----------------: |
| Seed 1          |                 13 |
| Seed 2          |                 13 |
| Seed 3          |                 13 |
| Seed 4          |                 13 |
| Seed 5          |                 12 |
| **Total**       |             **64** |

Các seeds phải được cố định trước evaluation và dùng nhất quán cho cả 5 methods.

### Avg\@32 và Avg\@64

Generate 64 responses/question một lần.

Sắp xếp các responses theo một thứ tự cố định, cân bằng giữa seeds:

- **Avg\@32:** Tính average correctness trên 32 responses đầu tiên.
- **Avg\@64:** Tính average correctness trên toàn bộ 64 responses.

Có thể sử dụng round-robin ordering giữa 5 seeds để 32 responses đầu tiên đại diện cho cả năm seeds.

Không cần generate thêm 32 responses riêng để tính Avg\@32.

### Pass\@1 và Pass\@3

Tính Pass\@1, Pass\@3 theo định nghĩa thống nhất với paper gốc.

Cần phân biệt:

- Mean accuracy trên sampled responses.
- Pass\@k theo nghĩa xác suất có ít nhất một correct response trong k samples.
- Average Pass\@k giữa các questions.

Nếu dùng pass\@k estimator, phải sử dụng cùng estimator cho toàn bộ methods/settings.

### Statistical reliability

Báo cáo:

- Avg\@32.
- Avg\@64.
- Pass\@1 và Pass\@3.
- Performance theo từng benchmark.
- 95% confidence intervals.
- Evaluation-seed variability analysis.

Có thể tính per-seed mean accuracy từ 12–13 responses/question của từng seed.

**Giới hạn cần công bố:** Năm seed-level estimates này không phải năm lần Avg\@64 độc lập. Chúng đo sampling variability với số samples nhỏ hơn; không thể dùng để chứng minh training-seed stability.

Confidence intervals chính nên tính bằng paired bootstrap theo questions.

Không dùng standard deviation của năm seed-level estimates như thể đó là variance của năm independent training runs.

---

## 3.4. Different-teacher evaluation

### Setting B — Gemini Teacher

Evaluate toàn bộ 5 checkpoints của Setting B trên:

- AIME 2024.
- AIME 2025.
- AMC12.
- MATH500.

Giữ nguyên math evaluation protocol đã định nghĩa ở mục 3.3.

### Comparison

So sánh Setting A và Setting B trên cùng backbone 7B:

| Method                | Original Teacher | Gemini Teacher | Difference |
| --------------------- | ---------------- | -------------- | ---------- |
| Long-CoT SFT          |                  |                |            |
| SGL                   |                  |                |            |
| Segment Selective SFT |                  |                |            |
| P-ALIGN               |                  |                |            |
| CoReD                 |                  |                |            |

Báo cáo results cho từng benchmark và average trên 4 benchmarks.

Mục tiêu là kiểm tra:

- Hiệu quả distillation khi thay teacher.
- Liệu CoReD vẫn cải thiện so với baselines hay không.
- Performance gains có phụ thuộc mạnh vào original teacher không.

Không evaluate OOD trên Setting B.

---

## 3.5. Non-math OOD evaluation

### Datasets

**Đã chốt: Chỉ sử dụng ARC-Challenge và LogiQA 2.0.**

Không sử dụng CommonsenseQA.

| Configuration      | ARC-Challenge        | LogiQA 2.0         |
| ------------------ | -------------------- | ------------------ |
| Domain             | Scientific reasoning | Logical reasoning  |
| Dataset split      | Test                 | English MRC Test   |
| Questions          | 1,172                | 1,572              |
| `max_new_tokens`   | **1,024**            | **4,096**          |
| Metric             | Pass\@1 / Accuracy   | Pass\@1 / Accuracy |
| Responses/question | **1**                | **1**              |
| Decoding           | Deterministic        | Deterministic      |

### Evaluation settings

Evaluate 10 checkpoints:

- Setting A — 7B Original: 5 methods.
- Setting C — 1.5B Original: 5 methods.

**OOD decoding:**

- `temperature=0`.
- `do_sample=False` nếu dùng Hugging Face Transformers.
- vLLM/SGLang continuous batching.
- `tensor_parallel_size=1` cho mỗi GPU B200 nếu phù hợp.
- Same prompt format và answer extraction rules cho tất cả methods.
- Model được phép reasoning trước khi sinh final answer.
- Final answer format chuẩn hóa, ví dụ `Answer: B`.

Mỗi câu chỉ generate một response.

### OOD workload

| Dataset       | Questions | Checkpoints | Generations |
| ------------- | --------: | ----------: | ----------: |
| ARC-Challenge |     1,172 |          10 |      11,720 |
| LogiQA 2.0    |     1,572 |          10 |      15,720 |
| **Total**     | **2,744** |      **10** |  **27,440** |

Không chạy multiple seeds hoặc Avg\@32/64 cho OOD.

### Required OOD results

| Method                | ARC 7B | LogiQA 7B | ARC 1.5B | LogiQA 1.5B |
| --------------------- | ------ | --------- | -------- | ----------- |
| Long-CoT SFT          |        |           |          |             |
| SGL                   |        |           |          |             |
| Segment Selective SFT |        |           |          |             |
| P-ALIGN               |        |           |          |             |
| CoReD                 |        |           |          |             |

Báo cáo thêm:

- Accuracy theo benchmark.
- Paired confidence intervals.
- Invalid answer rate.
- Truncation rate.
- Mean/P95 generated response length.

Giữ nguyên max-token caps đã chốt ngay cả khi pilot phát hiện output dài; mọi vấn đề truncation cần được ghi nhận và phân tích.

---

## 3.6. Raw evaluation outputs

**Bắt buộc lưu toàn bộ generated responses.**

Không chỉ lưu accuracy.

Mỗi response cần chứa:

| Field               | Description                          |
| ------------------- | ------------------------------------ |
| `run_id`            | Method + setting + model             |
| `checkpoint_id`     | Exact model checkpoint               |
| `benchmark`         | Benchmark name/version               |
| `question_id`       | Unique problem ID                    |
| `eval_seed`         | Generation seed                      |
| `sample_id`         | Sample index                         |
| `prompt`            | Complete evaluation prompt           |
| `response`          | Full generated text                  |
| `predicted_answer`  | Parsed answer                        |
| `ground_truth`      | Correct answer                       |
| `is_correct`        | Correctness                          |
| `input_tokens`      | Input token count                    |
| `generated_tokens`  | Output token count                   |
| `finish_reason`     | EOS / length / error                 |
| `generation_config` | Temperature, top-p, max tokens, etc. |

Lưu JSONL hoặc định dạng tương đương có thể resume.

Các outputs phải cho phép:

- Recompute Pass\@1/Pass\@3.
- Recompute Avg\@32/64.
- Re-run answer extraction.
- Inspect incorrect responses.
- Analyze truncated outputs.
- Compute confidence intervals.

Không cần generate lại khi chỉ thay đổi thống kê hoặc visualization.

---

## 3.7. Statistical analysis

### Mathematical benchmarks

Báo cáo:

- Per-benchmark Pass\@1/Pass\@3.
- Avg\@32 và Avg\@64.
- Average performance across four benchmarks.
- Paired differences giữa CoReD và từng baseline.
- 95% confidence intervals.
- Evaluation-seed sensitivity.

AIME có số questions nhỏ nên confidence intervals có thể rộng; cần tránh diễn giải những differences nhỏ thành statistically significant improvements.

### Teacher robustness

Báo cáo:

- Original vs Gemini teacher results.
- Per-method performance differences.
- Relative improvements của CoReD so với baselines.
- Changes in average benchmark accuracy.

### OOD generalization

Báo cáo:

- ARC-Challenge accuracy.
- LogiQA 2.0 accuracy.
- Paired bootstrap confidence intervals theo questions.
- Differences giữa 7B và 1.5B.
- Error/truncation analysis.

### Failure analysis

Phân tích những trường hợp:

- CoReD thua baseline trên một benchmark.
- Baseline đúng nhưng CoReD sai.
- CoReD đúng nhưng baseline sai.
- Answer extraction thất bại.
- Response bị truncate do đạt max token limit.

---

## 3.8. Pre-run validation

Trước khi chạy toàn bộ experiments:

### Training validation

- [ ] Load và initialize được cả 5 methods.
- [ ] Kiểm tra original và Gemini teacher datasets.
- [ ] Kiểm tra training configs cho 3 settings.
- [ ] Kiểm tra GPU-hours và wall-clock logging.
- [ ] Kiểm tra FLOPs và memory logging.
- [ ] Kiểm tra stage-wise CoReD cost accounting.
- [ ] Train thử một số steps để xác nhận loss và gradient.
- [ ] Kiểm tra checkpoint save/resume.

### Evaluation validation

- [ ] Load được checkpoints của cả 5 methods.
- [ ] Xác nhận 4 math benchmark versions.
- [ ] Math `max_new_tokens=4096`.
- [ ] Xác nhận math sampling configuration trong code.
- [ ] Cố định 5 eval seeds và phân bổ 64 samples.
- [ ] Xác nhận Avg\@32/64 và Pass\@k implementations.
- [ ] ARC `max_new_tokens=1024`.
- [ ] LogiQA `max_new_tokens=4096`.
- [ ] OOD deterministic decoding.
- [ ] Kiểm tra answer extraction với benchmark multiple-choice.
- [ ] Kiểm tra output truncation và finish reason.
- [ ] Kiểm tra raw output saving và resume.

### Infrastructure validation

- [ ] Kiểm tra đủ GPU memory và disk space.
- [ ] Chạy vLLM/SGLang pilot để đo inference throughput.
- [ ] Kiểm tra GPU scheduling cho 3 B200.
- [ ] Lưu logs/checkpoints/results trên persistent storage.
- [ ] Mỗi job có unique ID và completion status.
- [ ] Jobs có thể restart mà không lặp lại công việc đã hoàn thành.

---

# Final Deliverables

Sau khi hoàn thành toàn bộ workload, cần có:

| Deliverable                               | Required |
| ----------------------------------------- | -------- |
| 15 main training runs                     | ✓        |
| 15 final model checkpoints                | ✓        |
| Training cost comparison trên 7B Original | ✓        |
| CoReD stage-wise computational overhead   | ✓        |
| Math benchmark results cho 3 settings     | ✓        |
| Gemini teacher robustness comparison      | ✓        |
| ARC-Challenge OOD results                 | ✓        |
| LogiQA 2.0 OOD results                    | ✓        |
| Avg\@32 / Avg\@64                         | ✓        |
| Evaluation-seed variability               | ✓        |
| Confidence intervals                      | ✓        |
| Per-benchmark error analysis              | ✓        |
| Full training logs và raw generations     | ✓        |

**Experiment scope đã chốt:**

- 5 methods.
- 3 training settings.
- 15 method-setting runs.
- Original teacher + Gemini teacher.
- 4 math benchmarks.
- 2 OOD benchmarks.
- 5 math evaluation seeds.
- 64 math samples/question tổng cộng.
- 1 deterministic OOD response/question.
- 3× B200.
- Không 8B.
- Không multiple training seeds.
- Không CoReD-specific ablations trên server này.

**Nguyên tắc triển khai:** Mỗi training và evaluation job phải lưu đủ checkpoints, configs, logs và raw results để có thể resume, aggregate và tính lại metrics mà không phải chạy lại toàn bộ pipeline.
