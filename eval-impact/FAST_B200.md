# Chạy nhanh OPD / OPD + IMPACT trên B200

## Cấu hình hiện tại: 8.000 mẫu × 10 epochs, train xong tự eval

Mặc định launcher hiện dùng run riêng `llama2-tinyllama-opd-8k`, chọn 8.000 student train rows
với subset seed 42 giống nhau cho OPD và OPD + IMPACT. Giữ 10 epochs và effective batch 32:
`8000 / 32 × 10 = 2500` updates mỗi nhánh, giảm từ 3560 updates của run full cũ.
Subset chỉ lấy từ train; test và BI calibration giữ nguyên. ID subset và hash được lưu trong từng run.
Hai student bắt đầu từ base; không resume checkpoint full-data để đổi ngân sách cosine giữa chừng.

Teacher đã SFT được lấy từ `outputs/llama2-tinyllama-opd/teacher_sft/latest.json` và phải completed.
Đổi `TEACHER_FROM_RUN` nếu teacher nằm trong run tên khác; hoặc cung cấp `TEACHER_ADAPTER` trực tiếp.
Teacher không train lại; BI được tính cho run mới với teacher này.

Script gộp cài dependency eval, tải ba bộ test bổ sung và chạy đủ hai nhánh:

```bash
bash run_8k_b200.sh
```

Script dùng batch 8, effective batch 32, seed subset 42 và run 8k riêng;
xóa các biến `ONLY_METHOD`/`STOP_AT` cũ trong phiên script để chạy đủ ngân sách.
Chạy trong tmux và dừng launcher/GPU job cũ trước khi bắt đầu.
Nếu teacher ở run tên khác, dùng `TEACHER_FROM_RUN=<tên-run> bash run_8k_b200.sh`;
cũng có thể truyền `TEACHER_ADAPTER=<đường-dẫn>`.
Environment B200 và model/raw Dolly từ lần chạy trước cần còn trên máy.

Các bước tương đương khi muốn chạy riêng từng lệnh:

```bash
bash project_commands.sh setup-eval
bash project_commands.sh download-eval
MICRO_BATCH_SIZE=8 EVAL_BATCH_SIZE=16 bash project_commands.sh fast
```

Lệnh cuối chạy BI → OPD 2500 updates → eval → OPD + IMPACT 2500 updates → eval → báo cáo,
trên Dolly / SelfInst / VicunaEval / S-NI, không có UnNI. Mỗi nhánh dùng 5 decoding seeds.
Chạy lại cùng lệnh để resume run 8k từ checkpoint đã lưu. Run 4k cũ được giữ riêng;
không resume checkpoint 4k để đổi sang 8k giữa chừng.
Không chạy đồng thời launcher cũ và mới trên cùng GPU nếu muốn đo thời gian ổn định.

Nếu BI của run 8k đã có, có thể tiếp tục train + eval mà không chạy lại bước prepare:

```bash
FAST=1 MICRO_BATCH_SIZE=8 EVAL_BATCH_SIZE=16 bash project_commands.sh train
```

`TRAIN_SAMPLES` và `TRAIN_SUBSET_SEED` chỉ dùng khi tạo run mới; đổi subset của saved plan bị từ chối.
`TRAIN_SAMPLES=0` giữ toàn bộ dữ liệu cho run full cũ. Phần chuyển run cũ bên dưới áp dụng cho cấu hình full.

Với `MICRO_BATCH_SIZE=8`, chế độ `fast` dùng microbatch **8**, accumulation **4** cho recipe Dolly hiện tại
(mặc định code hiện tại là microbatch 16, accumulation 2):
effective batch vẫn **32**, cùng LoRA, learning-rate schedule,
CE/KD/IMPACT weighting và giới hạn sinh 256 tokens. Nó tối ưu cả hai nhánh chưa hoàn thành.
Không thêm math evaluation.

## Chuyển run đang có sang fast

Trong terminal/tmux chạy pipeline cũ, dừng launcher và tiến trình train cũ trước bằng Ctrl+C.
Kiểm tra tiến trình cũ đã dừng trước khi chạy lại. Không chạy hai launcher cùng output directory.
Run cũ chỉ resume được từ checkpoint đã lưu (mặc định mỗi 64 updates), nên Ctrl+C có thể
làm phải chạy lại tối đa khoảng 64 updates chưa được checkpoint. Không xóa outputs/checkpoints.

Trong thư mục `new_nothing/eval-impact` trên máy B200:

```bash
git pull --ff-only origin chi_eval_impact
TRAIN_SAMPLES=0 RUN_NAME=llama2-tinyllama-opd bash project_commands.sh fast
```

Giữ nguyên `OPD_PROJECT_ROOT`, `OPD_VENV`, `CUDA_VISIBLE_DEVICES` nếu trước đó đã đặt riêng.
`fast` tự đọc `outputs/<RUN_NAME>/plan.json` nếu có. Plan đã lưu quyết định model, teacher adapter,
dataset, seed, phương pháp và tổng steps; các biến tạo plan mới như `STEPS`, `METHODS`, `SEEDS`
không thay đổi plan đã lưu khi chạy `fast`. Chỉ cách chia microbatch, gradient checkpointing và
attention backend của student job chưa hoàn thành được cập nhật.

- Teacher SFT, BI và nhánh train đã hoàn thành được bỏ qua khi ledger/artifact hợp lệ.
- Nhánh còn dở resume từ `latest.json`, giữ optimizer, scheduler, thứ tự mẫu và vị trí epoch.
- Nhánh chưa bắt đầu chạy từ student base như trước.
- Không ghi đè checkpoint cũ khi đổi batch; config gốc được lưu trong `resume_history/` và plan gốc trong `plan_history/`.
- Các dòng metrics/rollout sau checkpoint được chuyển vào `resume_history/` để không đếm lặp khi chạy lại phần chưa lưu.

Sau khi đổi sang fast, tiếp tục dùng `fast` cho run đó. Ví dụ tổng hợp riêng:

```bash
FAST=1 RUN_NAME=llama2-tinyllama-opd bash project_commands.sh report
```

Các launcher mới có file lock chống chạy trùng, nhưng launcher cũ chưa có lock nên vẫn phải dừng nó trước.

## Run mới và điều chỉnh bộ nhớ

```bash
RUN_NAME=llama2-fast-new bash project_commands.sh fast
```

Mặc định fast dùng SDPA, tắt gradient checkpointing để giảm tính lại activation.
Nếu GPU có workload khác hoặc OOM, sau khi tiến trình lỗi đã dừng, giảm microbatch:

```bash
MICRO_BATCH_SIZE=4 RUN_NAME=llama2-tinyllama-opd bash project_commands.sh fast
```

Accumulation tự chuyển thành 8 để giữ effective batch 32. Có thể bật lại checkpointing:

```bash
MICRO_BATCH_SIZE=4 GRADIENT_CHECKPOINTING=1 RUN_NAME=llama2-tinyllama-opd bash project_commands.sh fast
```

Microbatch phải chia hết effective batch của plan. Không tự động giảm batch giữa một update.
Nếu lỗi xảy ra trước checkpoint đầu tiên, giữ thư mục lỗi để kiểm tra và dùng RUN_NAME mới.

## Code thay đổi gì?

Rollout được sinh theo batch, left-padding prompt, giữ EOS thật và bỏ padding sau EOS.
Train dùng right-padding và mask padding khỏi loss. CE và KD tính trung bình theo từng mẫu
rồi mới trung bình batch, giữ cách cân mẫu của bản microbatch 1 (không tăng trọng số câu dài).
IMPACT cũng giữ cách cân từng mẫu, mẫu EOS-only đóng góp zero như trước.

IMPACT tính các span một lần trên CPU, gom các response deltas để chiếu trong một phép nhân ma trận
mỗi layer và tính cosine theo tensor. Phép chiếu tuyến tính được đưa ra sau nội suy/lấy hiệu,
tránh chiếu toàn bộ prompt/padding và vòng lặp GPU từng token.

Đổi batch và thứ tự tính toán không bảo đảm giống từng bit: sampling RNG, dropout và sai số số thực
có thể khác. Config của mỗi checkpoint và lịch sử resume ghi lại thiết lập. Nếu OPD đã hoàn thành
với batch 1, không dùng chênh lệch thời gian với IMPACT batch 8 để kết luận về chi phí thuật toán.
Để so tốc độ công bằng, hai nhánh cần cùng chế độ fast.

## Xem tốc độ thực tế

```bash
tail -f outputs/llama2-tinyllama-opd/logs/dolly-dskd-opd_impact-10.log
```

`metrics.jsonl` thêm `step_seconds`, `rollout_seconds`, `learning_seconds`, `samples_per_second`,
`generated_tokens_per_second`, `micro_batch_size`, `grad_accum`. Các phần thời gian không gồm
toàn bộ overhead checkpoint; `step_seconds` là thời gian mỗi update, bỏ qua checkpoint write.
`training_cost.csv` cũng ghi batch/checkpointing và tốc độ step cuối.
Đo vài chục updates sau warm-up trên B200 trước khi ước lượng ETA mới.

Đã kiểm tra trên CPU bằng model LLaMA nhỏ: rollout padding/EOS, giá trị loss và gradients so với
cách tính cũ, LoRA + IMPACT train, resume checkpoint cũ từ microbatch 1 sang 2, optimizer/scheduler,
thứ tự mẫu và plan migration. Chưa benchmark model thật trên B200; không cam kết hệ số tăng tốc.

## Chạy riêng IMPACT qua đêm, theo giờ hẹn

`ONLY_METHOD=opd_impact` chọn riêng student job IMPACT ngay cả khi plan đã lưu có cả hai nhánh;
không xóa kết quả/case baseline. Với stage `train`, teacher SFT và BI phải đã hoàn thành.
Đặt `FAST=1` để đọc plan cũ, giữ tổng steps và resume nhánh IMPACT nếu có checkpoint.

Ví dụ phiên tối 31/10/2026 đến trưa 01/11/2026 (giờ Việt Nam), chừa khoảng 10 phút để lưu:

```bash
TRAIN_SAMPLES=0 FAST=1 ONLY_METHOD=opd_impact MICRO_BATCH_SIZE=8 \
STOP_AT="2026-11-01T11:50:00+07:00" RUN_NAME=llama2-tinyllama-opd \
  bash project_commands.sh train
```

`STOP_AT` là tùy chọn cho lần launch đó, không đổi ngân sách 10 epochs/cosine schedule.
Đến giờ, trainer hoàn tất update đang chạy rồi lưu checkpoint, ghi `deadline_reached` nếu
chưa đủ steps. Đây là run chưa hoàn thành, không được coi tương đương OPD đã đủ 10 epochs.
Không bảo đảm ngắt đúng giây: cần cộng thời gian update đang chạy và ghi checkpoint.
Lệnh shell `train` mặc định chạy `train-eval`: train xong tự eval Dolly, SelfInst, VicunaEval và S-NI,
rồi tổng hợp báo cáo. Khi tới `STOP_AT` trước khi đủ steps, pipeline eval checkpoint tạm và ghi rõ incomplete;
chạy lại với giờ hẹn mới hoặc bỏ `STOP_AT` để tiếp tục. Dùng `AUTO_EVAL=0` nếu chỉ muốn train.
`ONLY_METHOD=opd_impact` cũng chỉ chọn eval của nhánh IMPACT; kết quả baseline cũ được giữ để so sánh.

Log thêm ETA cửa sổ 50 steps: `rolling_step_seconds`, `eta_remaining_hours`,
`estimated_finish_utc`, `required_seconds_per_step`, `estimated_within_deadline`.
ETA loại trừ ghi checkpoint, có thể thay đổi theo độ dài rollout; nên đọc sau 20–50 updates.
Với 3560 updates từ đầu và khoảng 13 giờ, cần dưới 13 giây/update để có dư địa overhead.
Nếu batch 8 chưa đủ nhanh, có thể dừng ở checkpoint rồi thử `MICRO_BATCH_SIZE=16`
(accumulation tự thành 2). Chỉ giữ cấu hình đó nếu log đo được nhanh hơn và VRAM đủ;
không giả định batch lớn hơn luôn nhanh hơn.

## Eval Dolly, SelfInst, VicunaEval và S-NI

Chạy từ `new_nothing/eval-impact` sau khi cập nhật code trên B200:

```bash
bash project_commands.sh setup-eval
bash project_commands.sh download-eval
FAST=1 MICRO_BATCH_SIZE=8 EVAL_BATCH_SIZE=16 RUN_NAME=llama2-tinyllama-opd \
  bash project_commands.sh eval
```

Để tiếp tục nhánh IMPACT rồi tự eval ngay sau train (chạy hai lệnh setup/download phía trên một lần trước):

```bash
FAST=1 ONLY_METHOD=opd_impact MICRO_BATCH_SIZE=8 EVAL_BATCH_SIZE=16 \
RUN_NAME=llama2-tinyllama-opd bash project_commands.sh train
```

`download-eval` chỉ tải ba bộ test bổ sung và dựng JSONL local; Dolly dùng test split đã prepare
cho run hiện tại. Các revision và SHA-256 được cố định trong `opd/benchmarks.py`.
Sau download, eval chạy offline. Không cần train lại. `FAST=1` đọc và mở rộng plan cũ;
lệnh `eval` không chạy các train jobs dù có thiết lập microbatch.

| Benchmark | Split nguồn | Số mẫu |
|---|---|---:|
| Dolly | test của run Dolly hiện tại | 500 |
| SelfInst | MiniLLM/self-inst, valid.jsonl | 242 |
| VicunaEval | MiniLLM/Vicuna, valid.jsonl | 80 |
| S-NI | MiniLLM/sinst, 11_/valid.jsonl | 1.694 |

Nguồn: [MiniLLM data](https://github.com/microsoft/LMOps/blob/main/minillm/README.md#21-resources),
[SelfInst](https://huggingface.co/datasets/MiniLLM/self-inst),
[Vicuna](https://huggingface.co/datasets/MiniLLM/Vicuna),
[S-NI](https://huggingface.co/datasets/MiniLLM/sinst).
UnNI không nằm trong danh sách tải/chạy hoặc mẫu số Avg.

Hai nhánh OPD và OPD + IMPACT dùng checkpoint từ `latest.json`; mỗi bộ chạy 5 decoding seeds
10/20/30/40/50, sampling T=1, top_p=1, top_k=0, prompt tối đa 256 tokens và response tối đa 256 tokens.
Đây là các bộ instruction-following với ROUGE-L F1 và normalized exact match;
VicunaEval ở đây chấm theo reference, không dùng LLM-as-a-judge.
S-NI giữ mọi reference trong danh sách và lấy điểm cao nhất theo từng metric.

Kết quả từng bộ bổ sung lưu ở `<train_dir>/eval_benchmarks/<benchmark>/`.
Báo cáo trong `outputs/<RUN_NAME>/final_results/` gồm:

- `FINAL_RESULTS.md`: bảng Dolly / SelfInst / VicunaEval / S-NI / Avg.
- `benchmark_summary.csv`, `benchmark_metrics.csv`, `benchmark_per_seed.csv`.
- `benchmark_opd_vs_impact.csv`: chênh lệch IMPACT − OPD theo từng bộ.

Avg. là trung bình đều ROUGE-L trên 4 bộ, không cân theo số mẫu.
Chỉ tính khi đủ toàn bộ test, train hoàn thành và các bộ dùng cùng checkpoint/cấu hình sinh.
Checkpoint đang train vẫn có thể eval, nhưng báo cáo ghi provisional và không tính Avg.
Khi checkpoint thay đổi, lần eval kế tiếp lưu kết quả cũ trong `eval_history/` và chấm lại.
Điểm trên ảnh paper không được coi là tái lập chính xác chỉ vì dùng cùng tên benchmark.

Có thể chỉ chạy Dolly với `EVAL_BENCHMARKS=dolly`; mặc định là cả 4 bộ ở trên.
5 decoding seeds đo biến thiên lúc sinh, không thay thế nhiều training seeds.
