# Chạy nhanh OPD / OPD + IMPACT trên B200

Chế độ `fast` dùng microbatch **8**, accumulation **4** cho recipe Dolly hiện tại:
effective batch vẫn **32**, **10 epochs / 3560 updates**, cùng LoRA, learning-rate schedule,
CE/KD/IMPACT weighting và giới hạn sinh 256 tokens. Nó tối ưu cả hai nhánh chưa hoàn thành.
Teacher SFT và BI giữ cấu hình cũ; không thêm math/evaluation.

## Chuyển run đang có sang fast

Trong terminal/tmux chạy pipeline cũ, dừng launcher và tiến trình train cũ trước bằng Ctrl+C.
Kiểm tra tiến trình cũ đã dừng trước khi chạy lại. Không chạy hai launcher cùng output directory.
Run cũ chỉ resume được từ checkpoint đã lưu (mặc định mỗi 64 updates), nên Ctrl+C có thể
làm phải chạy lại tối đa khoảng 64 updates chưa được checkpoint. Không xóa outputs/checkpoints.

Trong thư mục `new_nothing/eval-impact` trên máy B200:

```bash
git pull --ff-only origin chi_eval_impact
RUN_NAME=llama2-tinyllama-opd bash project_commands.sh fast
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
FAST=1 ONLY_METHOD=opd_impact MICRO_BATCH_SIZE=8 \
STOP_AT="2026-11-01T11:50:00+07:00" RUN_NAME=llama2-tinyllama-opd \
  bash project_commands.sh train
```

`STOP_AT` là tùy chọn cho lần launch đó, không đổi ngân sách 10 epochs/cosine schedule.
Đến giờ, trainer hoàn tất update đang chạy rồi lưu checkpoint, ghi `deadline_reached` nếu
chưa đủ steps. Đây là run chưa hoàn thành, không được coi tương đương OPD đã đủ 10 epochs.
Không bảo đảm ngắt đúng giây: cần cộng thời gian update đang chạy và ghi checkpoint.
Launcher báo incomplete và dừng; chạy lại với giờ hẹn mới hoặc bỏ `STOP_AT` để tiếp tục.
Stage `train` cũng cập nhật báo cáo tổng hợp; nhánh thiếu steps được đánh dấu incomplete.

Log thêm ETA cửa sổ 50 steps: `rolling_step_seconds`, `eta_remaining_hours`,
`estimated_finish_utc`, `required_seconds_per_step`, `estimated_within_deadline`.
ETA loại trừ ghi checkpoint, có thể thay đổi theo độ dài rollout; nên đọc sau 20–50 updates.
Với 3560 updates từ đầu và khoảng 13 giờ, cần dưới 13 giây/update để có dư địa overhead.
Nếu batch 8 chưa đủ nhanh, có thể dừng ở checkpoint rồi thử `MICRO_BATCH_SIZE=16`
(accumulation tự thành 2). Chỉ giữ cấu hình đó nếu log đo được nhanh hơn và VRAM đủ;
không giả định batch lớn hơn luôn nhanh hơn.
