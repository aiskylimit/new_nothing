# Chạy thực nghiệm rebuttal trên 3 B200


## Chạy lần lượt từng bước, chỉ setting A và C

Chạy từ root repo. Bỏ qua hai lệnh đầu nếu đã cài thư viện và tải model/data.

```bash
# Cần Internet
bash rebuttal/project_commands.sh setup
bash rebuttal/project_commands.sh download

# Chuẩn bị và kiểm tra
bash rebuttal/project_commands.sh prepare
GPUS=3,4,5 bash rebuttal/project_commands.sh preflight

# Train hết 10 run A+C thành công rồi mới eval; không chạy setting B
GPUS=3,4,5 bash rebuttal/project_commands.sh train --settings A C && \
GPUS=3,4,5 bash rebuttal/project_commands.sh eval --settings A C

# Xuất báo cáo cho kết quả A+C
bash rebuttal/project_commands.sh report --allow-partial
```

Đổi `--settings A C` thành `--settings A` hoặc `--settings C` nếu chỉ chạy một
setting; bỏ tùy chọn này để chạy cả ba. Báo cáo dùng `--allow-partial` vì bỏ B
nên chưa đủ ma trận 15 run. Training log vẫn được lưu tự động như trước.
