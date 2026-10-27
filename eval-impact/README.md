# OPD vs OPD + IMPACT — local B200

Ba file chính bám theo ba mẫu bạn vừa lưu:

| Mẫu gốc | File cho thí nghiệm này |
|---|---|
| `talas-vlm-embed.txt` | `impact-opd-b200.txt` — uv, Python 3.11, torch 2.9.0, transformers 4.56.1 |
| `download.txt` | `download.txt` — cùng cú pháp `--hf` / `--url`, đích `/mnt/local/@PROJECT@/impact_opd/...` |
| `project_commands.sh` | `project_commands.sh` — activate env, chạy tuần tự, eval và tổng hợp |

Clone repo vào thư mục local trên B200, ví dụ:

```bash
git clone --branch chi_eval_impact --single-branch \
  https://github.com/aiskylimit/new_nothing.git /mnt/local/PROJECT_NAME/new_nothing
cd /mnt/local/PROJECT_NAME/new_nothing/eval-impact
```

Nếu dùng ZIP độc lập, có thể giải nén vào `/mnt/local/<PROJECT>/impact_opd`.
`@PROJECT@` trong download manifest là placeholder cho hệ thống của bạn. Downloader Python đi kèm tự đổi đích theo `OPD_PROJECT_ROOT`; không cần sửa tay mọi dòng.
Script chạy lấy thư mục chứa nó làm project root, dùng toàn bộ model/data/output/cache từ local. Không có đường dẫn Mac trong các file chạy B200.

## Chạy

```bash
cd /mnt/local/PROJECT_NAME/new_nothing/eval-impact
bash setup_b200.sh
bash project_commands.sh download
bash project_commands.sh all
```

Nếu hệ thống đã tạo env và download bằng hai file mẫu, chạy thẳng `bash project_commands.sh all`.
Env mặc định: `/mnt/local/uvenvs/impact-opd-b200`.
`setup_b200.sh` cài torch 2.9.0 từ wheel CUDA 12.8 theo [lệnh chính thức PyTorch](https://pytorch.org/get-started/previous-versions/#v290), rồi cài phần text train/eval. Không cần build FlashAttention hoặc DeepSpeed.
Preflight kiểm tra đúng B200, CUDA, BF16 và backward trước khi load model.

Pilot ngắn trước khi mở rộng:

```bash
DOMAINS=gsm8k BI_SEEDS=42 STEPS=64 RUN_NAME=pilot-b200 bash project_commands.sh all
```

Mặc định đầy đủ: domains `gsm8k mixed dolly`, training seed `42`, calibration subsets `42 43 44`, 256 optimizer updates cho mỗi nhánh.
Hai nhánh dùng cùng model/data/seed/step/generation cap:

```text
OPD:          reverse KL trên fresh student rollouts
OPD + IMPACT: cùng reverse KL + lambda * IMPACT
```

Không dùng ETA/dual-space hay CE từ reference ở mặc định. `kd_rate=1`, KL temperature=1.
Teacher/student mặc định Qwen2.5-Math-7B-Instruct → Qwen2.5-Math-1.5B-Instruct, cùng tokenizer đã xác minh.

## Model nhỏ hơn

```bash
OPD_PROFILE=tiny bash project_commands.sh download
OPD_PROFILE=tiny RUN_NAME=tiny-b200 bash project_commands.sh all
```

Profile `tiny`: Qwen2.5-1.5B-Instruct → Qwen2.5-0.5B-Instruct. File `download_tiny.txt` có thêm hai model này.

## Path local

```text
impact_opd/
  models/<model-name>/       # config, tokenizer, safetensors
  raw/dolly/                # Dolly JSONL
  raw/gsm8k/                # train/test parquet
  raw/math500/              # test.jsonl và parquet được chuyển đổi local
  data/opd/{dolly,gsm8k,mixed}/
  cache/huggingface/
  outputs/<RUN_NAME>/
    plan.json, environment.json, inputs.json, jobs.json
    bi/<domain>-calib<seed>/
    <domain>/{opd,opd_impact}/seed42/
    base/{student,teacher}/eval/
    logs/
    final_results/
```

Prepare tạo 4.000 train + 256 calibration/domain; mixed có cùng tổng số mẫu. Các calibration subsets đều tránh tập train. Prepare kiểm tra không trùng prompt với test sets và đúng 500/1319 câu eval.
MATH-500 nguồn hiện có là `test.jsonl`; script chuyển sang parquet local để dùng native harness task. Cả train và eval bật offline sau download.

Override chung:

```bash
OPD_PROJECT_ROOT=/mnt/local/my-project/impact_opd \
OPD_VENV=/mnt/local/uvenvs/impact-opd-b200 \
CUDA_VISIBLE_DEVICES=0 RUN_NAME=three-seeds SEEDS="42 43 44" \
bash project_commands.sh all
```

Chỉ chạy từng phần bằng `prepare`, `preflight`, `bi`, `train`, `eval`, `report`, hoặc `plan` thay `all`. Dùng cùng các biến cấu hình khi tiếp tục cùng RUN_NAME.
Mỗi bước có log riêng; ví dụ `tail -f outputs/opd-b200/logs/gsm8k-opd-42.log`.
Lặp lại cùng lệnh sẽ bỏ qua job đã hoàn thành; train bị ngắt sau checkpoint sẽ resume. Nếu input/model/config đổi, dùng RUN_NAME mới để không trộn kết quả.
`MAX_MINUTES=60` giới hạn mỗi lượt train ở biên update; eval tự lấy checkpoint trong `latest.json`. Báo cáo ghi rõ số update thực tế và budget chưa đạt nếu dừng sớm.

## Eval và tổng hợp cuối

Pipeline eval cả teacher, student ban đầu, OPD và OPD+IMPACT. Nó dùng grader/few-shot native của lm-eval 0.4.13, chỉ đổi nguồn dataset sang file parquet local: MATH-500 4-shot, GSM8K 8-shot, greedy, cap 1024 token, lưu từng prediction.

Kết quả cuối ở `outputs/<RUN_NAME>/final_results/`:

- `FINAL_RESULTS.md`: bảng accuracy theo domain/method, mean/std giữa seeds và delta OPD+IMPACT − OPD.
- `accuracy_by_run.csv`, `accuracy_summary.csv`: điểm từng run và thống kê.
- `opd_vs_impact_delta.csv`: chênh lệch theo percentage points, ghép đúng domain/seed/task.
- `training_cost.csv`: steps, token sinh, giây training loop (cộng qua resume), peak allocated VRAM, loss cuối.
- `bi_all_layers.csv`: score/rank/Top-K/weight/CI của mọi layer trên mọi calibration subset.
- `bi_comparisons.csv`: Spearman/Kendall/Top-K overlap giữa các tập calibration.
- `final_results.json`: toàn bộ số liệu, phạm vi eval, trường hợp còn thiếu và trạng thái complete/incomplete.

Không có kết quả thì báo thiếu, không điền 0. Một seed không có độ lệch chuẩn giữa seeds. Debug eval có `--limit` được ghi rõ `debug_subset`, không coi là benchmark đầy đủ.
Chi tiết job wall time nằm trong `jobs.json`; vòng training có thể mất số token/thời gian khác nhau dù hai nhánh có cùng update budget.

## Kiểm tra đã thực hiện

Đã kiểm tra syntax shell, plan dùng model/data local, unit tests và native eval **offline** bằng tiny random model trên 1 câu của mỗi tập thật; mục đích xác nhận đường chạy và grader, không đo chất lượng model.
Máy kiểm tra hiện tại không có CUDA/B200, nên chưa xác nhận tốc độ/VRAM hay accuracy của cặp pretrained trên B200. Script preflight sẽ ghi môi trường thật khi bạn chạy.
