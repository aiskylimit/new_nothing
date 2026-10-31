# OPD vs OPD + IMPACT — LLaMA2 → TinyLLaMA trên B200

Mặc định hiện tại: **LLaMA2-7B → TinyLLaMA-1.1B**, train trên Dolly, giữ cả **OPD** và **OPD + IMPACT**. **Không tải/chạy MATH-500 hoặc GSM8K**, không tạo điểm math giả khi tổng hợp.

## Chạy trên B200

```bash
git clone --branch chi_eval_impact --single-branch \
  https://github.com/aiskylimit/new_nothing.git /mnt/local/PROJECT_NAME/new_nothing
cd /mnt/local/PROJECT_NAME/new_nothing/eval-impact
bash setup_b200.sh
# HF_TOKEN phải thuộc tài khoản đã được cấp quyền LLaMA-2.
bash project_commands.sh download
bash project_commands.sh fast
```

Nếu đã clone, cập nhật branch bằng `git pull --ff-only` trước khi chạy. Có thể dùng token từ môi trường `HF_TOKEN`; không ghi token vào manifest/repo. Nếu dùng `hf auth login`, đặt `HF_HOME="$PWD/cache/huggingface"` để login và script dùng cùng cache. [LLaMA-2](https://huggingface.co/meta-llama/Llama-2-7b-hf) yêu cầu quyền truy cập trước khi tải.

`fast` dùng batch 8 × accumulation 4, SDPA, tắt gradient checkpointing cho student và vector hóa IMPACT; vẫn giữ effective batch 32, 10 epochs. Nếu đã có run, dừng launcher/train cũ rồi dùng cùng `RUN_NAME` với `fast` để resume checkpoint. Xem [FAST_B200.md](FAST_B200.md) về resume, bộ nhớ và đo tốc độ. `all` giữ cách chạy batch 1 để tương thích plan cũ.

Ba file để set máy vẫn là `impact-opd-b200.txt` (env uv/Python 3.11/torch 2.9.0 CUDA 12.8), `download.txt` (model và dataset local), `project_commands.sh` (chạy tuần tự). Env mặc định `/mnt/local/uvenvs/impact-opd-b200`; project root tự lấy từ thư mục script. Có thể override `OPD_VENV`, `OPD_PROJECT_ROOT`, `CUDA_VISIBLE_DEVICES`.

Preflight kiểm tra B200, BF16 và backward. Từ sau download, các bước dùng model/data/cache local và bật offline. Downloader lưu commit của model trong `download_manifest.json`; preflight kiểm tra tokenizer trước khi bắt đầu teacher SFT. Dolly được tải từ Hugging Face vào `raw/dolly_reference/`; thư mục `data/` không còn lưu trên GitHub.

```text
prepare Dolly → preflight → teacher SFT → BI ba subsets
             → OPD → OPD + IMPACT → tổng hợp loss/chi phí/BI
```

Teacher được fine-tune LoRA trước, rồi đóng băng và dùng chung cho hai nhánh. Student bắt đầu từ base TinyLLaMA ở mỗi nhánh. Nếu đã có teacher adapter phù hợp, dùng đường dẫn local để bỏ qua bước SFT:

```bash
TEACHER_ADAPTER=/mnt/local/PROJECT_NAME/teacher_adapter \
RUN_NAME=llama2-existing-teacher bash project_commands.sh all
```

## Data trên Hugging Face

Đã upload data đang dùng tại [pkcii/eval-impact-dolly](https://huggingface.co/datasets/pkcii/eval-impact-dolly), gồm `opd` (train/BI validation/test), `bi` (ba calibration subsets), và `original` (split gốc để đối chiếu). Chỉ có Dolly; không có MATH/GSM hoặc trọng số model. License data: CC BY-SA 3.0.

```python
from datasets import load_dataset

data = load_dataset("pkcii/eval-impact-dolly", "opd", revision="54795e8ce55e166971eeec88c283e66a5e228241")
bi = load_dataset("pkcii/eval-impact-dolly", "bi", revision="54795e8ce55e166971eeec88c283e66a5e228241")
```

Bản JSONL dùng cho local trainer nằm trong [jsonl/](https://huggingface.co/datasets/pkcii/eval-impact-dolly/tree/main/jsonl); source/checksum và thay đổi split được lưu ở `provenance.json` / `manifest.json`. Lệnh `project_commands.sh download` tải ba JSONL gốc từ revision đã pin trong `HF_DATASET.json` và kiểm tra checksum. Bước `prepare` dựng train/calibration từ `raw/dolly_reference/original/`, hoàn toàn offline. Các thư mục `raw/` và `data/` được gitignore; GitHub chỉ giữ code và metadata/link data. Link model base được tác giả liệt kê trong [DSKD Models](https://github.com/songmzhang/DSKD#models); không phải checkpoint teacher đã SFT sẵn.

## Cấu hình tham chiếu

Đối chiếu [repo DSKDv2](https://github.com/songmzhang/DSKDv2/tree/ae4993e8b857ca001742de651498dc627138dd32) và [paper, Table I / Experimental Setup](https://arxiv.org/html/2504.11426v1#S4.SS1). Chi tiết nguồn và phần khác biệt nằm ở [DSKDV2_REFERENCE.md](DSKDV2_REFERENCE.md).

| Thành phần | Mặc định |
|---|---|
| Teacher | `meta-llama/Llama-2-7b-hf` — base + Dolly LoRA |
| Student | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` — base |
| Student training | 10 epochs; effective batch 32 = microbatch 1 × accumulation 32 × 1 GPU |
| Optimizer | AdamW, LR 1e-3, cosine, weight decay 0.01, clip norm 1 |
| LoRA | rank 256, alpha 8, dropout 0.1; q_proj/v_proj |
| Context / prompt / rollout | 512 / 256 / tối đa 256 tokens |
| Prompt | Dolly đã xử lý, giữ nguyên chuỗi; không chat template/BOS/newline thêm |
| Objective chung | `0.5 * reference CE + 0.5 * forward KL` trên fresh student rollouts; KD temperature 2; không nhân T² |
| OPD + IMPACT | Cùng objective + `0.25 * IMPACT`; BI Top-6, tau 2 |
| Rollout | Sample temperature 1, top_p 1, top_k 0; OPD ngay từ update đầu |
| Training seed / BI subsets | 10 / 42, 43, 44 |

Đây là **OPD với recipe CE+KD tham chiếu**, không phải pure-KD (`kd_rate=1`). Hai nhánh không dùng DSKDv2/ETA/dual-space loss. Cặp này cùng tokenizer; runner kiểm tra vocabulary/tokenizer trước khi train.

Dataset gốc có 11.435 train / 1.000 dev / 500 test. Prepare loại **14 train rows** trùng prompt test, còn **11.421 train**, giữ các train duplicates khác để ít đổi split. BI lấy 256 prompt/subset từ 972 dev prompts duy nhất không trùng train/test. Không dùng test để tune BI. Full run mỗi nhánh có `floor(11421 / 32) × 10 = 3560` optimizer updates; bỏ batch cuối không đủ và shuffle lại mỗi epoch.

Chạy pilot để kiểm tra đường chạy trước khi chạy full:

```bash
TEACHER_STEPS=2 STEPS=2 BI_SEEDS=42 RUN_NAME=llama2-pilot \
  bash project_commands.sh all
```

Pilot dùng teacher mới train 2 updates, chỉ để kiểm tra, không coi là kết quả thực nghiệm. Full run dùng RUN_NAME khác và bỏ `TEACHER_STEPS`/`STEPS`. `STEPS=0` mặc định nghĩa là đủ 10 epochs.

```bash
SEEDS="10 20 30" RUN_NAME=llama2-three-seeds bash project_commands.sh all
```

Teacher seed 10 cố định và dùng chung. Đây là nhiều **training seeds**, khác năm **decoding seeds** trong paper.

## Tiếp tục và kết quả

Có thể chạy riêng `prepare`, `preflight`, `teacher`, `bi`, `train`, `report`, `plan`. Khi tách stages, chạy `teacher` trước `bi`/`train`. Lặp lại `all` với cùng cấu hình sẽ bỏ qua job hoàn thành và resume từ checkpoint. Cấu hình/input thay đổi phải dùng RUN_NAME mới. Không kéo dài tổng step budget của một run cosine đã bắt đầu.

`MAX_MINUTES=60` giới hạn mỗi lượt student train tại biên update; không giới hạn teacher SFT. Chạy lại lệnh để tiếp tục. Để chạy lâu trên máy local B200, dùng tmux/session bền vững.

```text
outputs/llama2-tinyllama-opd/
  plan.json, environment.json, inputs.json, jobs.json
  teacher_sft/checkpoint-*/
  bi/dolly-dskd-calib{42,43,44}/
    samples.jsonl            # ID, prompt, input IDs, BI từng sample
    summary.json, layers.csv # mọi layer, rank, Top-K, weight, bootstrap CI
  bi/comparisons.json        # score/rank delta từng layer và tương quan
  dolly-dskd/{opd,opd_impact}/seed10/
    config.json, metrics.jsonl, rollouts.jsonl, latest.json, checkpoint-*/
  logs/
  final_results/
    FINAL_RESULTS.md, training_cost.csv
    bi_all_layers.csv, bi_comparisons.csv, final_results.json
```

```bash
tail -f outputs/llama2-tinyllama-opd/logs/dolly-dskd-opd-10.log
bash project_commands.sh report
```

Báo cáo ghi loss, updates, token rollout, thời gian training loop, VRAM và trạng thái; BI có score/rank/Top-K/CI, Spearman/Kendall/Top-K overlap giữa các subsets. Hiện các subsets đều từ Dolly dev: chúng đo độ nhạy theo **lấy mẫu**, chưa chứng minh thay đổi giữa các dataset. MATH-500/GSM8K bị tắt nên không báo accuracy hoặc delta accuracy cho run này.

Đã kiểm tra unit tests và đường chạy bằng tiny random LLaMA trên CPU; chưa train cặp pretrained hay đo tốc độ/VRAM trên B200 tại máy hiện tại. Xem [VALIDATION_LLAMA.md](VALIDATION_LLAMA.md).
