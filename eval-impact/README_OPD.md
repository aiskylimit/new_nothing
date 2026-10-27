# So sánh OPD với OPD + IMPACT

Bộ env/download/chạy local B200 và báo cáo cuối: [README_B200.md](README_B200.md).

Bản sửa này so sánh **OPD thuần vs OPD + IMPACT**. DSKDv2 chỉ là tài liệu tham khảo setup; runner mới không dùng ETA, dual-space, projector logits hay pseudo-label CE của DSKDv2. Code gốc được giữ để đối chiếu.

```text
rollout y ~ student(. | prompt), sinh mới trước mỗi microbatch
L_OPD = mean_response_tokens KL(p_student(. | prompt, y_<i) || p_teacher(. | prompt, y_<i))

--method opd:        L = L_OPD
--method opd_impact: L = L_OPD + lambda * L_IMPACT
```

Mặc định reverse KL, temperature=1, `--kd-rate 1`, không reference CE. `--kd-objective forward_kl` hoặc `js_divergence` đổi divergence chung cho hai nhánh. `--kd-rate 0.5` chỉ là tùy chọn explicit CE + OPD, không phải mặc định.
Teacher frozen; sampling stop-gradient. Projection hidden-state của IMPACT dùng chung giữa các layer, train được và chỉ tồn tại ở nhánh `opd_impact`.

## Model nhỏ

| Profile | Teacher | Student |
|---|---|---|
| `math-small` (mặc định) | Qwen2.5-Math-7B-Instruct | Qwen2.5-Math-1.5B-Instruct |
| `tiny` | Qwen2.5-1.5B-Instruct | Qwen2.5-0.5B-Instruct |

Teacher/student nhận cùng prompt token IDs và sampled continuation IDs, không retokenize đáp án. Đã kiểm tra cặp Math dùng cùng tokenizer; xem `validation/direct/tokenizer_check.json`. Hai output head có số hàng padding khác nhau: KL và sampling dùng mọi token ID thực, bỏ riêng hàng không được gán token. Không bỏ token thực hay dùng vocabulary mapping.

Runner từ chối cặp khác tokenizer. Thí nghiệm này kiểm tra IMPACT trong OPD cùng tokenizer; chưa cung cấp bằng chứng cross-tokenizer OPD. Nếu khảo sát cross-tokenizer OPD cần định nghĩa riêng bridge cho output distribution.
[Teacher Math](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct), [student Math](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct).

## Chạy cặp so sánh

Chạy từ `eval-impact/`, một GPU/process:

```bash
python -m venv .venv-opd
source .venv-opd/bin/activate
pip install -r requirements-opd.txt
python -m unittest discover -s tests_opd -v

# In kế hoạch, chưa chạy model:
python -m opd.experiment --data-root data/opd --out-root runs/opd \
  --profile math-small --domains gsm8k --seeds 42 --steps 256 \
  --plan plan-opd.json

# Calibrate BI, rồi chạy hai nhánh tuần tự:
python -m opd.experiment --data-root data/opd --out-root runs/opd \
  --profile math-small --domains gsm8k --seeds 42 --steps 256 \
  --stage bi --execute --plan plan-opd.json
python -m opd.experiment --data-root data/opd --out-root runs/opd \
  --profile math-small --domains gsm8k --seeds 42 --steps 256 \
  --stage train --execute --plan plan-opd.json
```

Đổi `--profile tiny` nếu cần model bé hơn. Mở rộng bằng `--domains dolly gsm8k mixed --seeds 42 43 44` sau pilot.
Hai nhánh dùng cùng model, data, seed, prompt order, student optimizer/LR, số update và generation cap. IMPACT thêm auxiliary loss và projection hidden-state. Log số token/thời gian thực tế; cùng update budget không đồng nghĩa cùng compute thực tế.
Pilot: LoRA rank 32, LR 1e-5, accumulation 8 × batch 1, 256 updates, rollout tối đa 512 token, max context 2048, IMPACT lambda=0.25, K=6, BI tau=2.

Thêm `--max-minutes N` vào lệnh train để lưu ở biên optimizer update khi hết thời gian; lấy checkpoint từ `latest.json`. Resume bằng `--resume PATH` với cùng config/data. Checkpoint ETA/DSKDv2 cũ bị từ chối.
Run lưu config, train/excluded sample IDs, prompt/rollout text và sampled IDs, loss/token count/thời gian/VRAM, checkpoint LoRA, IMPACT projection, optimizer và RNG. Nhánh OPD không có projector t2s/s2t.
EOS-only hoặc rollout ngắn vẫn dùng cho OPD; khi chưa có response hidden-state span thì auxiliary IMPACT bằng 0. Không lọc short completions để tạo lợi thế cho nhánh nào.

## Data train và BI

`data/opd/` đã có 4.000 train + 256 calibration cho từng cấu hình Dolly, GSM8K, mixed. Mixed dùng 2.000 + 2.000 cho train, 128 + 128 cho calibration.
Mỗi domain có calibration subsets seed 42/43/44; tất cả tránh union các prompt train. Calibration subsets có thể giao nhau, được ghi nhận khi compare. Chỉ sample train splits; không dùng GSM8K test/MATH-500 cho train/calibration.

```bash
python -m opd.data --hf --out data/opd --seed 42
for domain in dolly gsm8k mixed; do
  for seed in 42 43 44; do
    python -m opd.bi collect \
      --model Qwen/Qwen2.5-Math-7B-Instruct \
      --data data/opd/$domain/calibration-seed$seed.jsonl \
      --out runs/bi/$domain-seed$seed --seed $seed --max-length 1024
  done
done
python -m opd.bi compare --runs \
  dolly=runs/bi/dolly-seed42/summary.json \
  gsm8k=runs/bi/gsm8k-seed42/summary.json \
  mixed=runs/bi/mixed-seed42/summary.json \
  --out runs/bi/domain-comparison.json
python -m opd.bi compare --runs \
  seed42=runs/bi/gsm8k-seed42/summary.json \
  seed43=runs/bi/gsm8k-seed43/summary.json \
  seed44=runs/bi/gsm8k-seed44/summary.json \
  --out runs/bi/subset-comparison.json
```

Mỗi collect lưu:

- `samples.jsonl`: ID/nguồn/domain, prompt/response, rendered text, token IDs/hash, token count, BI mọi layer theo sample.
- `layers.csv`: mọi layer theo rank, mean/std, bootstrap CI 95%, tần suất vào Top-K, selected flag và trọng số.
- `summary.json`: Top-K xếp hạng, model/tokenizer/revision, hash và cấu hình.

Compare xuất Spearman, Kendall tau-b, Top-K overlap/Jaccard, chênh lệch score/rank từng layer và số sample giao. Giữ cùng teacher/revision, tokenizer, max length và text mode.
BI lấy mean trên token hợp lệ từng sample rồi mean đều giữa sample để tránh padding/batch cuối ảnh hưởng kết quả. Mặc định prompt-only; `--text-mode prompt_response` kiểm tra thêm đáp án. Không gộp các text modes như một điều kiện.
Convention HF hidden states được ghi trong metadata: terminal state của một số kiến trúc đã qua final norm.
Tách selection/weighting bằng `--selection bi/random/first/last --weighting uniform`, rồi cố định BI set và đổi `--weighting bi`. Đổi `--top-k 2/4/6/8` với cùng frozen `--bi-summary`. Layer index 1-based; map relative depth dùng ceil theo code IMPACT.

## MATH-500 / GSM8K

```bash
python -m venv .venv-eval
source .venv-eval/bin/activate
pip install -r requirements-opd-eval.txt
python -m opd.experiment --data-root data/opd --out-root runs/opd \
  --profile math-small --domains gsm8k --seeds 42 --steps 256 \
  --stage eval --execute --plan plan-eval.json
python -m opd.report \
  --run OPD=runs/opd/math-small/gsm8k/opd/seed42/eval \
  --run OPD+IMPACT=runs/opd/math-small/gsm8k/opd_impact/seed42/eval \
  --out runs/comparison.json
```

Native tasks `minerva_math500` / `gsm8k_cot`, mặc định 4/8-shot, greedy, tối đa 1024 token, cùng chat-template protocol, thinking tắt và lưu samples. Eval teacher, student ban đầu và hai checkpoint. `--limit N` chỉ debug; manifest ghi rõ phạm vi.
Lặp cùng label `--run` với seed khác để tổng hợp mean/std/CI giữa training seeds. Một seed không có CI giữa seeds. MATH dùng native exact match; GSM dùng flexible-extraction exact match. Version harness 0.4.13 và command được lưu.

## Đã kiểm tra đến đâu?

Xem `VALIDATION.md` và `validation/direct/` cho kiểm thử đúng objective mới. Artifact validation ngoài `validation/direct/` từ lần triển khai trước dùng ETA chỉ là lịch sử; không dùng làm bằng chứng OPD thuần.
Chưa train/eval model pretrained vì máy hiện tại không có CUDA.
