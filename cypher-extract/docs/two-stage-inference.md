# Two-stage inference

Pipeline inference chạy cùng một multitask LoRA adapter qua hai bước:

1. `question + schema unit -> {"label": "YES|NO"}` bằng greedy decoding,
   không sampling;
2. merge các unit được chọn thành predicted sub-schema, rồi chạy
   `question + predicted sub-schema -> Cypher`.

Prediction path không dùng gold selector label, gold sub-schema hoặc gold
Cypher. Các trường gold chỉ được nối vào output sau generation để tính metric.

## Checkpoint

Mặc định script đọc checkpoint local dưới repository root:

```text
results
```

Với mỗi method, script đọc
`results/<model-family>/<method>/checkpoint-N`. Tên family/method khớp với
`output_dir` của config training. Inference không tải checkpoint từ Hugging
Face result repository và chọn checkpoint có `N` lớn nhất trong method directory.

Checkpoint được fingerprint từ các inference assets. Vì vậy nếu weight files
bị thay tại cùng path và cùng step, `run_config.json` không còn khớp và pipeline
sẽ từ chối reuse output cũ. Hãy chọn một `--output-dir` inference mới cho run đó.

`--methods all` gồm 13 model:

```text
teacher_lora
sft
fkl
rkl
sfkl
srkl
csd
hpd
amid
fdd_sfkl
fdd_srkl
distillm_adaptive_sfkl
distillm_adaptive_srkl
```

Base model được đọc từ `adapter_config.json`, vì vậy student adapter dùng
`Qwen/Qwen3-0.6B` còn `teacher_lora` dùng `Qwen/Qwen3-4B-Instruct-2507`.

## Chạy

Chạy toàn bộ model và cả ba benchmark:

```bash
bash scripts/infer_all_qwen3_lora.sh
```

Hoặc chạy trực tiếp trên Linux:

```bash
python scripts/infer_two_stage.py \
  --checkpoint-root results \
  --methods all \
  --datasets cypherbench,mind_the_query,neo4j_text2cypher \
  --seeds 10,42,50,100,1234 \
  --temperature 0.5 \
  --top-p 0.95 \
  --top-k 0 \
  --num-beams 1 \
  --selector-batch-size 128 \
  --generator-batch-size 16 \
  --dtype bfloat16 \
  --device cuda
```

Chạy một phần method trên một GPU cụ thể:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_two_stage.py \
  --methods teacher_lora,sft,fkl
```

Trên nhiều GPU, chạy nhiều process với danh sách method không giao nhau. Mỗi
process batch song song các schema unit trên GPU của nó; không nên load nhiều
adapter vào cùng một GPU.

Nếu base model trên Hugging Face cần authentication, đặt một trong hai biến:

```bash
export HF_READ_TOKEN=...
# hoặc HF_TOKEN
```

## Merge policy

Sub-schema luôn có đúng format:

```json
{
  "nodes": [],
  "relationships": []
}
```

Unit được deduplicate và giữ thứ tự canonical từ
`selection_inference_test.jsonl`. Khi
một relationship được chọn nhưng endpoint node chưa được chọn, node tương ứng
được thêm từ schema unit của chính sample đó. Có thể tắt bằng
`--no-relation-endpoint-closure`.

Inference đọc `selection_inference_test.jsonl` và
`generation_inference_test.jsonl`, nên example bị strict reject vì không trích
được gold sub-schema vẫn chạy qua cả hai stage. Selector metrics chỉ tính các
unit có gold `label`; Cypher metrics tính trên toàn bộ inference examples. Example
không có gold sub-schema đáng tin cậy nhận 0 ở selector qua coverage adjustment;
generator và execution metrics vẫn chấm predicted Cypher so với gold bình thường.

Selector output không parse được thành JSON `{"label": "YES"}` hoặc
`{"label": "NO"}` được ghi là `INVALID`, lưu nguyên raw output và không được
chọn vào sub-schema. Parser tạm thời vẫn nhận bare `YES/NO` để các checkpoint cũ
tiếp tục chạy. Empty schema được giữ nguyên thay vì âm thầm fallback sang
gold/full schema.

## Output và resume

```text
results/inference/qwen3/seed<seed>/<method>/<dataset>/
├── run_config.json
├── selector_predictions.jsonl
├── predicted_subschemas.jsonl
├── generator_predictions.jsonl
├── metrics.json
└── manifest.json
```

Selector và generator ghi vào `.partial` rồi mới publish file hoàn chỉnh. Nếu
process bị dừng giữa stage, lần chạy sau tiếp tục từ row cuối đã ghi. Stage đã
hoàn chỉnh được reuse.

`run_config.json` khóa đường dẫn checkpoint local, input paths, SHA-256 của input/prompt, ChatML template và generation options. Pipeline
sẽ từ chối reuse output nếu một trong các giá trị này thay đổi; khi đó dùng một
`--output-dir` khác hoặc chủ động xóa riêng directory method/dataset cũ.

Generation mặc định dùng sampling theo CypherKD (`do_sample=True`,
`temperature=0.5`, `top_p=0.95`, `top_k=0`, `num_beams=1`) và ChatML
`qwen3_nothink` đúng format LlamaFactory, không chèn thẻ `<think>`. Script chạy theo thứ tự
`seed -> method -> dataset`, hoàn tất mọi method/dataset của một seed trước khi chuyển seed. Các seed mặc định là
`10,42,50,100,1234`;
Python, NumPy, PyTorch và toàn bộ CUDA RNG được reset trước từng dataset. Seed chỉ
ảnh hưởng generator; selector dùng greedy decoding với tối đa 16 new tokens và
generator dùng tối đa 256 new tokens.

Để eval toàn bộ output sau inference (PowerShell hoặc Linux Bash), dùng:

```powershell
.\scripts\evaluate_cypher_all.ps1
```

```bash
bash scripts/evaluate_cypher_all.sh
```

Hai script đều chạy theo thứ tự `seed -> method -> dataset -> graph`, chỉ merge
khi đã chấm đủ graph của dataset. Có thể giới hạn phạm vi bằng tham số PowerShell
`-Seeds`, `-Methods`, `-Datasets`, hoặc các biến Bash `SEEDS`, `METHODS`, `DATASETS`.

Eval graph `nba` cho toàn bộ seed trên PowerShell:

```powershell
$seeds = 10, 42, 50, 100, 1234

foreach ($seed in $seeds) {
  evaluate-cypher `
    --input "results/inference/qwen3/seed$seed/sft/cypherbench/generator_predictions.jsonl" `
    --name cypherbench-db `
    --graph nba
}
```

Output được suy ra tự động dưới
`results/evaluation/qwen3/seed<seed>/sft/cypherbench/nba/`.

Sau khi đã eval đủ các graph của một seed, gộp kết quả bằng:

```powershell
merge-cypher-evaluations `
  --input-dir results/evaluation/qwen3/seed10/sft/cypherbench
```

Nếu chỉ chấm một phần metric, truyền cùng danh sách vào cả hai bước, ví dụ
`--metrics execution_accuracy executable`. Các script eval đã tự chuyển tiếp danh
sách metric sang bước merge.

CLI tạo `all_graphs_cypher_scores.jsonl` và `all_graphs_summary.json`; đồng thời
kiểm tra đủ graph của dataset, graph field đúng folder và không có ID trùng.
