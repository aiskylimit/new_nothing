# cypher-extract

Pipeline tạo dữ liệu huấn luyện cho schema grounding trong Text-to-Cypher.
Nó chuẩn hóa schema của CypherBench, Mind-the-Query và Neo4j Text2Cypher về
cùng một graph representation, suy ngược gold sub-schema từ gold Cypher, rồi
xuất hai loại dữ liệu:

```text
Schema selection:  question + schema unit -> label 0/1
Cypher generation: question + gold sub-schema -> gold Cypher
```

## Requirements

- Python **3.10+**. Code dùng type union `X | Y`, không hỗ trợ Python 3.9 trở xuống.
- Không có package bên thứ ba: pipeline và test chỉ dùng Python standard library.
- Cần đặt các benchmark muốn xử lý dưới `benchmarks/` theo cấu trúc:

  ```text
  benchmarks/
  ├── Cypherbench/
  ├── Mind_the_query/
  └── Neo4j_Text2Cypher/
  ```

  Mặc định pipeline chạy cả ba benchmark. Có thể giới hạn bằng `--sources`.

Kiểm tra phiên bản Python:

```powershell
python --version
```

Tùy chọn tạo virtual environment (không cần cài dependency):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

`requirements.txt` được giữ để lệnh cài đặt thống nhất giữa các môi trường;
hiện tại file không khai báo package nào:

```powershell
python -m pip install -r requirements.txt
```

## Chạy tests

```powershell
python -m unittest discover -s tests -v
```

## Tạo dữ liệu

Xem toàn bộ tham số:

```bash
python scripts/build_schema_grounding_data.py --help
```

### 1. Smoke test (Kiểm tra nhanh)

Chạy thử 25 mẫu của **mỗi** benchmark (`cypherbench`, `mind_the_query`, `neo4j_text2cypher`):

```bash
python scripts/build_schema_grounding_data.py \
  --output-dir /tmp/schema-grounding-smoke \
  --max-examples 25 \
  --negative-ratio 4 \
  --overwrite
```

---

### 2. CypherBench (Train, Dev, Test)

CypherBench dùng các file prompt/response có schema nhúng: `train.jsonl`,
`dev.jsonl`, và `test.jsonl`. Tạo đủ ba split trong một output directory:

```powershell
python scripts\build_schema_grounding_data.py `
  --benchmarks-root benchmarks `
  --sources cypherbench `
  --splits train,dev,test `
  --output-dir data\cypherbench_schema_grounding_full `
  --overwrite
```

### 3. Neo4j Text2Cypher (Train, Test)

Neo4j Text2Cypher dùng `train.json` và `test.json`, vì các file `.jsonl` chỉ là
prompt/response export và không chứa schema cần cho sub-schema extraction.

```powershell
python scripts\build_schema_grounding_data.py `
  --benchmarks-root benchmarks `
  --sources neo4j_text2cypher `
  --splits train,test `
  --output-dir data\neo4j_text2cypher_schema_grounding_full `
  --overwrite
```

### 4. Mind-the-Query (Train, Test)

Mind-the-Query dùng `train_val.json` và `test.json` vì các JSONL tương ứng
không mang schema metadata.

```powershell
python scripts\build_schema_grounding_data.py `
  --benchmarks-root benchmarks `
  --sources mind_the_query `
  --splits train,test `
  --output-dir data\mind_the_query_schema_grounding_full `
  --overwrite
```

Chỉ thêm `--negative-ratio N` khi muốn giới hạn số negative schema unit cho mỗi
positive unit trong `selection_<split>.jsonl`; nó không thay đổi generation data.

### Lọc selector stage-1

Tạo tập selector có prior nhãn và prior `node/relation` theo từng nhãn khớp với
phân phối gộp theo số row của selector test CypherBench, Mind-the-Query và Neo4j
Text2Cypher. Mọi row `YES` vẫn thuộc một cặp contrast `YES/NO` trên cùng
question; các row `NO` bổ sung được lấy từ question riêng biệt. Tập vẫn phân bổ
số row theo tỷ lệ question của mỗi graph và cover mọi schema unit × label quan
sát được trong `selection_train.jsonl`:

```powershell
python scripts\filter_selector_stage1.py `
  --input-dir data\cypherbench_schema_grounding_full `
  --output-dir data\cypherbench_schema_grounding_full_final `
  --target-positive-ratio 0.30 `
  --seed 42 `
  --overwrite
```

Script giữ nguyên generation data và selection dev/test; chỉ thay
`selection_train.jsonl` và cập nhật manifest với policy lọc. Mặc định số
selector row bằng số row trong `generation_train.jsonl`. Thiết lập train hiện tại
chủ động tăng positive prior lên `30%` để ưu tiên recall: với `6,827` generator,
đầu ra có `2,048 YES`, `4,779 NO`: `2,048` contrast pair và `2,731`
negative-only row. Hai thành viên
của mỗi pair nằm liền nhau và mọi question chỉ được chọn một lần. Quota
node/relation vẫn lấy theo phân phối gộp của ba tập test. Có thể dùng
`--target-rows` hoặc đổi `--target-positive-ratio` để chạy ablation khác.

### Chuẩn bị prompt multitask

Chuẩn bị `train.jsonl` và `eval.jsonl` sao cho batch có cả generator và
selector khi số row cho phép, không lặp lại bất kỳ row nguồn nào. Các cặp
selector cùng question được xếp trước, sau đó là các negative-only row; với
`batch-size=2`, hai thành viên của một pair nằm trong hai mixed batch liên tiếp
(`generator + selector`); với `batch-size=4`, `8`, ... pair được giữ nguyên trong
prepared batch khi có đủ selector capacity. Batch còn lại chỉ chứa generator.
Test generator và selector được ghi thành hai file riêng. Không bật shuffle từng
row ở data loader vì sẽ phá task mix và contrast pairing. Trainer tự suy ra số
batch chứa contrast pair, vùng negative-only có thể drop và kích thước block
shuffle từ `batch_size` trong layout. Khi train 2 GPU với
`batch-size=2`, hai selector row `YES/NO` của cùng question đi vào cùng một
global micro-step, còn mỗi GPU vẫn nhận một mixed batch `generator + selector`.
Với CypherBench final hiện tại (`6,827` generator, `6,827` selector) và
`batch-size=2`, train có `6,827` mixed batch (`1 generator + 1 selector`), không
còn batch generator-only; mọi mẫu nguồn chỉ xuất hiện một lần.

Generative eval giữ nguyên metrics hỗn hợp hiện tại nhưng dùng decoding theo
đúng task inference: selector chạy greedy, tối đa 16 token để trả JSON
`{"label": "YES"}` hoặc `{"label": "NO"`; generator giữ sampling
`temperature=0.5`, `top_p=0.95`, `top_k=0` và tối đa 256 token.

Mặc định code đọc dataset đã tải tại
`/mnt/local/aiskylimit_new_nothing/cypher-extract/datasets/cypher-extract-data`.
Có thể đặt `CYPHER_DATA_ROOT` nếu cần chạy từ vị trí khác.

```powershell
python scripts\prepare_multitask_prompts.py `
  --batch-size 2 `
  --overwrite
```

Chuyển bốn file prompt/response vừa tạo sang OpenAI chat JSONL và đăng ký tên
dataset local cho LlamaFactory:

```powershell
python scripts\prepare_llamafactory_data.py `
  --overwrite
```

Mỗi row đầu ra chỉ giữ `messages` theo thứ tự `system -> user -> assistant`.
Các config train dùng tên `cypher_prepared_train` và `cypher_prepared_eval`
trong `$CYPHER_DATA_ROOT/llamafactory/dataset_info.json`.

`scripts/train.sh` tự động đọc `per_device_train_batch_size` sau khi gộp
các CLI override. Nếu batch size khác layout hiện có, script sẽ tạo và dùng
cache riêng tại `$CYPHER_DATA_ROOT/prepared/batch_<N>` và
`$CYPHER_DATA_ROOT/llamafactory/batch_<N>`. Ví dụ, lệnh sau tự chuẩn bị data cho
batch size 8 trước khi khởi chạy `torchrun`:

```bash
RUN_GPUS=0,1 bash scripts/train.sh configs/qwen3/fkl.yaml \
  per_device_train_batch_size=8
```

Cache có thể dùng lại khi chuyển qua lại giữa các batch size. Fingerprint
của sáu file grounding, bốn file prompt và preparation seed được ghi trong
layout; thay đổi bất kỳ input nào sẽ tự build lại cache. Dùng
`AUTO_PREPARE_FORCE=1` để ép build lại, `AUTO_PREPARE_DATA=0` để tắt cơ
chế này cho dataset tự quản lý. Có thể đổi nguồn grounding bằng
`CYPHER_GROUNDING_INPUT_DIR` và seed bằng `CYPHER_PREPARE_SEED`.

> **Lưu ý:**
> - Pipeline mặc định từ chối ghi đè lên thư mục đã có dữ liệu. Hãy thêm `--overwrite` khi chủ động muốn tạo lại từ đầu.
> - Tham số `--negative-ratio` chỉ tác động đến tập `selection_<split>.jsonl` (Task A) để cân bằng tỷ lệ nhãn `0` và `1`; tập `generation_<split>.jsonl` (Task B) luôn giữ nguyên gold sub-schema.

## Output format

`schemas.jsonl` lưu mỗi full schema duy nhất. Phần graph chuẩn hóa chỉ gồm
`nodes` và `relationships`:

```json
{
  "nodes": [
    {
      "label": "Person",
      "properties": {"name": "STRING"}
    }
  ],
  "relationships": [
    {
      "source": "Person",
      "type": "ACTED_IN",
      "target": "Movie",
      "properties": {}
    }
  ]
}
```

Output directory gồm:

- `selection_<split>.jsonl`: một dòng cho mỗi cặp `(question, schema unit)`,
  chứa `unit`, `unit_type`, và `label` nhị phân.
- `generation_<split>.jsonl`: `question`, `sub_schema` (chỉ gồm `nodes` và
  `relationships`) và gold `cypher`.
- `schemas.jsonl`: full canonical schema cho mỗi `schema_id`.
- `rejected_<split>.jsonl`: các mẫu không thể suy gold sub-schema chính xác,
  kèm diagnostics để audit.
- `normalization_issues_<split>.jsonl`: các schema không thể chuẩn hóa thành
  node/relation units. File giữ `schema_reference`, lỗi và, với Neo4j
  Text2Cypher, nguyên văn `raw_schema` để phân tích sau. Pipeline không tự
  suy đoán sửa những trường hợp này.
- `manifest.json`: cấu hình build và thống kê theo benchmark/split.

## Chất lượng chuẩn hóa

Canonical schema chuẩn hóa **cấu trúc** của ba nguồn về `nodes` và
`relationships`; nó không tự thêm property suy luận. Đặc biệt, property `name`
được gold Cypher dùng nhưng không xuất hiện trong metadata gốc của CypherBench
vẫn được giữ nguyên là thiếu. Điều này giữ corpus trung thực với benchmark gốc.

Adapter Neo4j hỗ trợ section chuẩn, relevant schema dạng prose, inline Cypher,
JSON inspection và Neo4j inspection repr. Những biến thể chưa hỗ trợ hoặc
không có topology sẽ nằm trong `normalization_issues_<split>.jsonl`, còn lỗi
coverage giữa gold Cypher và schema hợp lệ nằm trong `rejected_<split>.jsonl`.

Xem thêm về thiết kế và format tại [docs/data-preparation.md](docs/data-preparation.md).

## Distillation

Phần `src/distillation` tích hợp đầy đủ các baseline từ template: SFT,
FKL/RKL, SFKL/SRKL, CSD, AMID, HPD, BDL, FDD và adaptive DistiLLM.
Môi trường train yêu cầu Python 3.11+, Linux/WSL2, GPU NVIDIA hỗ trợ BF16 và
driver tương thích với PyTorch CUDA 12.8. Các lệnh train bên dưới chạy từ thư
mục gốc của repository.

### 1. Cài môi trường train

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-distillation.txt
```

DeepSpeed chỉ được bật trên Linux/WSL2. Có thể chạy SFT kiểm tra trên CPU hoặc
Windows bằng override `deepspeed=null bf16=false`, nhưng KD yêu cầu DeepSpeed
và BF16.

### 2. Tạo dữ liệu LlamaFactory

Dataset tải về đã chứa sẵn thư mục `$CYPHER_DATA_ROOT/prepared`. Để tạo lại
OpenAI chat JSONL khi cần:

```bash
python scripts/prepare_llamafactory_data.py --overwrite
```

Lệnh tạo bốn dataset `cypher_prepared_train`, `cypher_prepared_eval`,
`cypher_prepared_test_generator`, `cypher_prepared_test_selector` và đăng ký
chúng trong `$CYPHER_DATA_ROOT/llamafactory/dataset_info.json`. Các YAML hiện tại đã trỏ
sẵn tới train/eval local này.

### 3. Train LoRA teacher Qwen

Train teacher `Qwen/Qwen3-4B-Instruct-2507`. Adapter được lưu tại
`results/lora/qwen3/teacher_lora`:

```bash
RUN_GPUS=0,1 bash scripts/train.sh configs/distillation/teacher_lora_qwen3.yaml
```

Nếu chỉ dùng một GPU:

```bash
RUN_GPUS=0 bash scripts/train.sh configs/distillation/teacher_lora_qwen3.yaml
```

### 4. Train SFT student làm baseline

Bước này không bắt buộc trước KD, nhưng dùng để lấy baseline student không có
teacher:

```bash
RUN_GPUS=0,1 bash scripts/train.sh configs/distillation/student_sft.yaml
```

Output nằm tại `results/lora/qwen3/sft`, đúng với checkpoint path mà inference dùng
cho method `sft`.

### 5. Train student bằng KD

Config project mặc định và các preset FKL/RKL của cả ba họ model đều dùng
`kd_ratio: 0.6`, tự nạp base teacher 4B và LoRA adapter vừa train tại
`results/lora/qwen3/teacher_lora`:

```bash
RUN_GPUS=0,1 bash scripts/train.sh configs/distillation/kd.yaml
```

Loss train là `0.4 * LM loss + 0.6 * FKL loss`; output mặc định nằm tại
`results/lora/qwen3/fkl`.

Có thể thay adapter, dataset hoặc output bằng override:

```bash
RUN_GPUS=0,1 bash scripts/train.sh configs/distillation/kd.yaml \
  ref_model_adapters=/path/to/teacher-adapter \
  dataset=cypher_prepared_train \
  eval_dataset=cypher_prepared_eval \
  output_dir=results/lora/qwen3/my_fkl
```

### 6. Chạy các method khác

Các preset trong `configs/qwen3` đã dùng dataset local và tự nạp adapter tại
`results/lora/qwen3/teacher_lora`. Ví dụ AMID:

```bash
RUN_GPUS=0,1 bash scripts/train.sh configs/qwen3/amid.yaml
```

Tương tự có thể thay `amid.yaml` bằng `fkl.yaml`, `rkl.yaml`, `sfkl.yaml`,
`srkl.yaml`, `csd.yaml`, `hpd.yaml`, các config `fdd_*` hoặc
`distillm_adaptive_*`.

Train teacher trước, sau đó chạy tuần tự toàn bộ preset Qwen (SFT student và
tất cả KD baseline):

```bash
RUN_GPUS=0,1 bash scripts/train_all_qwen3_lora.sh
```

Mọi override phía sau script, trừ `output_dir` và `ref_model_adapters` do script
quản lý, được chuyển cho từng run. Ví dụ smoke-test một epoch:

```bash
RUN_GPUS=0,1 bash scripts/train_all_qwen3_lora.sh num_train_epochs=1
```

Để train lại vào một cây output mới mà vẫn giữ đúng đường dẫn riêng cho teacher
và từng method, đặt `RESULTS_ROOT`:

```bash
RESULTS_ROOT=results_run2 \
RUN_GPUS=0,1 bash scripts/train_all_qwen3_lora.sh
```

Output tương ứng là `results_run2/qwen3/teacher_lora`,
`results_run2/qwen3/sft`, `results_run2/qwen3/fkl`, ...; mọi KD config trong
run này tự dùng teacher tại root mới. Không truyền `output_dir=...` cho
`train_all_*`, vì script quản lý riêng output của từng method.

Teacher là dependency bắt buộc nên nếu bước teacher lỗi, script luôn dừng. Sau
đó, mặc định script dừng ngay khi một method lỗi; đặt `CONTINUE_ON_ERROR=1` để
chạy các method còn lại. Log riêng của từng method nằm trong
`<RESULTS_ROOT>/qwen3/run_all_logs/<timestamp>/`.

Preset trong `configs/llama3` dùng student Llama 3.2 1B và teacher Llama 3 8B.
Đăng nhập Hugging Face với tài khoản có quyền truy cập model gated, sau đó
train đúng LoRA teacher Llama:

```bash
export HF_TOKEN=your_huggingface_token
RUN_GPUS=0,1 bash scripts/train.sh configs/distillation/teacher_lora_llama3.yaml
```

Adapter được lưu tại `results/lora/llama3/teacher_lora` và mọi preset Llama tự nạp
đường dẫn này. Ví dụ:

```bash
RUN_GPUS=0,1 bash scripts/train.sh configs/llama3/amid.yaml
```

Train teacher trước, sau đó chạy tuần tự toàn bộ preset Llama 3:

```bash
RUN_GPUS=0,1 bash scripts/train_all_llama3_lora.sh
```

Có thể truyền override cho mọi run, đổi cả cây output qua `RESULTS_ROOT`, và
đổi riêng thư mục log qua `LLAMA3_LOG_DIR`:

```bash
LLAMA3_LOG_DIR=results/lora/llama3/custom_logs \
RUN_GPUS=0,1 bash scripts/train_all_llama3_lora.sh num_train_epochs=1
```

Ví dụ train vào root mới:

```bash
RESULTS_ROOT=results_run2 \
RUN_GPUS=0,1 bash scripts/train_all_llama3_lora.sh
```

Chạy inference cho model family Llama 3:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/infer_all_llama3_lora.sh \
  --checkpoint-root results/lora
```

Không dùng adapter Qwen cho config Llama hoặc ngược lại.

### Qwen2.5-Coder 3B/7B

Các preset trong `configs/qwen2.5_coder` dùng
`Qwen/Qwen2.5-Coder-3B-Instruct` làm student và
`Qwen/Qwen2.5-Coder-7B-Instruct` làm teacher. Template phải là `qwen`;
EOS generation được lấy từ generation config của model để hỗ trợ cả
`<|im_end|>` và `<|endoftext|>`.

Train teacher LoRA rồi chạy toàn bộ SFT/KD preset:

```bash
RUN_GPUS=0,1 bash scripts/train_all_qwen2_5_coder_lora.sh
```

Có thể đổi toàn bộ output sang root khác mà không sửa config:

```bash
RESULTS_ROOT=results_run2 \
RUN_GPUS=0,1 bash scripts/train_all_qwen2_5_coder_lora.sh
```

Chạy inference cho các checkpoint local dưới model family `qwen2.5_coder`:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/infer_all_qwen2_5_coder_lora.sh \
  --checkpoint-root results/lora
```

### 7. GPU, resume và kiểm tra

Launcher mặc định dùng hai GPU. `RUN_GPUS` vừa chọn GPU vừa đặt số process;
có thể dùng `NPROC_PER_NODE` khi scheduler đã quản lý `CUDA_VISIBLE_DEVICES`:

```bash
RUN_GPUS=2,3 bash scripts/train.sh configs/distillation/kd.yaml
NPROC_PER_NODE=4 bash scripts/train.sh configs/distillation/kd.yaml
```

Resume chính xác một run từ checkpoint (không dùng `train_all_*`):

```bash
RUN_GPUS=0,1 bash scripts/train.sh configs/qwen3/fkl.yaml \
  resume_from_checkpoint=results/lora/qwen3/fkl/checkpoint-1000
```

Phải giữ nguyên config của run cũ, đặc biệt là model/method, dataset, seed,
batch size, gradient accumulation và số process. `resume_from_checkpoint` phải
trỏ rõ tới `checkpoint-N` nằm trực tiếp trong đúng `output_dir`; project không
tự chọn checkpoint mới nhất.

Trước khi load model, resume preflight kiểm tra:

- `trainer_state.json` và `global_step` khớp tên checkpoint;
- model/adapter, optimizer và scheduler state đầy đủ;
- RNG state đủ cho từng rank và `world_size` khớp lần train trước;
- DeepSpeed `latest`, model state và optimizer shard đều trỏ đúng step;
- `resume_manifest.json` xác nhận config ảnh hưởng training, fingerprint của train/eval
  dataset, commit/config model, toàn bộ teacher adapter weights, phiên bản
  runtime và loại GPU không bị thay đổi;
- với adaptive DistiLLM, có đủ `distillm_state_rankN.pt` để khôi phục threshold,
  replay buffer, RNG của scheduler và rollout counters.

Nếu một thành phần bị thiếu, lệnh dừng ngay thay vì âm thầm chỉ load weights rồi
train tiếp với optimizer, scheduler hoặc DistiLLM state bị reset. Checkpoint cũ
tạo trước cơ chế manifest vẫn resume được sau khi qua kiểm tra cấu trúc, nhưng sẽ
cảnh báo vì không thể tự đối chiếu config cũ.

Fresh training cũng từ chối chạy nếu `output_dir` đã chứa `checkpoint-*`.
Đổi `output_dir` hoặc `RESULTS_ROOT` cho run mới; không dùng
`overwrite_output_dir=true` để trộn hai run vào cùng một cây kết quả.

Mọi preset đã pin `model_revision` của student và `ref_model_revision` của
teacher bằng commit SHA riêng. Checkpoint LoRA mới cũng ghi commit base model;
inference ưu tiên commit trong `resume_manifest.json` thay vì branch `main`.
`save_total_limit: null` không giới hạn số checkpoint được giữ lại. Vì vậy cần
theo dõi dung lượng của `output_dir` khi chạy nhiều epoch hoặc lưu thường xuyên.

Chạy test trước khi train dài:

```bash
python -m pytest
```

## Two-stage inference trên Linux

Pipeline inference thực hiện tuần tự hai task bằng cùng một LoRA adapter:

1. Chạy selector cho từng cặp `question + schema unit` để dự đoán JSON
   `{"label": "YES"}` hoặc `{"label": "NO"}` bằng greedy decoding
   (`do_sample=False`, tối đa 16 token).
2. Merge các unit `YES` thành predicted sub-schema, rồi đưa
   `question + predicted sub-schema` vào generator để sinh Cypher.

Prediction path không sử dụng gold selector label, gold sub-schema hoặc gold
Cypher. Các trường gold chỉ được đọc sau generation để tính metric.

### 1. Yêu cầu

- Linux x86-64.
- Python 3.11, khuyến nghị dùng virtual environment.
- GPU NVIDIA hỗ trợ BF16 và driver tương thích với CUDA 12.8.
- Đủ dung lượng local cho checkpoint trong `results` và Hugging Face cache cho base models.
- Ba test dataset đã tồn tại:

  ```text
  /mnt/local/aiskylimit_new_nothing/cypher-extract/datasets/cypher-extract-data/cypherbench_schema_grounding_full_final/
  /mnt/local/aiskylimit_new_nothing/cypher-extract/datasets/cypher-extract-data/mind_the_query_schema_grounding_full/
  /mnt/local/aiskylimit_new_nothing/cypher-extract/datasets/cypher-extract-data/neo4j_text2cypher_schema_grounding_full/
  ```

Mỗi directory phải có `selection_inference_test.jsonl` và
`generation_inference_test.jsonl`. Hai file này chứa mọi test example có question,
gold Cypher và full schema hợp lệ, kể cả example không trích được gold sub-schema.
Các file `selection_test.jsonl` và `generation_test.jsonl` vẫn là tập strict dùng
cho supervision/eval có gold sub-schema đáng tin cậy.

Từ repository root, tạo môi trường và cài dependency:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-distillation.txt
```

Đặt Hugging Face token nếu base model yêu cầu authentication:

```bash
export HF_READ_TOKEN="hf_..."
```

Có thể đặt cache trên ổ đĩa riêng:

```bash
export HF_HOME="/mnt/models/huggingface"
mkdir -p "$HF_HOME"
```

Kiểm tra CLI trước khi chạy:

```bash
python scripts/infer_two_stage.py --help
```

### 2. Model được chạy

`--methods all` chạy 13 model sau:

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

Với mỗi model, script đọc checkpoint local tại
`results/<setting>/<model-family>/<method>/checkpoint-N`. Tên
`<model-family>/<method>` bên dưới checkpoint root của setting
khớp trực tiếp với `output_dir` trong config training; inference không tải
checkpoint từ một Hugging Face result repository. Trong một method directory,
inference ưu tiên các checkpoint hoàn tất có `resume_manifest.json`, xác nhận tất
cả thuộc cùng một training fingerprint rồi mới chọn `N` lớn nhất. Checkpoint cao
hơn nhưng chưa ghi xong manifest sẽ bị bỏ qua; nếu các fingerprint bị trộn, script
dừng và yêu cầu dùng checkpoint root sạch. Với checkpoint cũ chưa có manifest,
script vẫn giữ cách chọn `N` lớn nhất để tương thích ngược.

`run_config.json` lưu fingerprint của checkpoint đã chọn. Fingerprint bao gồm
metadata và hash của config/tokenizer/trainer/resume state, cùng size, modification
time và hash mẫu của weight files. Nếu weights bị thay ngay tại cùng path và
cùng `checkpoint-N`, pipeline từ chối reuse prediction cũ và yêu cầu một
`--output-dir` mới.

Base model được lấy từ `adapter_config.json`:

- `teacher_lora`: `Qwen/Qwen3-4B-Instruct-2507`;
- 12 student methods: `Qwen/Qwen3-0.6B`.

### 3. Chạy toàn bộ trên một GPU

Chạy đủ 13 model trên cả ba benchmark bằng GPU 0:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 bash scripts/infer_all_qwen3_lora.sh \
  --selector-batch-size 128 \
  --generator-batch-size 16 \
  --dtype bfloat16 \
  --device cuda
```

Lệnh trên chạy model tuần tự để không giữ nhiều model trong VRAM. Trong mỗi
model, selector units và generator samples được infer theo batch. Nếu một batch
gây CUDA OOM, runner tự chia đôi batch cho đến khi chạy được hoặc chỉ còn một
sample, sau đó ghi nhớ batch size an toàn cho các batch có cùng token budget.

Để dùng checkpoint root local khác:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/infer_all_qwen3_lora.sh \
  --checkpoint-root /mnt/checkpoints/cypher-extract/results/lora \
  --selector-batch-size 128 \
  --generator-batch-size 16
```

### 4. Chạy một số method hoặc một benchmark

Ví dụ chạy teacher, SFT và FKL trên CypherBench:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_two_stage.py \
  --methods teacher_lora,sft,fkl \
  --datasets cypherbench \
  --selector-batch-size 64 \
  --generator-batch-size 8 \
  --dtype bfloat16 \
  --device cuda
```

Các giá trị hợp lệ của `--datasets`:

```text
cypherbench
mind_the_query
neo4j_text2cypher
```

Ví dụ chạy một method trên cả ba benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_two_stage.py \
  --methods hpd \
  --datasets cypherbench,mind_the_query,neo4j_text2cypher
```

### 5. Chạy song song trên nhiều GPU

Pipeline song song schema units bằng batching trên từng GPU. Để chạy nhiều
method đồng thời, khởi tạo một process cho mỗi GPU và đảm bảo danh sách method
không trùng nhau. Ví dụ với bốn GPU:

```bash
mkdir -p logs/inference

CUDA_VISIBLE_DEVICES=0 python scripts/infer_two_stage.py \
  --methods teacher_lora,sft,fkl \
  > logs/inference/gpu0.log 2>&1 &
pid0=$!

CUDA_VISIBLE_DEVICES=1 python scripts/infer_two_stage.py \
  --methods rkl,sfkl,srkl,csd \
  > logs/inference/gpu1.log 2>&1 &
pid1=$!

CUDA_VISIBLE_DEVICES=2 python scripts/infer_two_stage.py \
  --methods hpd,amid,fdd_sfkl \
  > logs/inference/gpu2.log 2>&1 &
pid2=$!

CUDA_VISIBLE_DEVICES=3 python scripts/infer_two_stage.py \
  --methods fdd_srkl,distillm_adaptive_sfkl,distillm_adaptive_srkl \
  > logs/inference/gpu3.log 2>&1 &
pid3=$!

wait "$pid0" "$pid1" "$pid2" "$pid3"
```

Theo dõi log:

```bash
tail -f logs/inference/gpu0.log
```

Mỗi process nhìn GPU được gán qua `CUDA_VISIBLE_DEVICES` như `cuda:0`, vì vậy
giữ `--device cuda` hoặc bỏ tham số này.

### 6. Resume

Selector và generator ghi kết quả vào file `.partial`, sau đó mới publish file
JSONL hoàn chỉnh. Nếu process bị dừng, chạy lại chính xác command cũ:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_two_stage.py \
  --methods teacher_lora,sft,fkl \
  --datasets cypherbench \
  --selector-batch-size 64 \
  --generator-batch-size 8
```

Pipeline sẽ:

- tiếp tục selector/generator từ row cuối của file `.partial`;
- khôi phục Python/NumPy/PyTorch/CUDA RNG sau batch cuối đã commit, nên phần
  generator sampling sau resume giống một lần chạy liên tục;
- rollback phần đuôi của batch đang ghi dở rồi sinh lại batch đó;
- reuse stage đã hoàn thành;
- tính lại metrics và manifest sau khi đủ output.

`run_config.json` khóa đường dẫn checkpoint local, input paths, SHA-256 của
input/prompt, ChatML template và inference options. Nếu checkpoint local được
chọn thay đổi, command dùng option khác, dataset được build lại hoặc prompt thay đổi, pipeline sẽ không
reuse output cũ. Khi đó chọn output directory mới:

```bash
python scripts/infer_two_stage.py \
  --methods sft \
  --output-dir results/inference/lora/qwen3-run-2
```

### 7. Output

Kết quả mặc định nằm tại:

```text
results/inference/lora/qwen3/seed<seed>/<method>/<dataset>/
├── run_config.json
├── selector_predictions.jsonl
├── predicted_subschemas.jsonl
├── generator_predictions.jsonl
├── metrics.json
└── manifest.json
```

- `selector_predictions.jsonl`: label và raw output của từng schema unit.
- `predicted_subschemas.jsonl`: sub-schema sau merge và các endpoint node được
  closure tự động.
- `generator_predictions.jsonl`: raw generator output, parsed Cypher, prompt
  length và reference dùng để đánh giá.
- `metrics.json`: selector accuracy/precision/recall/F1 trên các unit có gold label
  strict, Cypher exact match/ROUGE trên toàn bộ test example đi qua inference, và
  schema diagnostics. `selector.inference_count` gồm cả unit không có gold label;
  `selector.count` chỉ gồm unit thực sự có gold label. Example không có gold
  sub-schema đáng tin cậy nhận 0 ở selector qua hệ số `selector.coverage`; các
  metric chưa nhân coverage được giữ dưới tên `selector.strict_*`. Generator và
  execution metrics vẫn chấm predicted Cypher bình thường trên mọi example.
- `manifest.json`: checkpoint, options, thời gian và trạng thái từng stage.

Xem metric của một run:

```bash
jq . results/inference/lora/qwen3/seed10/sft/cypherbench/metrics.json
```

Xem một prediction:

```bash
head -n 1 results/inference/lora/qwen3/seed10/sft/cypherbench/generator_predictions.jsonl | jq .
```

### Chấm Cypher bằng Neo4j

Project có ba metric execution-based lấy từ pipeline CypherKD:

- `execution_accuracy`: so sánh kết quả thực thi của Cypher dự đoán và gold;
- `psjs`: Jaccard của provenance subgraph (các node được phần `MATCH` chạm tới);
- `executable`: kiểm tra Cypher dự đoán có chạy được hay không.

Cài Neo4j driver và chấm trực tiếp output của two-stage inference trên PowerShell.
Kết quả eval được tách khỏi inference và lưu dưới `results/evaluation/`:

```powershell
python -m pip install -e ".[evaluation]"

Copy-Item .env.example .env
# Sau đó sửa NEO4J_PASSWORD và các cấu hình Neo4j trong .env.
# Nên dùng tài khoản Neo4j read-only vì metric sẽ thực thi trực tiếp Cypher dự đoán.

evaluate-cypher `
  --input results/inference/lora/qwen3/seed10/sft/cypherbench/generator_predictions.jsonl `
  --name cypherbench-db `
  --graph nba
```

Logic của cả ba metric, preprocessing `<end_of_turn>` và cách aggregate được giữ
theo `CypherKD_ref`; xem bảng đối chiếu tại
[`docs/cypherkd-metric-parity.md`](docs/cypherkd-metric-parity.md).

Chạy toàn bộ seed, method, dataset và graph bằng script có sẵn. Script hoàn tất
một seed rồi mới chuyển sang seed tiếp theo, và tự merge sau mỗi dataset:

```powershell
.\scripts\evaluate_cypher_all.ps1
```

```bash
bash scripts/evaluate_cypher_all.sh
```

Mặc định script tự phát hiện method trong từng folder `seed<seed>`. Có thể chỉ
chấm một phần, ví dụ SFT/CypherBench:

```powershell
.\scripts\evaluate_cypher_all.ps1 -Methods sft -Datasets cypherbench
```

```bash
METHODS=sft DATASETS=cypherbench bash scripts/evaluate_cypher_all.sh
```

Eval toàn bộ 5 inference seed cho SFT/CypherBench graph `nba`:

```powershell
$seeds = 10, 42, 50, 100, 1234

foreach ($seed in $seeds) {
  evaluate-cypher `
    --input "results/inference/lora/qwen3/seed$seed/sft/cypherbench/generator_predictions.jsonl" `
    --name cypherbench-db `
    --graph nba
}
```

Kết quả được ghi tự động vào các folder tương ứng:

```text
results/evaluation/lora/qwen3/seed10/sft/cypherbench/nba/
results/evaluation/lora/qwen3/seed42/sft/cypherbench/nba/
results/evaluation/lora/qwen3/seed50/sft/cypherbench/nba/
results/evaluation/lora/qwen3/seed100/sft/cypherbench/nba/
results/evaluation/lora/qwen3/seed1234/sft/cypherbench/nba/
```

Eval và merge toàn bộ 7 graph CypherBench cho cả 5 seed:

```powershell
$seeds = 10, 42, 50, 100, 1234
$graphs = "company", "fictional_character", "flight_accident", "geography", "movie", "nba", "politics"

foreach ($seed in $seeds) {
  foreach ($graph in $graphs) {
    evaluate-cypher `
      --input "results/inference/lora/qwen3/seed$seed/sft/cypherbench/generator_predictions.jsonl" `
      --name cypherbench-db `
      --graph $graph
  }

  merge-cypher-evaluations `
    --input-dir "results/evaluation/lora/qwen3/seed$seed/sft/cypherbench"
}
```

Mỗi seed tạo thêm hai file gộp ở folder `cypherbench`:

```text
all_graphs_cypher_scores.jsonl
all_graphs_summary.json
```

`all_graphs_summary.json` chứa điểm `overall` trên toàn bộ sample và breakdown
riêng cho từng graph. Lệnh merge sẽ báo lỗi nếu thiếu graph, graph trong record
không khớp folder hoặc có ID trùng.

Có thể chọn metric với `--metrics execution_accuracy executable`. Khi chạy thủ công,
phải truyền cùng danh sách đó cho cả `evaluate-cypher` và `merge-cypher-evaluations`;
hai script tự động làm việc này. CLI tự load
`NEO4J_URI`, `NEO4J_USERNAME`, và `NEO4J_PASSWORD` từ `.env`. Database được chọn
bằng graph (`flight_accident` được đổi thành database `flight.accident`) đúng theo
CypherKD. `cypherbench-db` và `mind-the-query-db` là logical connector name, chọn
bằng `--name`. `neo4j_text2cypher_db` dùng endpoint demo và username/password bằng
tên graph, đúng cấu hình reference. Graph chọn bằng `--graph`, mặc định là `nba`. Có thể dùng
`--database` để override database thực tế. Với command trên, CLI ghi kết quả từng
mẫu vào `results/evaluation/lora/qwen3/seed10/sft/cypherbench/nba/cypher_scores.jsonl` và
trung bình toàn bộ metric vào file `cypher_scores_summary.json` trong cùng folder.
Đường dẫn output được suy ra tự động từ `--input` và `--graph`; vẫn có thể truyền
`--output` nếu muốn ghi sang vị trí khác.

### 8. Điều chỉnh VRAM

Nếu GPU ít VRAM, giảm batch size. Với `teacher_lora` 4B có thể bắt đầu bằng:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_two_stage.py \
  --methods teacher_lora \
  --selector-batch-size 32 \
  --generator-batch-size 4 \
  --dtype bfloat16
```

Nếu GPU không hỗ trợ BF16, thử `--dtype float16`. Có thể giữ LoRA adapter chưa
merge bằng `--no-merge-adapter`, nhưng inference thường chậm hơn.

Generation mặc định dùng sampling theo CypherKD (`do_sample=True`,
`temperature=0.5`, `top_p=0.95`, `top_k=0`, `num_beams=1`) và ChatML
`qwen3_nothink` đúng format LlamaFactory, không chèn thẻ `<think>`. Pipeline chạy theo thứ tự
`seed -> method -> dataset`: hoàn tất mọi method/dataset của một seed rồi mới chuyển sang seed tiếp theo.
Các seed mặc định là `10,42,50,100,1234`, lưu vào folder `seed10`, `seed42`, ... Selector dùng greedy
decoding với JSON `{"label": "YES|NO"}` và tối đa 16 new tokens; generator dùng tối đa 256 new tokens.
Có thể override:

```bash
python scripts/infer_two_stage.py \
  --methods sft \
  --seeds 10,42,50,100,1234 \
  --temperature 0.5 \
  --top-p 0.95 \
  --top-k 0 \
  --num-beams 1 \
  --generator-max-new-tokens 256
```

Tài liệu chi tiết về merge policy và output schema nằm tại
[`docs/two-stage-inference.md`](docs/two-stage-inference.md).

Các preset task-normalized loss được đặt trong `configs/*_normalized_loss` và
ghi checkpoint sang cây `results/lora_normalized`. Cách train teacher, student
và inference được mô tả tại
[`docs/normalized-loss.md`](docs/normalized-loss.md).

Các preset full fine-tune không normalized nằm trong
`configs/*_full_finetune` và ghi checkpoint sang `results/full_finetune`.
Hướng dẫn train teacher/student và inference nằm tại
[`docs/full-finetune.md`](docs/full-finetune.md).

Preset kết hợp full fine-tune với task-normalized loss nằm trong
`configs/*_full_finetune_normalized_loss`, ghi checkpoint sang
`results/full_finetune_normalized`; xem
[`docs/full-finetune-normalized-loss.md`](docs/full-finetune-normalized-loss.md).

Các script chạy toàn bộ tuân theo mẫu
`scripts/train_all_<family>_<setting>.sh`, với `family` là `qwen3`, `llama3`
hoặc `qwen2_5_coder`; `setting` là `lora`, `lora_normalized`,
`full_finetune` hoặc `full_finetune_normalized`.

Inference dùng convention tương ứng
`scripts/infer_all_<family>_<setting>.sh`. Mỗi setting tự chọn đúng checkpoint
root, output root và loại teacher (`teacher_lora` hoặc `teacher_full`).

### Training logs and per-epoch evaluation artifacts

Training prints the aggregate `loss` at `logging_steps` and also reports
`train_lm_loss` plus `train_distill_loss` for KD runs. FDD runs additionally
report `train_token_kd_loss` and `train_feature_loss`. The `train_all_*` launchers
mirror this console output into a separate log file for each method.

With `eval_strategy: epoch` and `predict_with_generate: true`, each validation
pass reuses its generated outputs to write:

- `<output_dir>/eval/epoch-<epoch>-step-<step>/predictions.jsonl`
- `<output_dir>/eval/epoch-<epoch>-step-<step>/metrics.json`

The explicit final evaluation is stored separately under
`<output_dir>/eval/final-step-<step>/`. Set `save_eval_predictions=false` on the
training command line to disable these artifacts. These are validation-set
generation artifacts; full two-stage inference over benchmark test sets remains
an explicit `infer_all_*` operation.
