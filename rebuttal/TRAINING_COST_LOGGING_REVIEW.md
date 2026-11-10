# Review logging training cost — rebuttal_iclr_ssft

**Cập nhật sau sửa:** F1–F3 đã được xử lý: counter riêng cho IG; cost event được lưu trước completion marker và kiểm tra/phục hồi khi resume; kiểm tra đủ worker/rank và stage bắt buộc trước khi báo complete. Shard rỗng đã finalize vẫn hợp lệ. F4–F6 không nằm trong lần sửa này. Các finding bên dưới giữ lại để mô tả trạng thái trước sửa.

Kiểm chứng sau sửa: **150 tests passed, 4 subtests passed**, gồm test hồi quy IG không thay đổi attribution, resume ở hai phía của marker/event, thiếu/nhân đôi rank, shard rỗng, thiếu native CoReD stages và các test CoReD CPU/Gloo 2/3 ranks. Chưa chạy CUDA/Unsloth/vLLM trên GPU thật.

Ngày audit ban đầu: 2026-11-09. HEAD: `d71851d297a91ef7e88dffa9e3333761ac4d766c`. Các finding dưới đây phản ánh code trước lần sửa nêu trên; không dùng kết luận từ phiên bản pipeline cũ. Không chạy training/inference trên GPU.

## Kết luận

**Chưa đầy đủ để dùng các trường tổng hợp làm bảng method-total cuối cùng.** Bộ điều phối mới đã bao phủ các stage chính của năm phương pháp, nhưng còn lỗi thiếu FLOPs SSFT, không kiểm tra đủ worker/stage, có thể mất GPU-hours khi resume sau ngắt đột ngột, và peak memory chưa có coverage/scope đủ rõ. FLOPs có nhãn estimated đúng, nhưng công thức padding vẫn có thể gây sai lệch giữa phương pháp.

Entrypoint được review: `rebuttal/project_commands.sh` → `rebuttal.runner`; profile H100 gọi cùng `Job` và bộ tổng hợp. Những launcher riêng trong từng thư mục vẫn tồn tại nhưng không đồng nghĩa đã dùng logger chung.

**Lưu ý phạm vi P-ALIGN:** code mới chạy preprocessing đầy đủ từ canonical s1K rồi SFT trên dữ liệu vừa lọc (`runner.py:228–245`), không còn dùng phương án pilot preprocessing + SFT trên 966 mẫu public. Logger hiện không có cơ chế ngoại suy pilot. Đây là khác biệt so với quyết định trước trong chat, không phải một lỗi hạch toán của riêng logger.

## Bao phủ công đoạn hiện tại

| Phương pháp | Stage được bao quanh bằng runner | Nhận xét |
|---|---|---|
| Long-CoT SFT | optimization | Có timing/GPU-hours, counter hook và per-process telemetry |
| P-ALIGN | prefix_alignment → optimization | Gồm judge/search, continuation và lọc; generation có counter riêng |
| SGL | spectral_masks → optimization | Có counter riêng cho decoder/head-gradient và estimate SVD |
| SSFT | attribution → optimization | Có timing attribution, nhưng common FLOPs/token counters bỏ lỡ IG |
| CoReD | prepare → prefix → key → phase1 → weights → merge → distill | Có native timing/telemetry và counter merge; kiểm tra coverage còn thiếu |

Teacher-CoT generation không được gọi trong các training stage này. Evaluation được tách ra bằng tên `evaluation_*` và bị loại trong `cost.aggregate`. Allocated GPU-hours và worker/stage GPU-hours được tách riêng. Bộ tổng hợp lấy max memory thay vì cộng các peak. Đây là các điểm đúng của thiết kế hiện tại.

## Findings

### F1 — P1: SSFT IG không vào common FLOPs/token counters

**Vị trí:** `rebuttal/instrumented.py:18–20`, `rebuttal/telemetry.py:37–76`, `SegmentSelectiveSFT/Attribution/grad_analyze.py:160–193`, `rebuttal/cost.py:33–55`.

Instrumentation gắn forward hook vào outer `AutoModelForCausalLM`. IG lại gọi trực tiếp `self.model.model(...)` và `self.model.lm_head(...)`; outer forward không được gọi. Hai native token counters vẫn tăng, nhưng `telemetry.STATE` không nhận hoạt động đó. `method_prep` lưu native `.cost.json`, trong khi aggregator chỉ đọc `telemetry/.../process-*.json`, không dùng file native để bù khoản thiếu.

**Tái hiện:** chạy chính hàm IG hiện tại với model Qwen2 nhỏ khởi tạo local trên CPU, 6 input tokens × 2 interpolation steps. Chỉ thay `.cuda()` bằng phép giữ tensor trên CPU. Native forward/backward counters đều là 12; attribution hữu hạn. Common forward tokens và FLOPs vẫn bằng 0. Đây là phép tính model thật trên CPU, không phải suy luận từ tên hàm; không phải benchmark model 7B/CUDA.

**Ảnh hưởng:** stage attribution không có estimated FLOPs hợp lệ; tổng SSFT bị thiếu khoản IG. Aggregator thường đánh `flops_complete=false`, là cách báo thiếu đúng, nhưng vì vậy chưa thể lấy total FLOPs SSFT làm số cuối cùng. `token_counter_coverage_complete` vẫn có thể true vì chỉ kiểm tra các stage optimization/phase1/distill.

**Cần sửa:** instrument đúng decoder và LM head/đường autograd của IG, hoặc cung cấp counter stage riêng với cùng định nghĩa FLOPs; không đưa heuristic native vào total mà không chuẩn hóa và ghi nguồn. Kiểm tra coverage cả attribution.

### F2 — P1: Completion marker có thể tồn tại trước khi cost event được lưu

**Vị trí:** `rebuttal/runner.py:91–94`, `147–175`; job-level accounting ở `383–386` cũng chỉ được ghi trong finally.

Runner ghi `<stage>.complete.json` trước khi vào phần ghi `stage_events.jsonl`. Nếu process bị SIGKILL/máy mất trong khoảng đó, finally không chạy, marker vẫn tồn tại nhưng event chưa có. Lần chạy lại chỉ thấy marker rồi return, không phục hồi cost event hoặc đánh chi phí chưa biết.

**Tái hiện:** dùng `Job.execute` thật với child CPU không dùng GPU; mô phỏng process ngắt ngay sau khi marker được ghi. Lần gọi lại in `reuse optimization`; stage event vẫn không tồn tại. Aggregator trả `completed_stage_gpu_hours=0` và `flops_complete=true`. GPU count trong fixture là dữ liệu mô phỏng, không phải số đo GPU.

**Ảnh hưởng:** có thể đánh mất chi phí của công việc đã hoàn tất. Job event cũng không bảo đảm tồn tại khi process chính bị ngắt cứng. Không thể tuyên bố bao gồm mọi failed/preempted attempt chỉ nhờ finally.

**Cần sửa:** ghi event/accounting bền vững trước marker; marker tham chiếu attempt/event đã được xác nhận. Khi resume phải kiểm tra hai phía và từ chối báo total complete nếu event mất. Nếu cần tính cả hard-killed attempt, dùng journal start/heartbeat và chính sách phục hồi/đánh unknown rõ ràng.

### F3 — P1: Có một counter không đồng nghĩa đã đủ worker và đủ stage

**Vị trí:** `rebuttal/cost.py:18–42`, `53–69`.

Aggregator chỉ glob các file tìm được, rồi cộng chúng. Không đối chiếu với số worker/rank thực sự phải phát telemetry và không kiểm tra đủ stage bắt buộc của phương pháp. Cờ token coverage chỉ kiểm tra optimization/phase1/distill, bỏ qua preprocessing.

**Tái hiện 1:** stage khai báo 3 GPU, chỉ có telemetry của một worker: aggregator vẫn báo `flops_complete=true`, `token_counter_coverage_complete=true`, và trả số tổng chỉ của worker đó.

**Tái hiện 2:** outer `cored_pipeline` complete nhưng thiếu native `timings.jsonl`: mở rộng thành 0 stage, trả `completed_stage_gpu_hours=0`, `estimated_flops=0`, `flops_complete=true`. Không có lỗi thiếu native accounting.

**Tái hiện 3:** thiếu toàn bộ counter attribution nhưng optimization có counter: FLOPs được đánh incomplete, nhưng token coverage vẫn true và `training_tokens` trông như tổng toàn pipeline dù không có IG.

**Cần sửa:** lưu manifest expected workers/stages, PID/rank/device/role và final flush/exit coverage. Kiểm tra đủ native CoReD stages. Tách cờ coverage cho optimization-only và method-total; không biến absence thành complete/0.

### F4 — P2: Peak memory chưa đủ coverage và không có scope nhất quán

**Vị trí:** `rebuttal/telemetry.py:16–34`, `rebuttal/cost.py:21–51`, `72–75`, `rebuttal/runner.py:112–124`.

- `flush` chỉ lấy CUDA device theo `LOCAL_RANK` của chính process gọi hàm. P-ALIGN/CoReD generation dùng vLLM và counter được cập nhật ở process gọi API; chưa có cơ chế thu Torch peak trực tiếp từ mọi process GPU của engine. Có `nvidia-smi` riêng để quan sát device-used memory, nhưng metric đó khác Torch allocated/reserved. Chưa chạy vLLM/GPU thật để khẳng định coverage cụ thể của phiên bản server.
- Root Torch peak chỉ lấy completed stage; không lấy peak từ failed attempt. Fixture failed peak 9,000 và success peak 1,000 trả peak 1,000, trong khi allocated GPU-hours bao gồm các attempt có job event. Nếu muốn scope successful-only phải ghi rõ, không diễn giải là peak của mọi invocation.
- Device sampling dùng điều kiện tất cả stage phải có file; thiếu một file thì bỏ toàn bộ samples của các stage còn lại. Fixture có stage đo được 9,000 MiB nhưng stage sau thiếu file trả peak null thay vì giữ observed peak + incomplete coverage.
- Sample mỗi giây là observed sampled peak, không bảo đảm bắt đúng đỉnh tức thời. Không có cờ memory completeness hoặc sample coverage để thể hiện các hạn chế trên.

**Cần sửa:** xác định peak successful run hay tất cả attempt; thu đủ process/device hoặc dùng sampled device peak với đúng tên. Giữ các sample hợp lệ khi thiếu một phần và ghi coverage. Báo peak bằng max, không cộng giữa stage.

### F5 — P2: FLOPs lấy chiều dài không padding cho cả các phép tính dense vẫn chạy trên padding

**Vị trí:** `rebuttal/telemetry.py:49–70`.

Counter dùng `attention_mask.sum()` làm chiều dài cho toàn bộ phần `2*N*tokens` và attention. Token không padding là chỉ số dữ liệu hợp lệ, nhưng không phải mọi phép chiếu/MLP/head đều được kernel bỏ qua ở vị trí padding. Điều này có thể ảnh hưởng khác nhau giữa batch dài ngắn của baseline và microbatch/packing của CoReD.

**Tái hiện:** hai forward Qwen2 eager có cùng input shape và q_proj shape `[2,3,16]`. Chỉ đổi attention mask, estimate thay từ 42,048 xuống 27,904 FLOPs dù q_proj vẫn xử lý cùng số vị trí tensor. Đây là bằng chứng công thức đang tính theo giả định không padding, không phải phép đo toàn bộ thực thi dense.

**Cần sửa:** tách unpadded token exposures khỏi executed padded tensor positions; estimator cần theo backend/packing/unpadding thực tế. Nếu chủ đích chỉ báo normalized unpadded FLOPs thì đổi định nghĩa và không diễn giải là FLOPs thực thi. Việc FLOPs có nhãn estimated là đúng, nhưng nhãn đó chưa tự xử lý sai lệch padding.

### F6 — P2 cho các entrypoint cũ: Còn nhiều nguồn cost không tương đương

**Vị trí:** `SegmentSelectiveSFT/one_pass.py:200–228`; các `project_commands.sh` ở từng method.

SSFT standalone vẫn dùng `6 * n_params * retained_supervised_tokens * epochs`, nhân prep time với training world size, và peak từ native stats. Công thức theo retained labels không đại diện cho full-context masked computation. P-ALIGN/SFT/SGL launcher riêng cũng không tự đi qua `rebuttal.instrumented`/`Job.execute`.

Đây không phải đường tạo `training_cost.json` của coordinator mới, nên không được nhầm thành F1 đã được native cost bù. Với bảng rebuttal hiện tại chỉ nên dùng logger coordinator sau khi sửa các finding trên; đánh dấu standalone cost là legacy hoặc chặn việc trộn kết quả.

## Các phần cần phân biệt nhưng không phải lỗi đã chứng minh

- Dense FLOPs là estimate, không phải hardware counters; optimizer, một số bookkeeping, adapter/custom-loss work chưa được đếm đầy đủ. Tài liệu đã nêu một phần giới hạn. Cần giữ nhãn và cùng quy ước khi so sánh.
- Main job ghi số GPU cấp phát là 3, đúng profile chính. Formal H100 có wrapper riêng ghi 2; test hiện tại xác nhận expansion phase1/merge dùng 2/1 GPU, không phát hiện nhầm 3 ở profile đó.
- Failed stage không được đưa vào estimated FLOPs tổng; aggregator có đặt `flops_complete=false` khi có failed GPU stage. Đây là báo giới hạn đúng, không nên tự suy ra mọi failed FLOPs đã được phục hồi.
- `Active stage GPU-hours` trong bảng dùng completed stages, trong khi allocated hours gồm các job attempt đã ghi được. Cần ghi scope này rõ khi có retry; hai cột không cùng định nghĩa.
- Chưa có GPU validation cho CUDA/Unsloth/vLLM, exact per-rank memory, GPU throughput hay FLOP counters. Không có số benchmark mới từ review này.

## Kiểm chứng đã chạy

1. `tests/test_rebuttal_plan.py`, `tests/test_ssft_shards.py`, `tests/test_formal_b_2h100.py`: **35 passed**.
2. Bảy kiểm tra bổ sung trong `/private/tmp/audit_training_cost_logging.py`: tái hiện SSFT IG mất common counters; thiếu ranks; mất sampled peak khi thiếu file; failed Torch peak bị bỏ; thiếu prep token counters; thiếu CoReD timings vẫn complete/0; padding estimator. Tất cả assertion xác nhận hành vi đã mô tả.
3. `/private/tmp/audit_cost_completion_gap.py`: tái hiện completion-marker gap và resume bỏ qua stage không có cost event.

Evidence JSON: `/private/tmp/training-cost-logging-audit-evidence.json`, `/private/tmp/training-cost-completion-gap-evidence.json`. Các script/evidence chỉ phục vụ audit, không sửa production code và không tạo GPU result.

Ưu tiên trước full run: F1 (SSFT counters), F2 (durable accounting), F3 (coverage validation), sau đó F4 (memory coverage) và F5 (FLOPs padding convention). F6 cần xử lý nếu còn sử dụng launcher cũ.
