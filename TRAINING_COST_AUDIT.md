# Kiểm tra phạm vi training cost cho rebuttal

Ngày kiểm tra: 2026-11-09. Nhánh: `rebuttal_iclr_ssft`, HEAD `ce7c934d2fc1ea02f1960130d9728e445f2696d2`, có thay đổi chưa commit. Báo cáo phản ánh **working tree hiện tại**, không chỉ HEAD. Không chạy GPU training, không đo runtime/FLOPs/VRAM, không sửa pipeline.

**Cập nhật theo phạm vi người dùng đã chốt:** đo method total từ s1K có teacher CoT sẵn; loại bước sinh teacher CoT cho tất cả phương pháp. Không yêu cầu tái lập cấu hình gốc từng paper. Phần so sánh cấu hình paper và ghi chú logging bên dưới chỉ là bối cảnh của lượt audit trước; chưa thuộc công việc triển khai hiện tại. Với phạm vi này, SSFT `commands.sh` hiện tính đủ IG/selection trước SFT; nó chỉ giống P-ALIGN ở khả năng dùng artifact có sẵn của upstream/entrypoint train riêng.

## Kết luận

Không thể dùng tổng thời gian của các launcher hiện tại làm phép so sánh end-to-end công bằng: P-ALIGN nhận dữ liệu đã xử lý; SGL xây mask; SSFT `commands.sh` tính lại IG. SSFT upstream cũng phát hành dữ liệu có lựa chọn segment sẵn và script train mặc định bỏ qua việc tính IG.

Phát hành dữ liệu xử lý sẵn không đồng nghĩa tác giả không công bố thuật toán/code. P-ALIGN có code tìm prefix và sinh continuation; SSFT có code attribution. Vấn đề ở đây là ranh giới phép đo và khả năng truy vết chi phí tạo artifact.

## 1. Đối chiếu từng phương pháp

| Phương pháp | Công đoạn ngoài SFT theo paper | Launcher hiện tại | Phần dễ bị bỏ khỏi phép đo |
|---|---|---|---|
| SFT Long-CoT | Có sẵn hoặc sinh teacher CoT | Chuyển dữ liệu s1K, train, export, eval | Nguồn gốc/sinh teacher CoT nếu tuyên bố tính từ câu hỏi thô |
| P-ALIGN | Student tự judge prefix qua binary search; sinh continuation; lọc đúng đáp án | Đọc JSON P-ALIGN đã xử lý, train, export, eval | Toàn bộ bước judge/search, generation và lọc trước SFT |
| SGL | Gradient capture, SVD và chọn step để tạo mask offline | `build_masks` trên 3 GPU, train, eval | Bỏ nếu chỉ đo `training`; launcher đầy đủ hiện không bỏ bước này |
| SSFT | IG, tổng hợp/chọn segment; nhánh self-generated CoT có sampling và chọn đáp án đúng | `commands.sh` gọi `one_pass.py`: tính IG, chọn segment, train, eval | Script train riêng/upstream có thể bắt đầu từ mask sẵn; không được ghi preprocessing là 0 cho phương pháp |

### P-ALIGN

Paper §3.2.1–3.2.2 yêu cầu tự đánh giá prefix bằng **student**, không mặc định gọi thêm một judge lớn độc lập; sau đó student sinh continuation. Eq. (9) giữ các continuation có đáp án đúng.

`palign/project_commands.sh:71–104` chỉ kiểm tra JSON đầu vào rồi train/export. File hiện có 966 bản ghi, mỗi bản ghi chỉ có `instruction`, `input`, `output`; không đủ log để khôi phục tổng số judge calls, token prefill/decode hoặc các continuation bị loại. Không thể suy ra rằng 34 mẫu chênh lệch so với 1,000 chính là số mẫu bị Eq. (9) loại.

Code tìm prefix và continuation vẫn có trong `palign/src/binary_search.py` và `palign/src/prefix-alignment.py`; không được gọi bởi launcher mặc định. Cả bản local và upstream của script continuation đã kiểm tra đều ghi output mà không thực hiện bộ lọc đúng đáp án Eq. (9). Nếu dựng lại pipeline, cần bổ sung/xác minh bước lọc và cách ghép prefix với continuation; không thể coi chạy hai script là tái tạo đầy đủ dataset đã phát hành.

Nguồn: [paper v1](https://arxiv.org/pdf/2601.10064v1), [upstream README](https://github.com/NEUIR/P-ALIGN/blob/4ddef39ff8812cedac939739029d2678d1978284/README.md), [upstream continuation](https://github.com/NEUIR/P-ALIGN/blob/4ddef39ff8812cedac939739029d2678d1978284/src/prefix-alignment.py).

### SGL / Spectral-guided Learning

Paper §3 và Figure 5 tách bước phân tích phổ offline khỏi masked SFT. §4.1 mô tả 10,000 câu hỏi AceReason, CoT từ DeepSeek-R1-0528; Appendix A dùng 6 epochs. Bản rebuttal dùng s1K và 3 epochs: đây là thí nghiệm chuẩn hóa lại điều kiện, không phải chi phí tái lập nguyên cấu hình paper.

`sgl/project_commands.sh:37–72` chạy tạo mask rồi mới train. `sgl/src/sgl/spectral.py:140–221` forward decoder với `no_grad`, rồi lấy gradient loss theo hidden state qua LM head; không phải một full-model backward cho từng token. `exact_spectral_selection` thực hiện reduced SVD chính xác trên ma trận FP32. Vì vậy phải cộng decoder forward, LM-head gradient, SVD và việc chuyển dữ liệu; không nên ước lượng là T lượt backward toàn model.

Trong bản triển khai local này, không thấy công đoạn model-based selection cốt lõi bị thay bằng mask công khai trong launcher mặc định. Kết luận này không phải khẳng định về độ đầy đủ của một official SGL repository chưa được xác minh.

Nguồn: [bản xuất bản PMLR](https://proceedings.mlr.press/v306/fan26e.html), [PDF của nhà xuất bản](https://raw.githubusercontent.com/mlresearch/v306/main/assets/fan26e/fan26e.pdf).

### Segment Selective SFT

**Có trường hợp giống P-ALIGN ở upstream.** Repository tác giả có `data/limo/solutions_top70cohe80_lennorm_7B_J50.jsonl`; `SelectiveSFT/run_train.sh` đọc trực tiếp file này. Đo riêng script đó bỏ chi phí tạo lựa chọn segment. Code tính IG vẫn được công bố riêng trong `Attribution/`.

Paper Appendix C.4 dùng J=50 và R1-Distill-Qwen-7B làm attribution model. Tác giả báo khoảng **7 GPU-hours cho IG mỗi nguồn CoT**, so với **8 GPU-hours cho một lượt SFT model 7B**. Đây là số tác giả báo trên thiết lập của họ, không phải số đo cho rebuttal/B200.

Ngoài IG, §4.1 và Appendix C.3 cho biết cấu hình Qwen2.5-7B-Instruct dùng self-generated CoT: 32 candidate/câu, chọn shortest correct; full-CoT SFT 7 epochs rồi selective SFT 4 epochs, full-parameter. Baseline full-CoT dùng cùng lịch hai giai đoạn. Sampling/chọn trace là công đoạn tạo dữ liệu cho nhánh đó; không áp đặt nó lên mọi nhánh dùng CoT có sẵn. Các phân tích confidence/truncation và tìm hyperparameter trong paper không mặc định phải chạy lại khi áp dụng hyperparameter đã cố định.

Trong working tree hiện tại, `commands.sh:77–90` gọi `one_pass.py`; `stage_prep:100–143` **tính lại IG**, dù record đã có `selected_spans_ids`. Bước attribution dùng `args.model`, tức backbone Qwen2.5-7B-Instruct hiện tại; không tự chuyển sang R1-Distill-Qwen-7B. Các entrypoint train riêng có thể nhận kết quả chọn segment từ trước. Cấu hình LoRA 3 epochs hiện tại không có lịch 7+4 của paper.

Nguồn: [paper arXiv §4.1, C.3, C.4](https://arxiv.org/html/2602.00425v1), [repository tác giả](https://github.com/SiyuanWangw/SegmentSelectiveSFT/tree/cb1d834c3870f29b15bbb798946bf3ba88591c2e), [script train upstream](https://github.com/SiyuanWangw/SegmentSelectiveSFT/blob/cb1d834c3870f29b15bbb798946bf3ba88591c2e/SelectiveSFT/run_train.sh).

## 2. Ranh giới đo đề xuất

**Phạm vi được chốt cho rebuttal là mục 2 dưới đây.** Mục 1 chỉ là breakdown có thể lấy thêm, không phải một mục tiêu benchmark riêng được yêu cầu:

1. **SFT-only:** bắt đầu khi dữ liệu/mask đã sẵn sàng; đo optimization theo cùng quy ước cho mọi phương pháp. Công bố rõ không bao gồm method-specific preprocessing.
2. **Method total từ một tập teacher CoT chung:** cộng toàn bộ preprocessing riêng của phương pháp và SFT. P-ALIGN gồm judge/search, continuation và lọc; SGL gồm capture/SVD; SSFT gồm IG/selection. Tính cả mẫu/candidate xử lý rồi bị loại.

Chi phí tạo teacher CoT chung có thể loại khỏi phạm vi 2 nếu mọi phương pháp thật sự được cấp cùng artifact đầu vào. Phải nói rõ đây là chi phí có điều kiện trên dữ liệu CoT có sẵn, không phải từ câu hỏi thô. Nếu phương pháp sinh thêm candidate/continuation riêng thì vẫn cộng phần đó.

Nếu muốn đo **tái lập cấu hình gốc từng paper**, dùng một bảng khác và tính cả sampling 32 candidate, warm-start hoặc giai đoạn train bắt buộc của đúng nhánh thí nghiệm. Không ghép chi phí full-parameter 7+4 của SSFT gốc với LoRA 3 epochs của các phương pháp đã chuẩn hóa rồi diễn giải là lợi thế thuật toán.

Với kết quả xử lý công khai nhưng thiếu log, ghi **preprocessing chưa đo / total chưa đầy đủ**, không ghi 0. Có thể báo SFT-only trước; chưa đủ căn cứ xếp hạng end-to-end.

Nếu tái sử dụng preprocessing qua K lần train, công bố thêm chi phí amortized `C_sft + C_prep/K`, đồng thời giữ chi phí một lần chạy `C_sft + C_prep`. Chỉ amortize artifact thật sự dùng lại được; mask phụ thuộc backbone/tokenizer/truncation không tự chuyển sang backbone khác.

## 3. Vấn đề trong cost logger hiện tại

Các ghi chú trong mục này được giữ từ lượt audit trước. Người dùng đã yêu cầu tạm gác logging cho đến khi chốt implementation details; chưa sửa hoặc triển khai các đề xuất này.

### FLOPs của masked SFT

`SegmentSelectiveSFT/one_pass.py:198–226` dùng `6 * n_params * retained_supervised_tokens * epochs`. Đây không phải FLOPs thực thi của pipeline này. Cả SSFT và SGL giữ toàn chuỗi làm context. Loss bằng 0 ở một token không xóa phép tính forward tại vị trí đó; các token được supervise phía sau còn phụ thuộc vào hidden states trước đó qua attention. Dense backward thông thường cũng không tự giảm tuyến tính theo tỷ lệ mask.

Token retention là chỉ số mật độ supervision, cần tách khỏi token processing/FLOPs. LoRA còn làm công thức full-training `6NT` không khớp trực tiếp; attention dài, padding, checkpoint recomputation và các kernel thực tế phải được xét. Không dùng số trainable parameters đơn thuần để thay thế toàn bộ compute qua backbone.

`prep_flops` cũng chỉ là heuristic dựa trên token counters: code IG chỉ chạy LM head ở vị trí đáp án và bật gradient checkpointing. Không nên coi công thức theo toàn bộ parameter count là phép đo đã kiểm chứng. Với SGL, cần hạch toán SVD riêng; nhiều profiler neural-network không đếm đầy đủ linear algebra kernels. Với generation P-ALIGN phải xét cả prefill và decode với context/KV cache, không chỉ output tokens.

### GPU-hours và thời gian

`one_pass.py:223` nhân cả prep và train với `trained.world_size`. Prep lại dùng `device_map="auto"`, không được đảm bảo có cùng số GPU thực sự dùng như DDP train. Công thức tổng quát là `sum_stage(sum_gpu(time_gpu_allocated_to_stage))/3600`; nếu mỗi stage dùng số GPU cố định, tương đương `sum_stage(n_gpu_stage * wall_sec_stage)/3600`.

Cần phân biệt GPU được giữ riêng cho job với GPU thực sự gắn vào stage, chọn một quy ước và báo nhất quán. Nếu cùng 3 GPU được cấp riêng suốt pipeline, nhân 3 có thể đúng cho reserved GPU-hours; không đủ để khẳng định active GPU-hours.

Timer train của SSFT chỉ bao quanh `trainer.train()` (`train_mask.py:420–427`); timer của P-ALIGN bao quanh `run_exp()` (`palign/src/train.py:29–35`), có thể gồm loading/preparation. Không so trực tiếp hai log này. Tách optimization time khỏi full-stage time; quy định rõ model loading, checkpoint I/O, export/merge. Eval và cài dependency/download nên để ngoài training cost và có mục riêng nếu cần.

### Peak memory

Prep SSFT lấy `max_memory_allocated()` trên current CUDA device; model có thể bị shard sang nhiều GPU. Train chỉ rank 0 ghi `training_stats.json`, không tổng hợp peak từ các rank. Vì vậy các giá trị hiện có chưa chứng minh peak của toàn job.

Đo theo stage và trên **mọi GPU/process**. Báo max per-device peak, tách PyTorch allocated/reserved và device-level memory nếu dùng công cụ hệ thống; không cộng peak prep với peak train. Nếu báo một số cho pipeline tuần tự, lấy max qua stage/GPU. Nếu báo tổng memory đồng thời, cần snapshot đồng thời, không cộng các peak ở những thời điểm khác nhau.

## 4. Các điều kiện phải ghi cùng bảng kết quả

- Cùng loại GPU, số GPU từng stage, precision, LoRA settings, số epochs/updates, batch/accumulation và sequence limit.
- Dữ liệu nguồn/revision, số mẫu trước/sau filtering, hash và số token thực tế sau template/truncation/padding; P-ALIGN hiện có 966 mẫu, SFT/SGL nhắm 1,000, SSFT còn có thể lọc trong chuẩn bị dữ liệu. Chưa xác minh file train trên server của SSFT.
- Framework/kernel, attention backend, packing và gradient checkpointing: Unsloth, Transformers và LLaMA-Factory có thể khác hiệu năng dù cùng thuật toán/batch.
- Cache/resume: chạy mới từ input đã quy định để đo preprocessing; không chỉ đo phần còn thiếu sau resume. Không suy ra chi phí tạo artifact từ thời gian đọc artifact.
- Nhãn trạng thái cho từng giá trị: measured / estimated / author-reported / unavailable. FLOPs là số phép tính, FLOP/s là tốc độ; không thay số FLOPs thực bằng peak TFLOP/s phần cứng nhân runtime.

Mẫu bảng nên có: `method | prep GPU-h | SFT GPU-h | total GPU-h | prep FLOPs | SFT FLOPs | prep peak GiB | SFT peak GiB | measurement status`.

## Nguồn và giới hạn xác minh

OpenReview trực tiếp yêu cầu xác minh/chặn truy cập. Đã đọc toàn văn P-ALIGN v1 từ arXiv, SSFT arXiv v1, và SGL từ kho PDF chính thức của PMLR. Không khẳng định các bản này giống từng byte với PDF OpenReview hiện tại.

Đã kiểm tra live repository tác giả P-ALIGN tại tree SHA `4ddef39ff8812cedac939739029d2678d1978284` và SSFT tại tree SHA `cb1d834c3870f29b15bbb798946bf3ba88591c2e`. Không xác minh official SGL code; nhận xét về SGL code chỉ áp dụng thư mục `sgl/` đang được launcher rebuttal dùng.

Không có số đo GPU mới, không có kết luận phương pháp nào rẻ nhất. Đây là audit công đoạn và cách hạch toán trước khi benchmark.

## Kiểm tra bổ sung: khả năng chạy và ước lượng preprocessing

Đã thực thi bảy kiểm tra trên source hiện tại bằng Python chuẩn, dùng phản hồi model/attribution giả lập, không import hoặc chạy model thật. Bằng chứng nằm ở `/private/tmp/training-cost-followup-checks.json`.

- Binary search tìm đúng minimum trong tám trường hợp ranh giới khác nhau khi judge có tính đơn điệu.
- Đưa trực tiếp record schema s1K có `deepseek_thinking_trajectory` vào `process_jsonl` tạo output rỗng vì hàm chỉ đọc `Long-CoT`. Cần adapter trường dữ liệu.
- Lỗi inference judge giả lập bị chuyển thành NOT_ENOUGH; cuối cùng ghi full-CoT với `is_sufficient=False`, không làm run thất bại.
- Script continuation vẫn xử lý record có `is_sufficient=False`, giữ cả câu trả lời sai giả lập, làm mất gold answer/status, và chưa xuất schema SFT. Cần lọc/ghép prefix-continuation/export trước khi coi đây là pipeline hoàn chỉnh.
- Sentence splitter tái tạo văn bản có thể thêm dấu chấm; nó không bảo toàn nguyên văn.
- Hàm `stage_prep` thực tế của SSFT gọi attribution J=50 cho mọi record và ghi đè selection cũ trong kiểm tra giả lập. Vì vậy đường chạy one-pass hiện không có khoản IG bị bỏ qua.
- Năm file source liên quan parse thành công. Điều này không chứng minh GPU execution thành công; local là macOS arm64, không có torch/transformers/vLLM hay model weights trong các thư mục đã kiểm tra.

Có thể ước lượng preprocessing sau khi nối/sửa các bước tối thiểu trên: lấy mẫu ngẫu nhiên theo nhóm độ dài token của s1K (ví dụ bắt đầu 50–100 mẫu, tăng nếu sai số còn lớn), chạy đúng backbone/hardware/context/batching dự kiến, đo riêng setup và xử lý. Tổng chi phí biến đổi có thể ước lượng theo `sum_h(N_h * mean_cost_h)`; GPU-hours phải phản ánh số GPU của mỗi stage. Với vLLM batching, cần đo các batch có phân bố độ dài đại diện và hiệu chỉnh throughput; không bảo đảm nhân thời gian một request đơn lẻ lên N là đúng.

P-ALIGN cần chạy cả judge/search và generation/filter trên mỗi mẫu pilot, tính cả các mẫu bị loại; bỏ artificial sleep bằng `--sleep_sec 0` nếu không phải một phần thiết kế được giữ. SSFT cần chạy đủ J=50 trên mỗi mẫu pilot; không lấy J nhỏ rồi nhân tuyến tính làm số chính. Kiểm tra trạng thái OOM/long của SSFT: code có đường gán điểm 0 thay cho IG, nên một run kết thúc không tự chứng minh mọi mẫu đã được attribution thành công.

FLOPs từ pilot cần đếm/ước lượng công việc thực và công bố sai số. Peak memory không ngoại suy tuyến tính theo số mẫu: đo trực tiếp với các độ dài cực trị và batching thực tế, chỉ gọi là peak đo trên pilot nếu chưa kiểm tra workload đầy đủ. Nếu SFT vẫn dùng 966 bản ghi P-ALIGN công khai, overhead ước lượng trên s1K chỉ là ước lượng của pipeline đã chọn, không phải chi phí lịch sử được xác minh để tạo đúng 966 bản ghi đó.
