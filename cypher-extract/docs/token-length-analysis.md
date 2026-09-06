# Token length analysis

`scripts/analyze_token_lengths.py` đo độ dài token thực tế mà `cutoff_len` và
context lúc inference phải phủ. Prompt được render bằng đúng
`PromptTemplates` và renderer (`render_qwen3_nothink` / `render_llama3`) mà
training và pipeline dùng, nên số đo giữ prompt parity by construction.

Chạy lại:

```bash
python scripts/analyze_token_lengths.py --model-families all --local-files-only
```

Report ghi ra `data/token_length_analysis.json` (không được track vì `data/`
nằm trong `.gitignore`) — file này là bản chép lại kết quả để giữ lâu dài.

## Training (`$CYPHER_DATA_ROOT/llamafactory/cypher_prepared_*.jsonl`)

`cutoff_len` truncate đúng `prompt + response`, nên nó phải `>= max_total`.

| family | split | rows | max_prompt | max_total | max_response |
| --- | --- | --- | --- | --- | --- |
| qwen3 | train | 13654 | 520 | 604 | 113 |
| qwen3 | eval | 3414 | 515 | 608 | 109 |
| qwen2.5_coder | train | 13654 | 520 | 604 | 113 |
| qwen2.5_coder | eval | 3414 | 515 | 608 | 109 |
| llama3 | train | 13654 | 521 | 605 | 110 |
| llama3 | eval | 3414 | 516 | 609 | 108 |

`cutoff_len` hiện tại là `892` (qwen3, qwen2.5_coder) và `899` (llama3), tức là
dư khoảng 46% so với `max_total`. **Không có row nào bị truncate.**

## Inference (`*_inference_test.jsonl`)

Generator được prompt bằng *predicted* sub-schema, nên worst case là selector
đánh dấu toàn bộ schema unit của example là related — không phải gold
sub-schema. Cột `gold sub-schema` chỉ để tham chiếu.

Số dưới đây của qwen3 / qwen2.5_coder (hai family chung tokenizer); llama3
lệch trong khoảng ±4 token.

| dataset | selector prompt | generator worst case | generator gold sub-schema |
| --- | --- | --- | --- |
| cypherbench | 281 | 875 | 498 |
| mind_the_query | 298 | 1859 | 753 |
| neo4j_text2cypher | 371 | 2129 | 1270 |

Cộng `max_new_tokens` (selector 16, generator 256):

| family | required `cutoff_len` | required inference context |
| --- | --- | --- |
| qwen3 | 608 | 2385 |
| qwen2.5_coder | 608 | 2385 |
| llama3 | 609 | 2387 |

## Hệ quả

- `cutoff_len` hiện tại an toàn; không cần nâng.
- Context inference cần khoảng `2.4k` token, gấp gần `2.7x` `cutoff_len`.
  `cutoff_len` không áp cho generation nên đường HF `generate` hiện tại không
  bị cắt, nhưng bất kỳ chỗ nào set `max_model_len` (ví dụ khi chuyển sang vLLM)
  phải dùng `2387`, không phải `cutoff_len`.
- Response dài nhất trong training data là `113` token, nên
  `generator_max_new_tokens = 256` còn dư hơn `2x` headroom.
- Prompt generator lúc train dài nhất `~521` token, còn lúc infer có thể lên
  `2131`. Đây là distribution shift thật ở generator, không phải rủi ro
  truncation.
