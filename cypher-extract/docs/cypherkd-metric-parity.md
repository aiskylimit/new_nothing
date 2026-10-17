# CypherKD metric parity

Phần evaluation của project cố ý giữ nguyên hành vi của snapshot trong
`CypherKD_ref`. Các thay đổi về cấu trúc chỉ gồm package import, type adapter cho
JSONL của project và lazy import Neo4j để unit test có thể chạy khi chưa cài
evaluation extra; thuật toán metric không được “cải tiến” riêng.

## Nguồn chuẩn

| Thành phần | File reference | SHA-256 |
| --- | --- | --- |
| Execution accuracy | `CypherKD_ref/src/metrics/execution_accuracy.py` | `166AAC6B499E6CDE137BFE2EC8E4E817011DF3CF6BFBCFF26AE5EC3188D3858E` |
| Executable | `CypherKD_ref/src/metrics/executable.py` | `9B75B6CCFC48232328203A125211CBBD48D6FDB26D02AAD150AF0DA23659E561` |
| PSJS | `CypherKD_ref/src/metrics/provenance_subgraph_jaccard_similarity.py` | `2DD58FCA8431D4236614DDDB897EFF28A8ED85ED8C1C47E975EE184ABD6188D2` |
| Scoring | `CypherKD_ref/src/evaluator/scoring.py` | `21CB218480859B8F00AF2C8FED65445B420E4C7347FFFFE668821F1CC55BEE55` |

## Hành vi được khóa

- Exact-match shortcut so sánh nguyên chuỗi, không tự trim whitespace.
- Prediction chỉ bỏ suffix `<end_of_turn>`; không tự bóc markdown hoặc sửa JSON
  malformed.
- Execution accuracy giữ nguyên kiểm tra `ORDER BY`, multiset, hoán vị cột và 20
  lần lấy mẫu ngẫu nhiên cho output rộng hơn ba cột.
- PSJS giữ nguyên parser regex case-sensitive và cách thêm biến `ntmp`/`rtmp`.
- Metric exception do lớp `safe_compute` bắt được được lưu thành object `error`;
  khi aggregate object này được tính là `0.0`.
- Aggregate làm tròn bốn chữ số thập phân và trả về object `overall` giống
  reference.
- CypherBench và Mind the Query dùng database name đổi `_` thành `.`. Neo4j
  Text2Cypher dùng `bolt+s://demo.neo4jlabs.com:7687`, database giữ nguyên tên
  graph, username và password mặc định cũng bằng tên graph.

Các kiểm thử parity nằm trong `tests/test_cypher_evaluation.py`. Nếu muốn thay đổi
bất kỳ hành vi nào ở trên, cần coi đó là một metric mới thay vì âm thầm sửa metric
CypherKD hiện tại.
