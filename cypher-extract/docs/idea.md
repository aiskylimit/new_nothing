# Distillation cho Schema Grounding và Text-to-Cypher

## 1. Bối cảnh

Text-to-Cypher có thể xem là một trường hợp của bài toán **Text-to-Query-Language (Text-to-QL)**. Luồng chuẩn trực tiếp là:

```text
Question + Full Schema -> Cypher
```

Tuy nhiên, full schema thường chứa nhiều node, relationship, property không liên quan. Điều này làm tăng context length, gây nhiễu khi grounding, và mở rộng không gian tìm kiếm của model. Một hướng phổ biến là tách bài toán thành hai bước:

```text
Question + Full Schema -> Relevant Sub-schema
Question + Relevant Sub-schema -> Cypher
```

Mục tiêu của ý tưởng này là distill không chỉ năng lực sinh Cypher mà còn năng lực **chọn sub-schema phù hợp với câu hỏi**.

## 2. Dữ liệu huấn luyện

Với mỗi mẫu gồm câu hỏi `Q`, full schema `S`, và gold Cypher `Y`:

1. Parse `Y` thành AST/graph pattern và suy ngược ra sub-schema được dùng `S*`.
2. Tách full schema `S` thành các schema unit:
   - **Node unit**: node label cùng các property liên quan, ví dụ `(:Person {name, age})`.
   - **Relation unit**: source node type, relationship type, target node type, direction, và property nếu có, ví dụ `(:Person)-[:ACTED_IN]->(:Movie)`.
   - Có thể mở rộng thành **property unit** hoặc **path unit** cho các câu hỏi có filter, aggregation, `ORDER BY`, hoặc multi-hop path.
3. Với mỗi unit `u_i`, tạo nhãn:

   ```text
   z_i = yes  nếu u_i thuộc S*
   z_i = no   nếu u_i không thuộc S*
   ```

Từ đó tạo hai loại dữ liệu:

```text
Task A — Schema selection:
Question + Schema unit -> yes/no

Task B — Query generation:
Question + Gold sub-schema -> Gold Cypher
```

Nhãn selection được suy ra tự động từ gold Cypher. Đây là supervision có chất lượng cao nhưng về bản chất vẫn là **silver/weak supervision**: Cypher vàng phản ánh một cách viết query cụ thể, chưa chắc đã bao gồm mọi unit ngữ nghĩa có thể liên quan.

## 3. Baseline

### 3.1. Direct generation

```text
Question + Full Schema -> Cypher
```

Fine-tune hoặc áp dụng các phương pháp distillation Text-to-Cypher hiện có, bao gồm CypherKD.

### 3.2. Generative schema selection

Trộn hai task sau vào dữ liệu SFT/distillation:

```text
Question + Schema unit -> yes/no
Question + Gold sub-schema -> Cypher
```

Ở inference, model lần lượt dự đoán `yes/no` cho từng unit, ghép các unit được chọn thành predicted sub-schema, rồi dùng sub-schema đó để sinh Cypher.

## 4. Phương pháp đề xuất: Hidden-state Schema Selector

Thay vì để LLM sinh token `yes/no`, dùng biểu diễn ẩn của LLM làm input cho một classification head.

Với mỗi cặp `(Q, u_i)`:

1. Đưa prompt `Question + Schema unit` vào backbone LLM.
2. Lấy hidden state cuối `h_i` (ví dụ tại token `<eos>` hoặc token cuối của prompt).
3. Dự đoán score relevance bằng MLP:

   ```math
   p_i = sigmoid(MLP(h_i))
   ```

4. Chọn các unit có score vượt ngưỡng, hoặc lấy top-k, để tạo predicted sub-schema:

   ```text
   S_hat = {u_i | p_i > tau}
   ```

5. Sinh Cypher từ `Question + S_hat`.

### 4.1. Loss cho schema selection

Với nhãn nhị phân `z_i`, dùng binary cross entropy:

```math
L_select = -sum_i [z_i log(p_i) + (1-z_i) log(1-p_i)]
```

Vì số unit không liên quan thường lớn hơn nhiều số unit liên quan, cần cân nhắc weighted BCE, focal loss, hoặc hard-negative sampling.

### 4.2. Loss cho Cypher generation

Dùng causal language-modeling loss trên chuỗi Cypher vàng:

```math
L_gen = -sum_t log p_theta(y_t | y_<t, Q, S_input)
```

Trong giai đoạn đầu, `S_input` có thể là gold sub-schema `S*`. Để giảm chênh lệch train--test, nên có thêm fine-tuning với predicted/noisy sub-schema.

### 4.3. Joint multi-task training

Backbone LLM được dùng chung cho selection và generation. MLP là classification head cho selector; language-model head sinh Cypher. Huấn luyện với tổng loss:

```math
L = L_gen + lambda * L_select
```

Có thể triển khai theo hai lựa chọn:

1. **Two-stage**: huấn luyện selector, tạo predicted sub-schema, rồi huấn luyện/fine-tune generator với sub-schema này.
2. **Joint multi-task**: huấn luyện đồng thời `L_select` và `L_gen`. Đây là setting chính của phương pháp đề xuất.

## 5. Distillation

Để gọi đây là distillation một cách rõ ràng hơn, có thể đưa knowledge từ teacher vào cả hai task:

- **Generation KD**: distill token distribution, sequence-level preference, hoặc rationale/query của teacher cho Cypher generation.
- **Selection KD**: teacher mạnh xem câu hỏi cùng full schema và tạo soft relevance score/distribution cho mỗi schema unit.
- Student học đồng thời hard label từ Cypher vàng và soft label từ teacher.

Ví dụ selection loss mở rộng:

```math
L_select_total = L_hard + alpha * L_KD
```

Trong đó `L_hard` là BCE với nhãn suy từ gold Cypher và `L_KD` có thể là KL divergence/BCE giữa score của teacher và student.

## 6. Hướng tăng tính mới

Một MLP classifier thay cho việc sinh `yes/no` là một cải tiến hợp lý nhưng đơn lẻ có thể chưa đủ novelty. Các mở rộng tiềm năng:

1. **Structure-aware selection**: thêm ràng buộc graph consistency. Nếu một relationship được chọn, hai endpoint node tương ứng cần được ưu tiên chọn.
2. **Generator-aware selection**: selector không chỉ tối ưu membership label mà còn được tối ưu theo khả năng sub-schema hỗ trợ sinh query đúng.
3. **Contrastive grounding**: kéo biểu diễn câu hỏi gần relevant units, đẩy xa hard negative có lexical overlap.
4. **Multi-granularity retrieval**: chọn node, relation, property và path ở các mức khác nhau.
5. **Teacher-guided selection**: kết hợp hard labels suy từ gold Cypher với soft labels của teacher.

Hướng method mạnh nhất là: **multi-task, distillation-aware, structure-aware schema grounding cho Text-to-Cypher**.

## 7. Rủi ro và quyết định thiết kế

- Cần parser Cypher/AST đáng tin cậy; regex đơn giản dễ sai với alias, `OPTIONAL MATCH`, `WITH`, subquery, aggregation, direction, multi-hop path và relation property.
- Selection từng unit độc lập có thể chọn các thành phần đúng riêng lẻ nhưng không tạo được subgraph hợp lệ.
- Cần ưu tiên **recall** hơn precision: bỏ sót một relation cần thiết có thể khiến toàn bộ query sai, trong khi thêm một vài unit nhiễu đôi lúc vẫn cho phép generator trả lời đúng.
- Dùng threshold `tau` hoặc top-k cần được tune trên validation set; có thể đặt ràng buộc số unit hoặc ràng buộc connectivity.
- Không được dùng gold Cypher để extract sub-schema ở validation/test inference. Gold Cypher chỉ phục vụ xây dựng nhãn huấn luyện và đánh giá selection offline.

## 8. Đánh giá

### Query generation

- Exact Match.
- Execution Accuracy / Executable Accuracy.
- Valid Cypher rate.
- Token length, latency và chi phí so với dùng full schema.

### Schema selection

- Precision, Recall, F1 cho node/relation/property.
- Recall@k hoặc schema coverage.
- Graph validity/connectivity của predicted sub-schema.

### Ablation tối thiểu

| Thiết lập | Mục tiêu kiểm chứng |
| --- | --- |
| Full schema -> Cypher | Baseline không schema retrieval |
| Gold sub-schema -> Cypher | Upper bound của generator |
| Generative `yes/no` selector -> Cypher | Baseline selector bằng SFT |
| Hidden-state + MLP selector -> Cypher | Giá trị của classification head |
| Hidden-state + MLP + joint loss | Giá trị của multi-task training |
| Proposed + KD | Giá trị của distillation |
| Proposed + structural constraint | Giá trị của graph-aware selection |

Nên báo cáo riêng hiệu năng khi dùng gold sub-schema, predicted sub-schema và full schema. Việc này giúp tách biệt lỗi selector với lỗi generator.

## 9. Giả thuyết nghiên cứu

1. Relevant sub-schema giảm context noise và cải thiện Cypher execution accuracy so với dùng full schema.
2. Hidden-state classifier cho schema selection đạt recall/F1 tốt hơn cách để LLM sinh trực tiếp `yes/no`.
3. Joint optimization của selection và generation cải thiện hiệu năng end-to-end so với huấn luyện tách rời.
4. Distillation từ teacher và structural constraints tiếp tục cải thiện selection quality, đặc biệt trên schema lớn và câu hỏi multi-hop.
