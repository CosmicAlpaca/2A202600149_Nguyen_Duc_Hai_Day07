# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Đức Hải - 2A202600149
**Nhóm:** C401-A5
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
>  Cosine similarity cao (giá trị gần 1.0) chỉ ra rằng hai vector được biểu diễn cùng hướng trong không gian vector. Trong NLP, điều này đồng nghĩa với việc hai đoạn văn bản hàm chứa ngữ nghĩa  rất tương đồng.

**Ví dụ HIGH similarity:**
- Sentence A: Con chó đang nằm ngủ trên tấm thảm.
- Sentence B: Một chú cún đang say giấc trên thảm.
- Tại sao tương đồng: Cả hai câu đều mô tả cùng một hành vi của một loài thú cưng dù có sử dụng từ vựng khác nhau ("con chó" / "chú cún", "nằm ngủ" / "say giấc").

**Ví dụ LOW similarity:**
- Sentence A: Tôi thích uống cà phê đen vào sáng sớm.
- Sentence B: Lãi suất ngân hàng trung ương giảm mạnh trong tháng này.
- Tại sao khác: Hai câu thuộc hai domain  hoàn toàn không liên quan: một câu nói về sở thích cá nhân, trong khi câu kia đưa tin về tài chính kinh tế.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
>  Trong không gian text embeddings, Euclidean distance bị ảnh hưởng nặng bởi đặc điểm chiều dài văn bản . Ngược lại, Cosine similarity chỉ tập trung đo lường góc giữa hai vector, nên nó đối chiếu ngữ nghĩa chính xác hơn mà không bị sai lệch bởi độ dài ngắn của đoạn text.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* Kích thước của mỗi bước trượt (step) = chunk_size - overlap = 500 - 50 = 450. Chunk đầu tiên lấy 500 ký tự, phần còn dư 9500 ký tự cần chia qua các bước trượt 450, cần ceil(9500 / 450) = 22 chunk kế tiếp. (Công thức bao quát: `ceil((Total - chunk_size) / step) + 1`).
> *Đáp án:* Tổng cộng có 1 + 22 = 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
>  Nếu overlap tăng lên 100, bước trượt giảm chỉ còn 400. Dẫn đến số lượng chunk tính ra tăng lên thành ceil((10000 - 500) / 400) + 1 = 25 chunks. Việc tăng thông số overlap thường hữu ích nhằm bảo toàn ngữ cảnh  giữa các ranh giới cắt chữ để thông tin không bị gãy hoặc mất ý.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Customer Support Policy (E-commerce)

**Tại sao nhóm chọn domain này?**

> - Dễ kiếm doc (đổi trả, giao hàng, bảo hành, thanh toán…)
> - Có nhiều rule **rõ ràng + dễ sai**
> - RAG thể hiện rõ giá trị (retrieve đúng chunk hay không)

### Data Inventory


| #   | Tên tài liệu                                                                                        | Nguồn        | Số ký tự | Metadata đã gán                                                                                 |
| --- | --------------------------------------------------------------------------------------------------- | ------------ | -------- | ----------------------------------------------------------------------------------------------- |
| 1   | Chính sách hậu mãi: Đổi mới, trả hàng hoàn tiền và bảo hành sản phẩm                                | `data/1.md`  | 7277     | `source`, `extension`, `doc_title`, `doc_id`, `chunk_index`, `total_chunks`, `chunk_char_count` |
| 2   | Chính sách đổi trả sản phẩm                                                                         | `data/2.md`  | 5567     | `source`, `extension`, `doc_title`, `doc_id`, `chunk_index`, `total_chunks`, `chunk_char_count` |
| 3   | Các câu hỏi thường gặp về đổi trả                                                                   | `data/3.md`  | 4222     | `source`, `extension`, `doc_title`, `doc_id`, `chunk_index`, `total_chunks`, `chunk_char_count` |
| 4   | Hướng dẫn đổi trả online                                                                            | `data/4.md`  | 1688     | `source`, `extension`, `doc_title`, `doc_id`, `chunk_index`, `total_chunks`, `chunk_char_count` |
| 5   | Chính sách bảo hành tại Tiki như thế nào?                                                           | `data/5.md`  | 3035     | `source`, `extension`, `doc_title`, `doc_id`, `chunk_index`, `total_chunks`, `chunk_char_count` |
| 6   | Tiki hiện đang hỗ trợ các phương thức thanh toán nào                                                | `data/6.md`  | 2976     | `source`, `extension`, `doc_title`, `doc_id`, `chunk_index`, `total_chunks`, `chunk_char_count` |
| 7   | Làm thế nào để tôi có thể lưu và sử dụng mã coupon?                                                 | `data/7.md`  | 1262     | `source`, `extension`, `doc_title`, `doc_id`, `chunk_index`, `total_chunks`, `chunk_char_count` |
| 8   | Dịch vụ giao hàng từ nước ngoài                                                                     | `data/8.md`  | 3237     | `source`, `extension`, `doc_title`, `doc_id`, `chunk_index`, `total_chunks`, `chunk_char_count` |
| 9   | Dịch vụ giao hàng TikiNOW                                                                           | `data/9.md`  | 2244     | `source`, `extension`, `doc_title`, `doc_id`, `chunk_index`, `total_chunks`, `chunk_char_count` |
| 10  | Tôi có thể yêu cầu giao theo thời gian cụ thể, giao vào chủ nhật hoặc trên lầu/phòng chung cư không | `data/10.md` | 927      | `source`, `extension`, `doc_title`, `doc_id`, `chunk_index`, `total_chunks`, `chunk_char_count` |
---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `markdown/1.md` | FixedSizeChunker (`fixed_size`) | 49 | 197.5 | Không (dễ bị cắt giữa câu) |
| | SentenceChunker (`by_sentences`) | 14 | 516.7 | Tốt (giữ nguyên câu) |
| | RecursiveChunker (`recursive`) | 57 | 127.5 | Rất tốt (theo nội dung khối) |
| `markdown/10.md` | FixedSizeChunker (`fixed_size`) | 6 | 196.2 | Không (dễ bị cắt giữa câu) |
| | SentenceChunker (`by_sentences`) | 2 | 460.5 | Tốt (giữ nguyên câu) |
| | RecursiveChunker (`recursive`) | 7 | 132.1 | Rất tốt (theo nội dung khối) |

### Strategy Của Tôi

**Loại:** MarkdownChunker (custom strategy)

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?*
MarkdownChunker sử dụng lệnh phân cắt text để tách văn bản dựa trên các thẻ tiêu đề (heading) của định dạng Markdown (ví dụ: `\n## `). Cụ thể, mỗi phần của tài liệu nằm dưới một tiêu đề phụ sẽ được gom chung lại thành một chunk riêng biệt. Nếu nội dung không có dấu hiệu này, nó được giữ nguyên ranh giới mặc định.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?*
Với domain Customer Service FAQ của Tiki, hầu hết các câu trả lời đều được định dạng rõ ràng dưới một heading mô tả câu hỏi tư vấn khách hàng. Việc cắt theo heading (MarkdownChunker) sẽ đảm bảo thông tin của một chính sách không bị đứt đoạn, giúp Agent đọc toàn vẹn một nghiệp vụ tư vấn đầy đủ.

**Code snippet (nếu custom):**
```python
class MarkdownChunker:
    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        parts = text.split("\n## ")
        chunks = []
        for i, p in enumerate(parts):
            if i > 0 and not p.startswith("## "):
                chunks.append("## " + p.strip())
            else:
                chunks.append(p.strip())
        return [c for c in chunks if c.strip()]
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| `markdown/1.md` | best baseline (`recursive`) | 24 | 303.0 | Tốt (cắt theo đoạn văn lẻ) |
| | **của tôi** (`markdown`) | 12 | 604.5 | Rất tốt (giữ trọn 1 câu hỏi FAQ) |
| `markdown/10.md` | best baseline (`recursive`) | 3 | 309.0 | Tốt (cắt theo đoạn văn lẻ) |
| | **của tôi** (`markdown`) | 1 | 926.0 | Rất tốt (giữ trọn 1 chính sách) |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Nguyễn Đức Hải  | MarkdownChunker| 8/10 | | |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Viết 2-3 câu: dùng regex gì để detect sentence? Xử lý edge case nào?*
Sử dụng biểu thức chính quy (regular expression) `r'(?<=[.!?])\s+|\.\n'` để chia cắt văn bản nguyên bản. Cách này giữ lại được dấu ngắt câu `.`, `!`, `?` nhờ Lookbehind, loại bỏ các empty string và gộp những câu liên tiếp lại bằng vòng lặp step theo `max_sentences_per_chunk`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Viết 2-3 câu: algorithm hoạt động thế nào? Base case là gì?*
Thuật toán đệ quy kiểm tra xem đoạn text có khối lượng nhỏ hơn `chunk_size` không, hoặc danh sách separators có cạn không để làm điểm dừng (base case). Nếu chưa, văn bản sẽ tự động chẻ tại phần tử đầu tiên của `separators`, nếu dính đoạn quá dài quá thì gọi đệ quy tiếp với layer separator nhỏ hơn.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Viết 2-3 câu: lưu trữ thế nào? Tính similarity ra sao?*
Chức năng `add_documents` tuần tự tạo record object chứa content, id và metadata kèm theo giá trị embedding lấy từ `_embedding_fn()`, sau đó gọi client `_collection.add()` để chèn. Với `search`, hệ thống encode truy vấn thành vector rồi gọi `.query()` qua ChromaDB bằng chỉ số n_results để tìm các kết quả gần nhất dựa trên hàm tính khoảng cách nội suy.

**`search_with_filter` + `delete_document`** — approach:
> *Viết 2-3 câu: filter trước hay sau? Delete bằng cách nào?*
Thực hiện pre-filtering (lọc trước) thông qua tham số `where` của collection khi gọi query để giảm gian chi phí quét vector, chỉ tìm vector liên quan bên trong vùng metadata thỏa yêu cầu. Chức năng `delete_document` truyền trực tiếp danh sách mã định danh vào lệnh mảng `ids` trong method `.delete(ids=[doc_id])` để xoá triệt để các chunk của văn bản tương ứng.

### KnowledgeBaseAgent

**`answer`** — approach:
> *Viết 2-3 câu: prompt structure? Cách inject context?*
Chuẩn bị một prompt tĩnh định dạng có ranh giới phân biệt phần `Context:` và `Question:`. `Context` được sinh bằng việc tìm top_k relevant chunk và nối chuỗi bằng toán tử xuống dòng, sau đó chuyển xuống cho LLM tạo lập lời văn trả lời chi tiết.

### Test Results

```
================================================= test session starts =================================================
platform win32 -- Python 3.12.6, pytest-9.0.2, pluggy-1.6.0 -- C:\Python312\python.exe
cachedir: .pytest_cache
rootdir: D:\New folder\Day-07-Lab-Data-Foundations
plugins: anyio-4.13.0, langsmith-0.7.29
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                            [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                     [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                              [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                               [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                    [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                    [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                          [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                           [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                         [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                           [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                           [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                      [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                  [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                            [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                   [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                       [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                 [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                       [ 42%] 
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                             [ 47%] 
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                               [ 50%] 
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                     [ 52%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                          [ 54%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                            [ 57%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                [ 59%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                             [ 61%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                      [ 64%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                     [ 66%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                [ 69%] 
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                            [ 71%] 
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                       [ 73%] 
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                           [ 76%] 
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                 [ 78%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                           [ 80%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED        [ 83%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                      [ 85%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                     [ 88%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED         [ 90%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                    [ 92%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED             [ 95%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED   [ 97%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED       [100%] 

================================================= 42 passed in 0.94s =======================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Mèo là thú cưng rất đáng yêu. | Chó có lòng trung thành với chủ. | low | 0.5312 | Có |
| 2 | Hà Nội có nhiều quán bún chả. | Bún chả là món ăn ngon ở Hà Nội. | high| 0.7100 | Có |
| 3 | Thời tiết dạo này mưa rất to. | Máy tính của tôi bị hỏng màn hình. | low | 0.4253 | Có |
| 4 | Tiki giao hàng nhanh trong 2h. | Bạn có thể nhận hàng trên Tiki sau 2h. | high| 0.6824 | Có |
| 5 | Lãi suất vay đang giảm mạnh. | Doanh số quần áo tăng vào dịp tết. | low | 0.5101 | Có |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:* Kết quả đáng chú ý ở Câu 1 (chó vs mèo) và Câu 5 (lãi suất và doanh số dịp lễ) có điểm tương đồng không phải thấp tuyệt đối (trên mức `0.5`) dù ý nghĩa bề mặt không giống nhau. Điều này cho thấy thuật toán Embeddings có khả năng liên kết ngầm các vùng không gian ngữ nghĩa vĩ mô (cùng là "chủ đề thú cưng/động vật nuôi" hoặc "kinh tế/tiêu dùng") thay vì chỉ đối chiếu hình thức từ vựng rập khuôn.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Sau khi kiện trả hàng tới nhà bán thì bao lâu Tiki xử lý hoàn tiền? | Tiki hoàn tiền sau khi quy trình kiểm tra, đánh giá chất lượng sản phẩm đổi/trả hoàn tất; riêng phần xử lý này cần 3 ngày làm việc kể từ khi kiện hàng được chuyển tới nhà bán. |
| 2 | Nếu tôi gửi sản phẩm bảo hành về Tiki thì bao lâu nhận lại? | Nếu khách gửi hàng bảo hành về Tiki, thời gian bảo hành dự kiến là 15–30 ngày, chưa tính thời gian vận chuyển đi và về. |
| 3 | Đơn giao từ nước ngoài mà giao không thành công thì Tiki giao lại mấy lần, giữ kho bao lâu? | Với đơn giao từ nước ngoài, nếu giao không thành công thì Tiki hỗ trợ giao lại tối đa 03 lần; sau đó hàng được giữ tại kho Tiki 14 ngày. Nếu quá thời hạn đó khách không liên hệ nhận hàng thì Tiki tiến hành hoàn tiền qua đơn hàng. |
| 4 | Muốn dùng Tiki Xu và mã giảm giá thì có điều kiện gì? | Khách chỉ có thể dùng Tiki Xu khi có từ 1000 Xu trở lên; còn mỗi mã giảm giá chỉ dùng 1 lần trên 1 tài khoản. |
| 5 | Tôi có thể yêu cầu giao vào giờ cụ thể hoặc hẹn lại chủ nhật không? | Sau khi đặt hàng thành công, Tiki sẽ thông báo thời gian giao dự kiến. Nếu thời điểm shipper liên hệ chưa phù hợp, khách có thể trao đổi qua điện thoại để hẹn lại thời gian giao khác, và nhân viên vận chuyển sẽ cố gắng hỗ trợ trong mức có thể. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Sau khi kiện trả hàng tới nhà bán thì bao lâu Tiki xử lý hoàn tiền? | ## 2. Đối tượng áp dụng  Khách hàng mua hàng hóa từ nhà bán hàng ngoài lãnh thổ Việt Nam trên trang thương mại điện tử: ... | 0.5477 | Không | Dựa trên thông tin bạn cung cấp, **không có thông tin nào nói về thời gian Tiki xử lý hoàn tiền sau khi kiện trả hàng tới nhà bán.** Các đoạn văn bản... |
| 2 | Nếu tôi gửi sản phẩm bảo hành về Tiki thì bao lâu nhận lại | ## 3. Sau bao lâu tôi có thể nhận lại sản phẩm bảo hành? Nếu sản phẩm của quý khách vẫn còn trong thời hạn bảo hành, Ti... | 0.3336 | Có | Nếu quý khách gửi sản phẩm bảo hành về Tiki, thời gian nhận lại dự kiến là **15 - 30 ngày** (không tính thời gian vận chuyển đi và về). Tuy nhiên, thờ... |
| 3 | Đơn giao từ nước ngoài mà giao không thành công thì Tiki giao lại mấy lần, giữ kho bao lâu? | # Dịch vụ giao hàng TikiNOW  > Cập nhật lần cuối: 9 ngày trướcLượt xem: 506 Khi cần Tiki có, khi khó Tiki giao! Giờ đây... | 0.4868 | Không | Dựa trên thông tin được cung cấp, **không có thông tin chi tiết về số lần giao lại hoặc thời gian giữ kho khi đơn hàng giao từ nước ngoài không thành ... |
| 4 | Muốn dùng Tiki Xu và mã giảm giá thì có điều kiện gì? | ### c. Làm sao để tôi biết được sản phẩm mình muốn mua có áp dụng dịch vụ TikiNOW hay không? Những sản phẩm có áp dụng ... | 0.4180 | Không | Dựa trên nội dung bạn cung cấp, không có thông tin nào đề cập đến điều kiện sử dụng Tiki Xu và mã giảm giá. Các phần trong văn bản chỉ tập trung vào:... |
| 5 | Tôi có thể yêu cầu giao vào giờ cụ thể hoặc hẹn lại chủ nhật không? | ### c. Chính sách hủy và hoàn tiền: Tiki vẫn hỗ trợ hủy, hoàn lại giá trị đã thanh toán thực tế cho sản phẩm và phí vận... | 0.3868 | Có | Không, Tiki không hỗ trợ giao hàng theo giờ cụ thể hoặc hẹn lại vào chủ nhật. Tiki giao hàng trong giờ hành chính, từ 8h30 sáng đến 17h30 chiều vào cá... |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 2 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:* Tôi đã học được cách các bạn tối ưu hóa hệ thống bằng cách gán thêm Metadata chi tiết (như category, keywords) cho từng chunk thay vì chỉ dừng ở mức document. Việc này giúp quá trình `search_with_filter` hoạt động hiệu quả hơn hẳn, giúp lọc bỏ nhiễu từ các chính sách không liên quan trước khi tính toán similarity.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:* Qua demo của các nhóm khác, tôi thấy ấn tượng với việc sử dụng Hybrid Search (kết hợp vector search và keyword search) để xử lý các câu hỏi chứa thuật ngữ chuyên môn hoặc mã số chính sách cụ thể. Ngoài ra, kỹ thuật Re-ranking cũng là một bài học quý giá giúp họ đạt độ chính xác cao hơn khi xử lý các chunk có độ tương đồng gần bằng nhau.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:* Tôi sẽ thực hiện tiền xử lý dữ liệu (data cleaning) kỹ hơn để loại bỏ các phần navigation hoặc footer thừa trong file Markdown, tránh làm loãng vector embedding. Đồng thời, tôi sẽ áp dụng kỹ thuật "Small-to-Big Retrieval" để đảm bảo Agent nhận được đủ ngữ cảnh bao quanh khi truy vấn mà vẫn giữ được độ tập trung của embedding ban đầu.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5/ 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm |  / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân |30 / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
