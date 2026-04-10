# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Mạc Phạm Thiên Long

**Nhóm:** C401-E2

**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
High cosine similarity biểu thị góc giữa hai vector trong không gian nhiều chiều rất nhỏ (gần bằng 0), đồng nghĩa với việc hai đoạn văn bản có sự tương đồng cao về mặt ngữ nghĩa (semantic meaning), bất kể độ dài của chúng.

**Ví dụ HIGH similarity:**
- **Sentence A:** "Chú chó nhỏ đang nằm ngủ say sưa trên tấm thảm."
- **Sentence B:** "Một con cún đang ngủ say giấc trên chiếc thảm."
- **Tại sao tương đồng:** Cả hai câu đều miêu tả cùng một ngữ cảnh và ý nghĩa (một con vật nuôi (con chó) đang ngủ trên thảm), dù sử dụng các từ vựng đồng nghĩa khác nhau. Vector của chúng sẽ cùng hướng trong không gian.

**Ví dụ LOW similarity:**
- **Sentence A:** "Chú chó nhỏ đang nằm ngủ say sưa trên tấm thảm."
- **Sentence B:** "Lãi suất ngân hàng vừa tăng thêm 0.5% vào sáng nay."
- **Tại sao khác:** Hai câu này thuộc hai lĩnh vực hoàn toàn khác biệt (thú cưng và kinh tế vĩ mô), không chia sẻ chung bất kỳ ngữ nghĩa hay từ vựng nào. Vector của chúng sẽ gần như vuông góc.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
Cosine similarity chỉ quan tâm đến hướng của vector (ngữ nghĩa) mà bỏ qua độ lớn (độ dài văn bản hoặc tần suất từ xuất hiện), giúp đánh giá chính xác sự tương đồng ngay cả khi so sánh một câu ngắn với một đoạn văn dài. Trong khi đó, Euclidean distance đo khoảng cách tuyệt đối nên rất dễ bị sai lệch bởi chênh lệch độ dài văn bản.

---

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
**Trình bày phép tính:**
- Bước nhảy (Stride) = `chunk_size` - `overlap` = 500 - 50 = 450 ký tự.
- Tổng số ký tự cần xử lý sau khi trừ đi chunk đầu tiên = Tổng ký tự - `overlap` = 10,000 - 50 = 9,950 ký tự.
- Số lượng chunk tính toán = 9,950 / 450 ≈ 22.11.
- Vì không thể có một chunk lẻ, ta làm tròn lên (ceiling) -> 23 chunks.
**Đáp án:** 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
- Nếu overlap = 100, bước nhảy sẽ là 400. Số chunk = làm tròn lên của (10,000 - 100) / 400 = 9,900 / 400 = 24.75 -> **25 chunks**. Số lượng chunk tăng lên.
- **Tại sao muốn overlap nhiều hơn:** Việc tăng overlap giúp đảm bảo các câu, các đoạn văn hoặc các ý quan trọng không bị cắt xén mất bối cảnh ở ranh giới giữa hai chunk. Điều này đặc biệt quan trọng trong các hệ thống RAG để LLM luôn nhận được ngữ cảnh trọn vẹn nhất.

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Y tế (cụ thể Vinmec)

**Tại sao nhóm chọn domain này?**
Nhóm chọn lĩnh vực y tế, đặc biệt là hệ thống Vinmec, vì đây là môi trường trọng yếu có tiềm năng chuyển đổi số và ứng dụng AI cực kỳ lớn để tối ưu hóa trải nghiệm bệnh nhân lẫn quy trình vận hành y tế. Hơn nữa, Vinmec là một trong những đơn vị tư nhân tiên phong xây dựng mô hình "bệnh viện thông minh" tại Việt Nam, mang đến những bài toán nghiệp vụ phức tạp và nguồn dữ liệu phong phú, giúp dự án mang lại giá trị thực tiễn cao.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | addison-suy-tuyen-thuong-than-nguyen-phat-4696.md | Vinmec | 7413 | `source`: Tên tài liệu, `category`: Bệnh học |
| 2 | aids-4699.md | Vinmec | 6957 | `source`: Tên tài liệu, `category`: Bểnh học |
| 3 | alkapton-nieu-4695.md | Vinmec | 10762 | `source`: Tên tài liệu, `category`: Bệnh hiếm |
| 4 | alzheimer-3112.md | Vinmec | 5064 | `source`: Tên tài liệu, `category`: Tuổi già |
| 5 | am-anh-so-hai-4678.md | Vinmec | 8533 | `source`: Tên tài liệu, `category`: Tâm lý |
| 6 | amip-an-nao-3153.md | Vinmec | 7380 | `source`: Tên tài liệu, `category`: Truyền nhiễm |
| 7 | ap-xe-3331.md | Vinmec | 12705 | `source`: Tên tài liệu, `category`: Nhiễm trùng |
| 8 | ap-xe-gan-3238.md | Vinmec | 7784 | `source`: Tên tài liệu, `category`: Khoa nội |
| 9 | ap-xe-gan-do-amip-4748.md | Vinmec | 15405 | `source`: Tên tài liệu, `category`: Khoa nội |
| 10 | ap-xe-hau-mon-3332.md | Vinmec | 7697 | `source`: Tên tài liệu, `category`: Tiêu hóa |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `source` | `str` | `ap-xe-nao-3205.md` | Giúp xác định chính xác nguồn gốc dòng thông tin, hỗ trợ filter chi tiết hoặc dùng để tạo trích dẫn "Nguồn tham khảo:..." cho Agent. |
| `category` | `str` | `Bệnh lý thần kinh` | Tăng độ chính xác khi truy vấn. Ví dụ: khi được hỏi về não, có thể pre-filter để Agent bỏ qua các tài liệu tiêu hóa/truyền nhiễm, thu hẹp không gian tìm kiếm. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 3 tài liệu tiêu biểu: `aids-4699.md`, `am-anh-so-hai-4678.md`, `ap-xe-nao-3205.md`.

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `aids-4699.md` | FixedSizeChunker (`fixed_size`) | 16 | 481.69 | **No**: Cắt giữa chừng, mất ngữ nghĩa. |
| `aids-4699.md` | SentenceChunker (`by_sentences`) | 24 | 288.62 | **Yes**: Giữ trọn vẹn câu. |
| `aids-4699.md` | RecursiveChunker (`recursive`) | 21 | 329.76 | **Best**: Giữ cấu trúc đoạn văn/câu. |
| `am-anh-so-hai-4678.md` | FixedSizeChunker (`fixed_size`) | 19 | 496.47 | **No**: Chia cắt từ ngữ cơ học. |
| `am-anh-so-hai-4678.md` | SentenceChunker (`by_sentences`) | 26 | 326.96 | **Yes**: Đảm bảo câu hoàn chỉnh. |
| `am-anh-so-hai-4678.md` | RecursiveChunker (`recursive`) | 24 | 354.08 | **Best**: Ưu tiên ngắt ở đoạn/dòng. |
| `ap-xe-nao-3205.md` | FixedSizeChunker (`fixed_size`) | 23 | 495.04 | **No**: Rủi ro mất thông tin đầu/cuối. |
| `ap-xe-nao-3205.md` | SentenceChunker (`by_sentences`) | 26 | 394.38 | **Yes**: Giữ bối cảnh trong câu. |
| `ap-xe-nao-3205.md` | RecursiveChunker (`recursive`) | 35 | 292.29 | **Best**: Linh hoạt theo cấu trúc tài liệu. |

### Strategy Của Tôi

**Loại:** Document-structure Chunking (custom strategy)

**Mô tả cách hoạt động:**
Chiến lược này tôi thực hiện dựa trên việc phân tích cấu trúc tiêu đề của tài liệu Markdown. Thay vì cắt văn bản một cách cơ học, tôi sử dụng Regex để nhận diện các dòng bắt đầu bằng các ký hiệu tiêu đề như `#`, `##` hoặc `###`. Khi gặp một tiêu đề mới, hệ thống sẽ tự động đóng gói nội dung cũ và bắt đầu một chunk mới, đảm bảo tiêu đề luôn nằm ở đầu mỗi chunk để giữ ngữ cảnh.

**Tại sao tôi chọn strategy này cho domain nhóm?**
Trong lĩnh vực Y tế, các bài viết thường được chia thành các phân mục rất rõ ràng và mạch lạc như: Triệu chứng, Chẩn đoán, và Điều trị. Việc chunking theo cấu trúc giúp tôi giữ lại trọn vẹn thông tin chuyên môn trong cùng một mục, tránh việc các hướng dẫn điều trị quan trọng bị cắt rời, từ đó giúp agent truy xuất được dữ liệu chính xác và an toàn nhất.

**Code snippet (nếu custom):**
```python
import re

class DocumentStructureChunker:
    """
    Chiến lược chunking dựa trên cấu trúc tiêu đề của tài liệu.
    """
    def __init__(self, header_patterns: list[str] | None = None):
        self.header_patterns = header_patterns or [r'^#\s+', r'^##\s+', r'^###\s+']

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
            
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        
        header_regex = "|".join(self.header_patterns)
        
        for line in lines:
            if re.match(header_regex, line):
                if current_chunk:
                    chunks.append("\n".join(current_chunk).strip())
                current_chunk = [line]
            else:
                current_chunk.append(line)
        
        if current_chunk:
            chunks.append("\n".join(current_chunk).strip())
            
        return [c for c in chunks if c]
```
### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| `aids-4699.md` | RecursiveChunker (Best Baseline) | 21 | 329.76 |Tốt nhưng đôi chỗ bị chia quá vụn. |
| `aids-4699.md` | Document-structure | 17 | 407.29 |Cấu trúc bệnh lý nằm trọn trong một chunk. |
| `am-anh-so-hai-4678.md` | RecursiveChunker (Best Baseline) | 24 | 354.08 |Tốt, giữ được ngữ nghĩa cấp độ đoạn văn. |
| `am-anh-so-hai-4678.md` | Document-structure | 17 | 500.00 |Gom trọn phương pháp điều trị không bị cắt nhỏ quá nhiều. |
| `ap-xe-nao-3205.md` | RecursiveChunker (Best Baseline) | 35 | 292.29 |Khá rời rạc, quá nhiều chunk nhỏ làm loãng context. |
| `ap-xe-nao-3205.md` | Document-structure | 20 | 512.35 |Dấu hiệu và biến chứng nằm chung bối cảnh. |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Mạc Phạm Thiên Long | Document-structure Chunking | 8.5 | Nội dung liền mạch, kết quả 4/5 Top-3 | File quá dài có thể sẽ làm chunk quá to |
| Nguyễn Doãn Hiếu | RecursiveChunker | 8.0 | Chia chunk linh hoạt, kích thước vừa phải | Dễ cắt đứt nội dung của danh sách/phác đồ |
| Cao Chí Hải | Semantic Chunking | 6.5 | Q2 precision=1.0; chunk=0.5 tạo 468 chunks coherent, giữ nguyên ý tốt với đoạn dài | Q3 fail hoàn toàn khi search≥0.5; `vi_retrieval_notes` át các file khác do cross-lingual gap | Dùng chunk=0.5 + search=0.4 + top_k=3; thêm metadata filter theo ngôn ngữ |
| Phan Hoài Linh| Semantic Chunking | 7.0 | Giữ nguyên ý tưởng, phù hợp với đoạn dài|  Phụ thuộc vào chất lượng embedding |
| Nguyễn Văn Đạt | Document-structure (custom) + OpenAI embeddings | 10.0 | 5/5 queries top-1 đúng expected_source (hit@5=1.00, MRR=1.00) | Chưa chứng minh hơn baseline (đang hòa); tốn chi phí/độ trễ do gọi API |
| Bùi Hữu Huấn | Recursive Character Splitting | 6.7 / 10 | Retrieval tốt với 2/3 query đạt tối đa | Fail hoàn toàn 1 query (Áp xe gan → 0/2) |


**Strategy nào tốt nhất cho domain này? Tại sao?**
Theo kết quả đánh giá (bảng `Kết Quả Của Tôi`), chiến lược **Document-structure Chunking** do em lựa chọn hoạt động tốt nhất. Đặc thù tài liệu Vinmec là chia thành các phần rõ rệt (Nguyên nhân, Triệu chứng, Điều trị), chia cắt theo cấu trúc Markdown (`#`) giúp không làm rách mạch phác đồ (medical regimens), giúp RAG luôn có đủ lượng Text để trả lời trọn vẹn câu mà không lo bị cắt xén lưng chừng.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
Sử dụng biểu thức chính quy (`regex`) với kỹ thuật look behind `(?<=[.!?])\s+` để tách văn bản thành các câu mà không làm mất đi dấu câu ở cuối. Sau khi có danh sách các câu, gom nhóm chúng lại thành từng chunk dựa trên tham số mục tiêu, giúp giữ được sự mạch lạc về ngữ nghĩa trong mỗi đơn vị thông tin.

**`RecursiveChunker.chunk` / `_split`** — approach:
Áp dụng thuật toán chia để trị (divide and conquer) thông qua đệ quy. Văn bản sẽ được tách bằng các ký tự phân cách theo thứ tự ưu tiên giảm dần: từ đoạn văn (`\n\n`), dòng (`\n`), câu (`. `) cho đến khi đạt được kích thước nhỏ hơn `chunk_size`. Điểm then chốt là nếu một phần vẫn quá lớn sau khi dùng hết các separator, dùng `FixedSizeChunker` làm phương án dự phòng cuối cùng.

### EmbeddingStore

**`add_documents` + `search`** — approach:
Mỗi tài liệu khi nạp vào sẽ được vector hóa qua hàm embedding và lưu trữ kèm metadata trong danh sách (`list[dict]`) hoặc ChromaDB. Trong quá trình tìm kiếm,  ính toán độ tương đồng giữa vector câu hỏi và các chunk bằng phép toán dot product. Do các vector đã được chuẩn hóa từ trước, kết quả này phản ánh chính xác Cosine similarity.

**`search_with_filter` + `delete_document`** — approach:
Để tối ưu, thực hiện lọc metadata (pre-filtering) trước khi tính toán độ tương đồng vector. Với việc xóa tài liệu, sử dụng `doc_id` làm khóa định danh duy nhất trong metadata để tìm và loại bỏ tất cả các chunk liên quan, đảm bảo dữ liệu trong store luôn sạch và đồng nhất.

### KnowledgeBaseAgent

**`answer`** — approach:
Xây dựng prompt theo cấu trúc "Context-First": đầu tiên lấy Top-K chunk liên quan nhất từ Store, nối chúng lại thành một khối ngữ cảnh (context) và đưa vào prompt cùng với câu hỏi của người dùng. Thêm chỉ dẫn nghiêm ngặt để Agent chỉ trả lời dựa trên ngữ cảnh này, giúp hạn chế việc mô hình tự ý đưa ra thông tin không có trong tài liệu.

### Test Results

```
pytest tests/
====================================================== test session starts =======================================================
platform win32 -- Python 3.13.9, pytest-9.0.2, pluggy-1.6.0
rootdir: E:\VinAI\assignments\2A202600384-MacPhamThienLong-Day07
plugins: Faker-40.4.0, langsmith-0.7.6, asyncio-1.3.0, anyio-4.10.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 42 items

tests\test_solution.py ..........................................                                                           [100%]

======================================================= 42 passed in 0.50s =======================================================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Bệnh nhân bị sốt cao và đau bụng vùng hạ sườn phải. | Triệu chứng phổ biến nhất là sốt và đau ở hạ sườn bên phải. | High | 0.5959 | Yes |
| 2 | Amip xâm nhập vào cơ thể qua niêm mạc mũi. | Ký sinh trùng đi vào người bằng đường hô hấp qua mũi. | High | 0.5304 | Yes |
| 3 | Lãi suất tiền gửi tiết kiệm đang giảm mạnh. | Tỷ lệ lạm phát đang có dấu hiệu hạ nhiệt. | Low | 0.4095 | Yes |
| 4 | Cần phẫu thuật để nạo vét hết các ổ mủ. | Điều trị bằng thuốc kháng sinh liều cao. | Low | 0.3654 | Yes |
| 5 | Nước tiểu sẽ chuyển sang màu đen khi để ngoài không khí. | Màu sắc của nước tiểu thay đổi do phản ứng oxy hóa với khí quyển. | High | 0.6410 | Yes |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
Kết quả bất ngờ nhất là cặp `3` (Lãi suất/Lạm phát) lại có mức điểm similarity lên đến `0.4095`. Điều này chỉ ra rằng Embedding models (đặc biệt là model từ pre-training) không chỉ đánh giá sự giống nhau tuyệt đối về cấu trúc, mà nó mang cả sự hiểu biết về "trường từ vựng liên quan" (cụ thể là khái niệm kinh tế vĩ mô), khiến điểm số của những chủ đề liên quan (related topic) không bao giờ hạ sát về 0 tuyệt đối dù chúng là hai hiện tượng khác nhau.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers 

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Bệnh Amip ăn não (Naegleria fowleri) lây nhiễm vào cơ thể qua đường nào và thời gian ủ bệnh thường là bao lâu? | Amip xâm nhập vào cơ thể qua mũi khi tiếp xúc với nước hoặc bụi bẩn (thường ở ao, hồ nước ngọt ấm), sau đó đi dọc theo dây thần kinh khứu giác lên não. Các triệu chứng thường bắt đầu xuất hiện từ 2 đến 15 ngày sau khi nhiễm amip. |
| 2 | Những nguyên nhân nào khiến tình trạng áp xe hậu môn dễ bị tái phát sau khi đã tiến hành điều trị? | Áp xe hậu môn dễ tái phát do: (1) dùng kháng sinh chưa đủ mạnh để tiêu diệt mầm bệnh; (2) điều trị bằng bài thuốc dân gian nhưng không kiên trì; (3) phẫu thuật chưa đúng quy trình, chưa nạo vét hết ổ áp xe và dịch mủ; (4) hệ miễn dịch của người bệnh kém. |
| 3 | Tam chứng Fontam điển hình trong bệnh áp xe gan do amip bao gồm những biểu hiện lâm sàng nào? | Tam chứng Fontam bao gồm 3 triệu chứng chủ yếu: sốt (sốt nhẹ hoặc lên tới 39-40°C), đau hạ sườn phải và vùng gan (đau tức nặng, có thể xuyên lên vai phải), và gan to, đau (gan to 3-4 cm dưới sườn phải, ấn đau, làm nghiệm pháp rung gan dương tính). |
| 4 | Bệnh Alkapton niệu (nước tiểu sẫm màu) hình thành do đột biến gen nào và tại sao nước tiểu của người bệnh lại chuyển màu đen? | Bệnh hình thành do đột biến gen HGD làm giảm vai trò của enzyme homogentisate oxygenase. Sự thiếu hụt này làm axit homogentisic không bị phá vỡ mà tích tụ trong cơ thể. Khi axit này được bài tiết qua nước tiểu và tiếp xúc với không khí, nó sẽ bị oxy hóa và làm nước tiểu chuyển sang màu sẫm nâu đen. |
| 5 | Các phương pháp chính được sử dụng để điều trị hội chứng rối loạn ám ảnh sợ hãi là gì? | Phương pháp điều trị chính bao gồm sử dụng thuốc (như thuốc an thần giải lo âu, thuốc SSRI để giảm hoảng sợ, giảm nhịp tim) kết hợp với liệu pháp hành vi (để người bệnh tưởng tượng tiếp xúc với sự vật gây sợ hãi). Ngoài ra, có thể áp dụng thêm thôi miên hoặc phản hồi sinh học. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Bệnh Amip ăn não (Naegleria fowleri) lây nhiễm vào cơ thể qua đường nào và thời gian ủ bệnh thường là bao lâu? | `### Triệu chứng bệnh Amip ăn não`:  Khi người bệnh nhiễm amip Naegleria sẽ gây ra một bệnh gọi là Viêm não màng não tiên phát do amip (Primary amebic me...) | 0.730 | Yes | Bệnh Amip ăn não (Naegleria fowleri) lây nhiễm vào cơ thể qua mũi khi mũi tiếp xúc với nước hoặc bụi bẩn. Thời gian ủ bệnh thường từ hai đến 15 ngày sau khi nhiễm. |
| 2 | Những nguyên nhân nào khiến tình trạng áp xe hậu môn dễ bị tái phát sau khi đã tiến hành điều trị? | `## Nguyên nhân bệnh Áp xe hậu môn...` | 0.705 | No | **Tôi không biết.**|
| 3 | Tam chứng Fontam điển hình trong bệnh áp xe gan do amip bao gồm những biểu hiện lâm sàng nào? | `## Triệu chứng bệnh Áp xe gan do amip...`: Chứa thông tin về tam chứng. | 0.729 | Yes | Tam chứng Fontan trong bệnh áp xe gan do amip bao gồm sốt, gan lớn và đau. |
| 4 | Bệnh Alkapton niệu (nước tiểu sẫm màu) hình thành do đột biến gen nào và tại sao nước tiểu của người bệnh lại chuyển màu đen? | `### Tổng quan bệnh Alkapton niệu`: Giải thích về đột biến gen HGD và axit homogentisic. | 0.765 | Yes | Đột biến gen HGD, axit homogentisic bị oxy hóa ngoài không khí. |
| 5 | Các phương pháp chính được sử dụng để điều trị hội chứng rối loạn ám ảnh sợ hãi là gì? | `### Các biện pháp điều trị bệnh Ám ảnh sợ hãi`: Chứng ám ảnh sợ hãi có thể được điều trị khỏi hoàn toàn. Nguyên tắc chính của việc điều trị là tránh xa... | 0.754 | Yes |Các phương pháp chính để điều trị hội chứng rối loạn ám ảnh sợ hãi bao gồm thuốc (như thuốc an thần giải lo âu, thuốc SSRI) và liệu pháp hành vi. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 4 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
Tôi hiểu được tầm quan trọng của việc tuning tham số `overlap` khi xử lý tài liệu liên tục ở chiến lược `RecursiveChunker`, giúp giải quyết triệt để vấn đề mất liên kết ngữ cảnh ở các câu nối tiếp.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
Đó là tư duy sử dụng công cụ metadata filtering từ ChromaDB trước khi tính vector similarity. Việc này tăng tốc quá trình truy xuất dữ liệu thay vì bắt embedding phải so khớp trên toàn bộ hàng triệu chunks của dự án lớn.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
Tôi sẽ sử dụng kỹ thuật hybrid search (kết hợp BM25 keyword search và vector search). Như đã thấy ở `Failure Analysis` mục 6, từ khóa "tái phát" bị mất vì similarity model đánh giá từ khóa "Nguyên nhân" tổng quan cao hơn nội dung cụ thể trong văn bản. Hybrid search sẽ cứu vãn những trường hợp tìm kiếm cụ thể và ngách như thế này.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 10 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 3 / 5 |
| **Tổng** | | **82 / 100 (* điểm tự chấm*)** |
