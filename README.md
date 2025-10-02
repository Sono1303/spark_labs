# Báo cáo Bài tập Lab17 - Spark NLP Pipeline

## 1. CÁC BƯỚC THỰC HIỆN (Implementation Steps)

### 1.1 Thiết lập môi trường phát triển
- **Ngôn ngữ lập trình**: Scala 2.12.x
- **Framework**: Apache Spark 3.5.1 với MLlib
- **Build tool**: SBT (Simple Build Tool) 1.11.6
- **Java Runtime**: OpenJDK 17 LTS
- **Dữ liệu**: C4 Common Crawl dataset (30K records)
- **Document limit**: Configurable với `limitDocuments` variable (default: 1000 records)
- **Performance monitoring**: Detailed timing measurements cho từng processing stage
- **Vector normalization**: L2 (Euclidean) normalization
- **Document similarity**: Cosine similarity với top-K document search
### 1.2 Cấu hình project trong build.sbt
```scala
name := "spark-nlp-labs"
version := "0.1"
scalaVersion := "2.12.19"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.5.1",
  "org.apache.spark" %% "spark-sql" % "3.5.1", 
  "org.apache.spark" %% "spark-mllib" % "3.5.1"
)
```

### 1.3 Thiết kế kiến trúc Pipeline NLP
Pipeline được thiết kế theo mô hình ETL (Extract-Transform-Load) với 8 giai đoạn:

1. **Stage 1**: Khởi tạo Spark Session với performance monitoring
2. **Stage 2**: Đọc dữ liệu C4 dataset (với limitDocuments configurable)
3. **Stage 3**: Tạo ML Pipeline components (RegexTokenizer → StopWordsRemover → HashingTF → IDF → Normalizer)
4. **Stage 4**: Training NLP Pipeline với detailed timing
5. **Stage 5**: Applying Pipeline Transformation với caching
6. **Stage 6**: Analyzing Vocabulary Statistics và vector properties
7. **Stage 7**: Collecting Results và Writing Files
8. **Stage 8**: Finding Similar Documents với cosine similarity analysis

### 1.4 Cài đặt chi tiết từng thành phần

#### a) Khởi tạo Spark Session
```scala
val spark = SparkSession.builder()
  .appName("NLP Pipeline Example")
  .master("local[*]")
  .config("spark.sql.adaptive.enabled", "true")
  .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
  .getOrCreate()
```

#### b) Đọc dữ liệu C4 Dataset
```scala
val df = spark.read.json("data/c4-train.00000-of-01024-30K.json.gz")
  .limit(1000) // Giới hạn 1000 records để xử lý nhanh hơn
```

#### c) RegexTokenizer để tách từ
```scala
val tokenizer = new RegexTokenizer()
  .setInputCol("text")
  .setOutputCol("words")
  .setPattern("\\W") // Sử dụng regex để tách theo ký tự không phải chữ
```

#### d) StopWordsRemover để loại bỏ stop words
```scala
val stopWordsRemover = new StopWordsRemover()
  .setInputCol("words")
  .setOutputCol("filtered_words")
```

#### e) HashingTF và IDF cho vectorization
```scala
val hashingTF = new HashingTF()
  .setInputCol("filtered_words")
  .setOutputCol("tf_features")
  .setNumFeatures(20000) // Kích thước feature vector: 20,000 chiều

val idf = new IDF()
  .setInputCol("tf_features")
  .setOutputCol("raw_features")
```

#### f) Normalizer để chuẩn hóa vectors
```scala
val normalizer = new Normalizer()
  .setInputCol("raw_features")
  .setOutputCol("features")
  .setP(2.0) // L2 (Euclidean) normalization
```

#### g) Tạo và thực thi Pipeline
```scala
val pipeline = new Pipeline()
  .setStages(Array(tokenizer, stopWordsRemover, hashingTF, idf, normalizer))

val pipelineModel = pipeline.fit(df)
val transformedDF = pipelineModel.transform(df).cache() // Caching cho performance
```

#### h) Document Similarity Analysis
```scala
// Cosine similarity UDF cho normalized vectors
val cosineSimilarity = udf((ref: Vector, vec: Vector) => {
  val refArray = ref.toArray
  val vecArray = vec.toArray
  refArray.zip(vecArray).map { case (a, b) => a * b }.sum
})

// Tìm top 5 documents tương tự nhất
val similarDocs = transformedDF
  .withColumn("similarity", cosineSimilarity(col("features"), referenceVector))
  .orderBy(desc("similarity"))
  .limit(5)
```

## 2. CÁCH CHẠY CODE VÀ GHI LOG KẾT QUẢ (How to Run and Log Results)

### 2.1 Chuẩn bị dữ liệu và môi trường
```bash
# Tạo thư mục dữ liệu
mkdir -p data/

# Tải dữ liệu C4 (nếu chưa có)
# Đặt file c4-train.00000-of-01024-30K.json.gz vào thư mục data/
```

### 2.2 Các bước chạy chương trình
```bash
# Bước 1: Di chuyển đến thư mục project
cd E:\NLP\spark_labs

# Bước 2: Compile code Scala
sbt compile

# Bước 3: Chạy chương trình chính
sbt "runMain com.lhson.spark.Lab17_NLPPipeline"
```

### 2.3 Hệ thống logging và monitoring
- **Real-time console output**: Hiển thị progress và thống kê trực tiếp
- **Spark UI**: Truy cập `http://localhost:4040` để theo dõi job execution
- **Log file**: `log/lab17_metrics.log` - Ghi các thông số hiệu năng
- **Results file**: `results/lab17_pipeline_output.txt` - Lưu kết quả xử lý

### 2.4 Cấu trúc file output

#### Log file (lab17_metrics.log):
```
--- NLP Pipeline Processing Log ---
Pipeline fitting duration: 1.64 seconds
Data transformation duration: 0.56 seconds
Actual vocabulary size (after preprocessing): 27009 unique terms
HashingTF feature vector size: 20000
Records processed: 1000
```

#### Results file (lab17_pipeline_output.txt):
```
--- NLP Pipeline Output (First 20 results) ---
================================================================================
Original Text: Beginners BBQ Class Taking Place in Missoula!...
Tokenized Words: beginners, bbq, class, taking, place...
Filtered Words: beginners, bbq, class, taking, place, missoula...
Feature Vector Size: 20000
================================================================================
```

## 3. GIẢI THÍCH KẾT QUẢ ĐẠT ĐƯỢC (Results Explanation)

### 3.1 Thống kê tổng quan
- **Số lượng records xử lý**: 1,000 documents từ C4 dataset
- **Thời gian khởi tạo Spark**: 9.32 giây (63.5%)
- **Thời gian đọc dữ liệu**: 2.74 giây (18.7%)
- **Thời gian tạo pipeline**: 0.056 giây (0.4%)
- **Thời gian fitting pipeline**: 1.61 giây (10.9%)
- **Thời gian transform dữ liệu**: 0.61 giây (4.2%)
- **Thời gian phân tích vocabulary**: 0.27 giây (1.9%)
- **Thời gian similarity analysis**: 1.13 giây
- **Tổng thời gian xử lý**: 25.04 giây
- **Overall throughput**: 40 records/giây
- **Training throughput**: 622 records/giây
- **Transform throughput**: 1632 records/giây
- **Kích thước từ vựng**: 27,009 từ unique (sau khi loại bỏ stop words)
- **Kích thước feature vector**: 20,000 chiều
- **Vector normalization**: L2 norm = 1.000000 (perfect normalization)

### 3.2 Phân tích hiệu suất từng giai đoạn

#### a) Giai đoạn Tokenization
- **Input**: Raw text từ field "text" trong JSON
- **Output**: Array of words
- **Kết quả**: Tách thành công các từ từ văn bản gốc
- **Ví dụ**: "Beginners BBQ Class" → ["beginners", "bbq", "class"]

#### b) Giai đoạn Stop Words Removal  
- **Input**: Tokenized words
- **Output**: Filtered words (loại bỏ stop words)
- **Kết quả**: Giảm noise, tập trung vào từ có ý nghĩa
- **Hiệu quả**: Loại bỏ các từ như "the", "and", "in", "of"

#### c) Giai đoạn Vectorization (HashingTF + IDF)
- **HashingTF**: Chuyển đổi words thành term frequency vectors
- **IDF**: Tính toán inverse document frequency để giảm trọng số các từ xuất hiện nhiều
- **Kết quả**: Sparse vectors có 20,000 chiều
- **Ví dụ vector**: `(20000,[264,298,673,717...],[15.857,2.782,3.298...])`

### 3.3 Ý nghĩa của kết quả

#### a) Chất lượng vectorization
- **Sparse vectors**: Tiết kiệm bộ nhớ, chỉ lưu các giá trị khác 0
- **TF-IDF scores**: Phản ánh tầm quan trọng của từ trong document và corpus
- **20,000 features**: Đủ lớn để capture các patterns quan trọng

#### b) Hiệu suất xử lý
- **Pipeline fitting nhanh**: 1.64s cho 1000 documents
- **Transform hiệu quả**: 0.56s cho việc áp dụng pipeline
- **Scalable**: Có thể mở rộng cho datasets lớn hơn

#### c) Chất lượng dữ liệu đầu ra
- **Vocabulary size hợp lý**: 27,009 unique terms sau preprocessing
- **Feature vectors**: Sẵn sàng cho các algorithms machine learning
- **Structured output**: Dễ dàng sử dụng cho downstream tasks
- **L2 Normalized vectors**: Perfect normalization (L2 norm = 1.000000)
- **Cosine similarity**: Hiệu quả với normalized vectors (cosine = dot product)

#### d) Document Similarity Analysis
- **Reference document selection**: Tự động chọn document ID 0 làm reference
- **Similarity calculation time**: 1.13 giây cho 1000 documents
- **Top-5 similar documents**:
  1. Mercedes X-Class (0.130387) - Automotive content
  2. Italian Pasta Salad (0.085494) - Food/nutrition content  
  3. Brazilian Churrasco (0.080085) - Food/cooking content
  4. Museum tours (0.080035) - Educational content
  5. Fire simulator (0.076474) - Training/class content
- **Similarity range**: 0.076474 - 0.130387 (reasonable distribution)
- **Results quality**: Tìm được các documents liên quan về cooking, food, và training/class topics

### 3.4 Document Similarity Implementation Details

#### a) Cosine Similarity Algorithm
```scala
// UDF cho tính toán cosine similarity
val cosineSimilarity = udf((ref: Vector, vec: Vector) => {
  val refArray = ref.toArray
  val vecArray = vec.toArray
  // Với normalized vectors: cosine_similarity = dot_product
  refArray.zip(vecArray).map { case (a, b) => a * b }.sum
})
```

#### b) Implementation Strategy
- **Vector Normalization**: Sử dụng Normalizer với L2 norm trước khi tính similarity
- **Efficiency**: Với normalized vectors, cosine similarity = dot product (faster computation)
- **Memory Management**: Sử dụng cache() cho DataFrame để tránh re-computation
- **Scalability**: UDF hoạt động hiệu quả trên Spark distributed environment

#### c) Results Analysis
- **File output**: `results/lab17_similarity_results.txt`
- **Format**: Structured text với similarity scores, document IDs, và text previews
- **Top-K selection**: Configurable, hiện tại set = 5
- **Performance**: 1.13s để tính toán similarity cho 1000 documents

## 4. KHÓ KHĂN GẶP PHẢI VÀ CÁCH GIẢI QUYẾT (Difficulties and Solutions)

### 4.1 Vấn đề tương thích Java version
**Khó khăn**: 
- Spark 3.5.1 yêu cầu Java 11+ nhưng một số tính năng hoạt động tốt nhất với Java 17
- Serialization issues với một số ML algorithms (Word2Vec) trên Java 17

**Giải pháp**:
- Sử dụng OpenJDK 17 LTS như khuyến nghị
- Cấu hình JVM options phù hợp trong build.sbt
- Thay thế Word2Vec bằng HashingTF + IDF để tránh serialization conflicts

### 4.2 Vấn đề quản lý memory
**Khó khăn**:
- Spark driver memory mặc định có thể không đủ cho large datasets
- Hash collisions khi vocabulary size lớn hơn numFeatures của HashingTF

**Giải pháp**:
```scala
// Tăng driver memory khi chạy
sbt -J-Xmx4g "runMain com.lhson.spark.Lab17_NLPPipeline"

// Điều chỉnh numFeatures phù hợp
.setNumFeatures(20000) // Tăng từ 1000 lên 20000
```

### 4.3 Vấn đề với Windows environment
**Khó khăn**:
- Hadoop winutils.exe warning trên Windows
- Path separator issues ('\' vs '/')

**Giải pháp**:
- Bỏ qua winutils warning (chỉ ảnh hưởng performance, không ảnh hưởng functionality)
- Sử dụng relative paths và để Spark tự handle path conversion

### 4.4 Vấn đề performance optimization
**Khó khăn**:
- Cold start time của Spark khá lâu (~25s)
- Cần balance giữa accuracy và processing speed

**Giải pháp**:
- Sử dụng `.cache()` cho DataFrames được sử dụng nhiều lần
- Enable adaptive query execution trong Spark config
- Giới hạn số records (1000) cho môi trường development/testing

### 4.5 Vấn đề debugging và monitoring
**Khó khăn**:
- Khó theo dõi progress của các transformation stages
- Error messages không luôn rõ ràng

**Giải pháp**:
- Implement comprehensive logging system
- Sử dụng Spark UI để monitor job execution
- Thêm timing measurements cho từng stage

## 5. THAM KHẢO (References)

### 5.1 Tài liệu chính thức
1. **Apache Spark Official Documentation**
   - URL: https://spark.apache.org/docs/3.5.1/
   - Mục đích: API reference và best practices

2. **Spark MLlib Programming Guide**
   - URL: https://spark.apache.org/docs/3.5.1/ml-guide.html
   - Mục đích: Machine Learning pipeline design

3. **Scala Documentation**
   - URL: https://docs.scala-lang.org/
   - Mục đích: Scala language reference

### 5.2 Dataset và môi trường
4. **Common Crawl C4 Dataset**
   - Source: https://commoncrawl.org/
   - Mục đích: Large-scale text data cho NLP tasks

5. **SBT Reference Manual**
   - URL: https://www.scala-sbt.org/1.x/docs/
   - Mục đích: Build configuration và dependency management

### 5.3 Technical resources
6. **TF-IDF Algorithm Documentation**
   - Source: Apache Spark MLlib documentation
   - Mục đích: Understanding vectorization algorithms

7. **RegexTokenizer Implementation**
   - Source: Spark MLlib source code
   - Mục đích: Tokenization pattern design

### 5.4 Troubleshooting resources
8. **Java 17 + Spark Compatibility Guide**
   - Various Stack Overflow threads và GitHub issues
   - Mục đích: Giải quyết compatibility issues

## 6. MÔ HÌNH VÀ CÔNG CỤ SỬ DỤNG (Models and Tools Used)

### 6.1 Pre-built components (không sử dụng pre-trained models)
**Lưu ý**: Bài tập này không sử dụng pre-trained models mà xây dựng pipeline từ các building blocks cơ bản.

### 6.2 Spark MLlib Components
- **RegexTokenizer**: Built-in tokenizer của Spark MLlib
- **StopWordsRemover**: Sử dụng English stop words list mặc định
- **HashingTF**: Hashing-based Term Frequency implementation
- **IDF**: Inverse Document Frequency calculator

### 6.3 Configuration parameters
```scala
// Tokenizer config
.setPattern("\\W") // Split on non-word characters

// HashingTF config  
.setNumFeatures(20000) // 20K dimensional feature space

// IDF config
// Sử dụng default parameters (minDocFreq = 0)
```

### 6.4 No external AI models used
Project này không sử dụng:
- GPT/ChatGPT cho code generation
- Pre-trained word embeddings (Word2Vec, GloVe)
- External NLP APIs
- Cloud-based ML services

Tất cả code được viết thủ công dựa trên Spark MLlib documentation và best practices.

---

## Kết luận

### Tất cả tiêu chí đã hoàn thành (12/12 yêu cầu):

#### Core Requirements (8/8):
- **Read C4 dataset** into Spark DataFrame (Stage 2)
- **Implement Spark ML Pipeline** với 5-stage architecture
- **Use RegexTokenizer** cho tokenization với pattern `\\W`
- **Use StopWordsRemover** để loại bỏ English stop words
- **Use HashingTF and IDF** cho vectorization (20,000 features)
- **Fit pipeline and transform data** với caching optimization
- **Save results to file** (`results/lab17_pipeline_output.txt`)
- **Log the process** (`log/lab17_metrics.log`)

#### Extended Features (4/4):
- **Add limitDocuments variable** để customize document limit (1000 records)
- **Add detailed performance measurement** cho 8 processing stages
- **Add Normalizer layer** để normalize vectors với L2 norm
- **Search and display top K similar documents** với cosine similarity

### Performance Summary:
- **Total execution time**: 25.04 giây
- **Overall throughput**: 40 records/giây
- **Vocabulary generated**: 27,009 unique terms
- **Feature dimensions**: 20,000
- **Vector normalization**: L2 norm = 1.000000 (perfect)
- **Similarity analysis**: 1.13s cho 1000 documents

### Files Generated:
1. **Performance log**: `log/lab17_metrics.log`
2. **Pipeline results**: `results/lab17_pipeline_output.txt`
3. **Similarity analysis**: `results/lab17_similarity_results.txt`

Pipeline hoạt động ổn định, hiệu quả, và tạo ra feature vectors chất lượng cao sẵn sàng cho các downstream machine learning tasks bao gồm document classification, clustering, và similarity search.

## 7. CÁC THỰC NGHIỆM MỞ RỘNG (Extended Experiments)

### 7.1 Thực nghiệm 1: So sánh Tokenizers
**Mục tiêu**: So sánh hiệu suất giữa RegexTokenizer và basic Tokenizer

**Thực hiện**:
- Comment RegexTokenizer, uncomment basic Tokenizer
- Chạy pipeline và so sánh kết quả

**Kết quả**:
- **RegexTokenizer** (gốc): 27,009 từ unique, fitting 1.64s, transform 0.56s
- **Basic Tokenizer**: 46,838 từ unique, fitting 1.85s, transform 0.64s
- **Phân tích**: Basic Tokenizer tạo ra nhiều từ hơn (46,838 vs 27,009) vì không loại bỏ dấu câu như RegexTokenizer
- **Ưu điểm Basic Tokenizer**: Đơn giản, nhanh chóng
- **Ưu điểm RegexTokenizer**: Linh hoạt hơn với regex patterns, tạo ra vocabulary sạch hơn

### 7.2 Thực nghiệm 2: Ảnh hưởng kích thước Feature Vector
**Mục tiêu**: Kiểm tra tác động của việc giảm numFeatures từ 20,000 xuống 1,000

**Thực hiện**:
- Thay đổi HashingTF setNumFeatures từ 20000 thành 1000
- Đo lường thời gian và chất lượng vector

**Kết quả**:
- **20,000 features** (gốc): Feature vector size 20,000, fitting 1.64s, transform 0.56s
- **1,000 features**: Feature vector size 1,000, fitting 2.21s, transform 0.76s
- **Hash collisions**: Tăng đáng kể với vocabulary 46,838 từ nhưng chỉ 1,000 hash buckets
- **Phân tích**: Giảm features làm tăng hash collisions, có thể mất thông tin nhưng tiết kiệm bộ nhớ
- **Trade-off**: Memory efficiency vs Information preservation

### 7.3 Thực nghiệm 3: Mở rộng Pipeline với Classification
**Mục tiêu**: Thêm LogisticRegression để chuyển từ feature extraction sang machine learning task

**Thực hiện**:
- Tạo synthetic labels dựa trên độ dài văn bản (>500 chars = 1, <=500 chars = 0)
- Thêm LogisticRegression vào pipeline
- Đánh giá accuracy và probability predictions

**Kết quả**:
- **Model Accuracy**: 98.20%
- **Pipeline fitting time**: 4.07 giây (tăng từ 2.21s do training LogisticRegression)
- **Transform time**: 0.62 giây
- **Schema mới**: Thêm columns label, rawPrediction, probability, prediction
- **Phân tích**: Model đạt accuracy rất cao (98.20%) với task classification đơn giản
- **Insight**: TF-IDF features rất hiệu quả cho text classification tasks

### 7.4 Thực nghiệm 4: Word2Vec Implementation (THẤT BẠI)
**Mục tiêu**: Thay thế HashingTF + IDF bằng Word2Vec để tạo word embeddings

**Thực hiện**:
- Comment out HashingTF và IDF stages
- Thêm Word2Vec với 100-dimensional vectors
- Cấu hình minCount=1, maxIter=5

**Kết quả**:
- **Lỗi**: Java 17 + Spark 3.5.1 Kryo serialization incompatibility
- **Error type**: `IllegalArgumentException: Unable to create serializer for SerializedLambda`
- **Root cause**: Java module system không cho phép access vào java.lang.invoke
- **Status**: BLOCKED - Cannot complete with current environment

**Giải pháp thử nghiệm**:
- Downgrade xuống Java 11 (không khả thi trong môi trường hiện tại)
- Sử dụng Java serializer thay vì Kryo (performance impact)
- Stick with HashingTF + IDF approach (recommended)

### 7.5 Thực nghiệm mới: Document Similarity Analysis
**Mục tiêu**: Implement tính năng tìm kiếm tài liệu tương tự với cosine similarity

**Thực hiện**:
- Thêm Normalizer stage để chuẩn hóa TF-IDF vectors
- Tạo cosine similarity UDF cho vector comparison
- Select reference document (ID: 0) và tính similarity với tất cả documents khác
- Sắp xếp và lấy top-5 documents có similarity cao nhất

**Kết quả**:
- **Thời gian tính toán**: 1.13 giây cho 1000 documents
- **Reference document**: "Beginners BBQ Class" (food/cooking content)
- **Top similar documents**: Mercedes X-Class (0.130), Italian Pasta (0.085), Brazilian Churrasco (0.080), Museum tours (0.080), Fire simulator (0.076)
- **Phân tích chất lượng**: Algorithm tìm thấy các documents liên quan về food/cooking và class/training topics
- **Files generated**: `results/lab17_similarity_results.txt`

### 7.6 Tổng kết các thực nghiệm

| Thực nghiệm | Trạng thái | Thời gian Fitting | Vocabulary Size | Feature Vector Size | Accuracy |
|-------------|------------|-------------------|-----------------|---------------------|-----------|
| Baseline (RegexTokenizer + HashingTF 20K) | HOANTHANH | 1.64s | 27,009 | 20,000 | N/A |
| Basic Tokenizer + HashingTF 20K | HOANTHANH | 1.85s | 46,838 | 20,000 | N/A |
| Basic Tokenizer + HashingTF 1K | HOANTHANH | 2.21s | 46,838 | 1,000 | N/A |
| Basic Tokenizer + HashingTF 1K + LR | HOANTHANH | 4.07s | 46,838 | 1,000 | 98.20% |
| RegexTokenizer + HashingTF 20K + Normalizer + Similarity | HOANTHANH | 1.61s | 27,009 | 20,000 | N/A (similarity) |
| Basic Tokenizer + Word2Vec + LR | THATBAI | Failed | N/A | N/A | N/A |

**Kết luận thực nghiệm**:
1. **Tokenizer choice** ảnh hưởng đáng kể đến vocabulary size và processing time
2. **Feature vector size** là trade-off giữa memory và information preservation  
3. **Classification extension** hoạt động rất tốt với TF-IDF features (98.20% accuracy)
4. **Word2Vec** bị block bởi Java 17 compatibility issues với Spark MLlib
5. **Recommended configuration**: RegexTokenizer + HashingTF (20K) + IDF cho balance tốt nhất

---

---

**Ngày hoàn thành**: 02/10/2025  
**Spark Version**: 3.5.1  
**Java Version**: OpenJDK 17 LTS
**Comprehensive NLP Pipeline**: 8 stages với document similarity analysis