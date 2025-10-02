# Báo cáo Bài tập Lab17 - Spark NLP Pipeline

## 1. CÁC BƯỚC THỰC HIỆN

### 1.1 Thiết lập môi trường phát triển
- **Ngôn ngữ lập trình**: Scala 2.12.x
- **Framework**: Apache Spark 3.5.1 với MLlib
- **Build tool**: SBT (Simple Build Tool) 1.11.6
- **Java Runtime**: OpenJDK 17 LTS
- **Dữ liệu**: C4 Common Crawl dataset (30K records)
- **Document limit**: Configurable với `limitDocuments` variable (default: 30000 records)
- **Pipeline approaches**: Dual implementation (HashingTF vs CountVectorizer)
- **Performance monitoring**: Detailed timing measurements cho từng processing stage
- **Vector normalization**: L2 (Euclidean) normalization
- **Document similarity**: Cosine similarity với top-K document search
- **Demo features**: 5 instructor-guided demonstrations
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
Pipeline được thiết kế theo mô hình ETL (Extract-Transform-Load) với 8 giai đoạn tuần tự và 5 demo features:

1. **Stage 1**: Khởi tạo Spark Session với performance monitoring và UI configuration
2. **Stage 2**: Đọc dữ liệu C4 dataset với limitDocuments = 30000 và materialization
3. **Stage 3**: Tạo dual ML Pipeline components:
   - Pipeline 1: RegexTokenizer → StopWordsRemover → HashingTF → IDF → Normalizer
   - Pipeline 2: RegexTokenizer → StopWordsRemover → CountVectorizer → IDF → Normalizer
4. **Stage 4**: Training CountVectorizer Pipeline với vocabulary learning từ 30K corpus
5. **Stage 5**: Applying Pipeline Transformation với DataFrame caching và sparse vector analysis
6. **Stage 6**: Analyzing Vocabulary Statistics với CountVectorizer vocabulary (187K → 20K terms)
7. **Stage 7**: Collecting Results và Writing detailed output files với normalization demos
8. **Stage 8**: Document Similarity Analysis với cosine similarity và top-10 selection

#### 5 Demo Features theo yêu cầu:
1. **Sparse vector representation**: Analysis of document vectors với sparsity 99.7%+
2. **Normalization of count vectors**: L2 normalization demo với before/after comparison
3. **CountVectorizer pipeline**: tokenizer → countVectorizer → idf implementation
4. **30K-doc corpus demo**: Large-scale processing với 30,000 documents
5. **Top-10 similarity search**: Cosine similarity với document matching

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

#### b) Đọc dữ liệu C4 Dataset với limitDocuments = 30000
```scala
// Configurable document limit for large-scale demo
val limitDocuments = 30000  // 30K documents for comprehensive demo
val df = spark.read.json("data/c4-train.00000-of-01024-30K.json.gz")
  .limit(limitDocuments)

// Force DataFrame materialization để đo thời gian đọc chính xác
val recordCount = df.count()
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

#### e) Dual vectorization approaches: HashingTF và CountVectorizer

**HashingTF approach:**
```scala
val hashingTF = new HashingTF()
  .setInputCol("filtered_words")
  .setOutputCol("tf_features")
  .setNumFeatures(20000)

val idf = new IDF()
  .setInputCol("tf_features")
  .setOutputCol("tf_features_idf")
```

**CountVectorizer approach:**
```scala
val countVectorizer = new CountVectorizer()
  .setInputCol("filtered_words")
  .setOutputCol("count_features")
  .setVocabSize(20000)
  .setMinDF(2.0)
  .setMaxDF(0.95)

val idfCount = new IDF()
  .setInputCol("count_features")
  .setOutputCol("count_tfidf_features")
```

#### f) Normalizer để chuẩn hóa vectors
```scala
val normalizer = new Normalizer()
  .setInputCol("raw_features")
  .setOutputCol("features")
  .setP(2.0) // L2 (Euclidean) normalization
```

#### g) Tạo và thực thi Dual Pipeline
```scala
// Pipeline 1: HashingTF approach
val hashingPipeline = new Pipeline()
  .setStages(Array(tokenizer, stopWordsRemover, hashingTF, idf, normalizer))

// Pipeline 2: CountVectorizer approach
val countPipeline = new Pipeline()
  .setStages(Array(tokenizer, stopWordsRemover, countVectorizer, idfCount, normalizerCount))

// Sử dụng CountVectorizer pipeline cho demo
val pipelineModel = countPipeline.fit(df)
val transformedDF = pipelineModel.transform(df).cache()
```

#### h) Document Similarity Analysis (30K Corpus)
```scala
// Cosine similarity UDF cho CountVectorizer normalized vectors
val cosineSimilarityUDF = udf((vector: Vector) => {
  val refArray = referenceVector.toArray
  val vecArray = vector.toArray
  // Với L2-normalized vectors: cosine_similarity = dot_product
  refArray.zip(vecArray).map { case (a, b) => a * b }.sum
})

// Tìm top 10 documents tương tự nhất trong 30K corpus
val documentsWithSimilarity = documentsWithIndex
  .filter(col("doc_id") =!= referenceDocIndex)
  .withColumn("cosine_similarity", cosineSimilarityUDF(col("count_normalized_features")))
  .orderBy(desc("cosine_similarity"))
  .limit(10)
```

## 2. CÁCH CHẠY CODE VÀ GHI LOG KẾT QUẢ

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
=== COMPREHENSIVE NLP PIPELINE PERFORMANCE REPORT ===
Records processed: 30000
Pipeline Training: 16.240 seconds
Data Transformation: 6.900 seconds
Vocabulary size (unique terms): 187214
CountVectorizer kept: 20000 terms (compression ratio: 10.7%)
Feature vector dimensions: 20000
Total Processing Time: 66.360 seconds
Overall Throughput: 452 records/second
```

#### Results file (lab17_pipeline_output.txt):
```
--- NLP Pipeline Output (First 20 results) ---
================================================================================
Record #1
Original Text: Beginners BBQ Class Taking Place in Missoula!...
Tokenized Words (131 total): beginners, bbq, class, taking, place...
Filtered Words (73 total): beginners, bbq, class, taking, place, missoula...
Raw TF-IDF Vector: 20000 dimensions, 59 non-zero, L2 norm = 42.646810
Normalized Vector: 20000 dimensions, 59 non-zero, L2 norm = 1.000000
================================================================================
```

## 3. GIẢI THÍCH KẾT QUẢ ĐẠT ĐƯỢC

### 3.1 Thống kê tổng quan (30K Corpus Demo)
- **Số lượng records xử lý**: 30,000 documents từ C4 dataset
- **Thời gian khởi tạo Spark**: 10.55 giây
- **Thời gian đọc dữ liệu**: 3.62 giây
- **Thời gian tạo dual pipeline**: 0.08 giây
- **Thời gian fitting CountVectorizer pipeline**: 16.24 giây
- **Thời gian transform dữ liệu**: 6.90 giây
- **Thời gian phân tích vocabulary**: 1.11 giây
- **Thời gian similarity analysis**: 17.24 giây
- **Tổng thời gian xử lý**: 66.36 giây
- **Overall throughput**: 452 records/giây
- **Training throughput**: 1847 records/giây
- **Transform throughput**: 4347 records/giây
- **Kích thước từ vựng thực tế**: 187,214 từ unique trong corpus
- **Kích thước từ vựng CountVectorizer**: 20,000 từ (compression ratio: 10.7%)
- **Kích thước feature vector**: 20,000 chiều
- **Vector sparsity**: 99.7%+ (sparse representation)
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

#### c) Giai đoạn Vectorization (CountVectorizer + IDF)
- **CountVectorizer**: Xây dựng vocabulary thực tế từ 30K corpus (187,214 → 20,000 terms)
- **IDF**: Tính toán inverse document frequency cho count-based vectors
- **Kết quả**: Sparse vectors có 20,000 chiều với sparsity 99.7%+
- **Ví dụ vector**: Size=20000, NonZeros=59, Sparsity=99.71%
- **Demo features**: Sparse vector representation analysis và normalization effects

### 3.3 Ý nghĩa của kết quả

#### a) Chất lượng vectorization
- **Sparse vectors**: Tiết kiệm bộ nhớ, chỉ lưu các giá trị khác 0
- **TF-IDF scores**: Phản ánh tầm quan trọng của từ trong document và corpus
- **20,000 features**: Đủ lớn để capture các patterns quan trọng

#### b) Hiệu suất xử lý
- **Pipeline fitting nhanh**: 1.70s cho 1000 documents
- **Transform hiệu quả**: 0.63s cho việc áp dụng pipeline
- **Scalable**: Có thể mở rộng cho datasets lớn hơn

#### c) Chất lượng dữ liệu đầu ra
- **Vocabulary size hợp lý**: 27,009 unique terms sau preprocessing
- **Feature vectors**: Sẵn sàng cho các algorithms machine learning
- **Structured output**: Dễ dàng sử dụng cho downstream tasks
- **L2 Normalized vectors**: Perfect normalization (L2 norm = 1.000000)
- **Cosine similarity**: Hiệu quả với normalized vectors (cosine = dot product)

#### d) Document Similarity Analysis (30K Corpus)
- **Reference document selection**: Document ID 42 (St. Boniface, religious content)
- **Similarity calculation time**: 17.24 giây cho 30,000 documents
- **Top-10 similar documents**:
  1. Document ID: 12083 (Similarity: 0.180184) - Dominican Bishop Anthony Fisher
  2. Document ID: 10888 (Similarity: 0.178638) - Haughey|Gregory political content
  3. Document ID: 12465 (Similarity: 0.166638) - Ancient trees và Cadzow Castle
  4. Document ID: 27102 (Similarity: 0.164087) - Diocese of Arlington seminary
  5. Document ID: 9205 (Similarity: 0.158835) - Bishop J. Jon Bruno legal case
  6. Document ID: 18111 (Similarity: 0.145258) - St. Patrick's day recipe
  7. Document ID: 4400 (Similarity: 0.138954) - Coptic Pope in Cairo/Alexandria
  8. Document ID: 23889 (Similarity: 0.135815) - Elizabeth Bishop poet
  9. Document ID: 16194 (Similarity: 0.131704) - St Paul's Jubilee copes
  10. Document ID: 27715 (Similarity: 0.125407) - Bishop Ranch business location
- **Similarity range**: 0.125407 - 0.180184
- **Results quality**: Tìm được các documents liên quan về religious topics, bishops, và church content

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
- **Performance**: 1.10s để tính toán similarity cho 1000 documents

## 4. KHÓ KHĂN GẶP PHẢI VÀ CÁCH GIẢI QUYẾT

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
- Cold start time của Spark khá lâu (~27s với initialization 10s)
- Cần balance giữa accuracy và processing speed
- Memory và network resource management

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

## 5. MÔ HÌNH VÀ CÔNG CỤ SỬ DỤNG

### 5.1 Pre-built components (không sử dụng pre-trained models)
**Lưu ý**: Bài tập này không sử dụng pre-trained models mà xây dựng pipeline từ các building blocks cơ bản.

### 5.2 Spark MLlib Components
- **RegexTokenizer**: Built-in tokenizer của Spark MLlib
- **StopWordsRemover**: Sử dụng English stop words list mặc định
- **HashingTF**: Hashing-based Term Frequency implementation
- **IDF**: Inverse Document Frequency calculator

### 5.3 Configuration parameters
```scala
// Tokenizer config
.setPattern("\\W") // Split on non-word characters

// HashingTF config  
.setNumFeatures(20000) // 20K dimensional feature space

// IDF config
// Sử dụng default parameters (minDocFreq = 0)
```

## 6. CÁC THỰC NGHIỆM MỞ RỘNG

### 6.1 Thực nghiệm 1: So sánh Tokenizers
**Mục tiêu**: So sánh hiệu suất giữa RegexTokenizer và basic Tokenizer

**Thực hiện**:
- Comment RegexTokenizer, uncomment basic Tokenizer
- Chạy pipeline và so sánh kết quả

**Kết quả**:
- **RegexTokenizer** (gốc): 27,009 từ unique, fitting 1.70s, transform 0.63s
- **Basic Tokenizer**: 46,838 từ unique, fitting 1.85s, transform 0.64s
- **Phân tích**: Basic Tokenizer tạo ra nhiều từ hơn (46,838 vs 27,009) vì không loại bỏ dấu câu như RegexTokenizer
- **Ưu điểm Basic Tokenizer**: Đơn giản, nhanh chóng
- **Ưu điểm RegexTokenizer**: Linh hoạt hơn với regex patterns, tạo ra vocabulary sạch hơn

### 6.2 Thực nghiệm 2: Ảnh hưởng kích thước Feature Vector
**Mục tiêu**: Kiểm tra tác động của việc giảm numFeatures từ 20,000 xuống 1,000

**Thực hiện**:
- Thay đổi HashingTF setNumFeatures từ 20000 thành 1000
- Đo lường thời gian và chất lượng vector

**Kết quả**:
- **20,000 features** (gốc): Feature vector size 20,000, fitting 1.70s, transform 0.63s
- **1,000 features**: Feature vector size 1,000, fitting 2.21s, transform 0.76s
- **Hash collisions**: Tăng đáng kể với vocabulary 46,838 từ nhưng chỉ 1,000 hash buckets
- **Phân tích**: Giảm features làm tăng hash collisions, có thể mất thông tin nhưng tiết kiệm bộ nhớ
- **Trade-off**: Memory efficiency vs Information preservation

### 6.3 Thực nghiệm 3: Mở rộng Pipeline với Classification
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

### 6.4 Thực nghiệm 4: Word2Vec Implementation (THẤT BẠI)
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

### 6.5 Thực nghiệm 5: Document Similarity Analysis
**Mục tiêu**: Implement tính năng tìm kiếm tài liệu tương tự với cosine similarity

**Thực hiện**:
- Thêm Normalizer stage để chuẩn hóa TF-IDF vectors
- Tạo cosine similarity UDF cho vector comparison
- Select reference document (ID: 0) và tính similarity với tất cả documents khác
- Sắp xếp và lấy top-5 documents có similarity cao nhất

**Kết quả**:
- **Thời gian tính toán**: 1.10 giây cho 1000 documents
- **Reference document**: "Beginners BBQ Class" (food/cooking content)
- **Top similar documents**: Mercedes X-Class (0.130), Italian Pasta (0.085), Brazilian Churrasco (0.080), Museum tours (0.080), Fire simulator (0.076)
- **Phân tích chất lượng**: Algorithm tìm thấy các documents liên quan về food/cooking và class/training topics
- **Files generated**: `results/lab17_similarity_results.txt`

### 6.6 Tổng kết các thực nghiệm

| Thực nghiệm | Trạng thái | Documents | Thời gian Fitting | Vocabulary Size | Feature Vector Size | Demo Features |
|-------------|------------|-----------|-------------------|-----------------|---------------------|---------------|
| CountVectorizer 30K + All Demos | Success | 30,000 | 16.24s | 187,214 → 20,000 | 20,000 | 5 demos |
| Baseline (RegexTokenizer + HashingTF) | Success | 1,000 | 1.70s | 27,009 | 20,000 | Basic |
| Basic Tokenizer + HashingTF | Success | 1,000 | 1.85s | 46,838 | 20,000 | Basic |
| CountVectorizer + Classification | Success | 1,000 | 4.07s | 46,838 | 1,000 | Classification |
| CountVectorizer + Similarity Analysis | Success | 30,000 | 16.24s | 187,214 → 20,000 | 20,000 | Top-10 search |
| Word2Vec implementation | Fail | N/A | Failed | N/A | N/A | Blocked |

**Kết luận thực nghiệm**:
1. **CountVectorizer approach** vượt trội so với HashingTF cho vocabulary learning
2. **30K corpus processing** thể hiện scalability của Spark MLlib
3. **Sparse vector representation** rất hiệu quả cho text data (99.7%+ sparsity)
4. **L2 normalization** đảm bảo cosine similarity calculation chính xác
5. **Document similarity** hoạt động tốt với religious/church content matching

---