package com.lhson.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, IDF, Normalizer, RegexTokenizer, StopWordsRemover, Tokenizer, Word2Vec}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.Vector
import java.io.{File, PrintWriter}
import org.apache.log4j.{Logger, Level}
import java.text.SimpleDateFormat
import java.util.Date

object Lab17_NLPPipeline {
  def main(args: Array[String]): Unit = {
    // =============================================================================
    // STAGE 1: SPARK SESSION INITIALIZATION
    // =============================================================================
    println("=== STAGE 1: Initializing Spark Session ===")
    val sparkInitStartTime = System.nanoTime()
    
    val spark = SparkSession.builder()
      .appName("NLP Pipeline Example")
      .master("local[*]")
      .config("spark.ui.port", "4040")
      .config("spark.serializer", "org.apache.spark.serializer.JavaSerializer") // Use Java serializer for Word2Vec compatibility
      .getOrCreate()
    
    val sparkInitDuration = (System.nanoTime() - sparkInitStartTime) / 1e9d
    println(f"Spark Session initialization completed in $sparkInitDuration%.2f seconds")

    import spark.implicits._
    import org.apache.spark.sql.functions._

    // Set log level to INFO for our application
    spark.sparkContext.setLogLevel("ERROR")

    // Custom logging configuration
    val rootLogger = Logger.getRootLogger
    rootLogger.setLevel(Level.INFO)

    println("Spark Session created successfully.")
    println(s"Spark UI available at http://localhost:4040")
    println("Pausing for 10 seconds to allow you to open the Spark UI...")
    Thread.sleep(10000)

    // =============================================================================
    // STAGE 2: DATA READING AND LOADING
    // =============================================================================
    println("\n=== STAGE 2: Reading C4 Dataset ===")
    val dataReadStartTime = System.nanoTime()
    
    // REQUIREMENT: Add limitDocuments variable to customize the document limit
    // Configurable variable để giới hạn số lượng documents xử lý
    // Giúp kiểm soát memory usage và thời gian xử lý cho testing/development
    val limitDocuments = 1000
    
    // REQUIREMENT: Read the C4 dataset into a Spark DataFrame
    // Đọc C4 Common Crawl dataset (compressed JSON format) vào Spark DataFrame
    // C4 là large-scale text dataset được sử dụng rộng rãi cho NLP tasks
    val df = spark.read.json("data/c4-train.00000-of-01024-30K.json.gz")
      .limit(limitDocuments) // Áp dụng limit để kiểm soát dataset size
    
    // Force DataFrame materialization to measure actual read time
    val recordCount = df.count()
    val dataReadDuration = (System.nanoTime() - dataReadStartTime) / 1e9d
    
    println(f"Successfully read $recordCount records in $dataReadDuration%.2f seconds")
    println(f"Data reading throughput: ${recordCount / dataReadDuration}%.0f records/second")
    df.printSchema()
    println("Sample of initial DataFrame:")
    df.show(5, truncate = false)

    // =============================================================================
    // STAGE 3: PIPELINE COMPONENTS CREATION
    // =============================================================================
    println("\n=== STAGE 3: Creating ML Pipeline Components ===")
    val pipelineCreateStartTime = System.nanoTime()

    // REQUIREMENT: Use RegexTokenizer or Tokenizer for tokenization
    // RegexTokenizer chia văn bản thành các từ riêng lẻ bằng regex pattern
    // Pattern "\\W" = split on non-word characters (spaces, punctuation, etc.)
    val tokenizerCreateStart = System.nanoTime()
    val tokenizer = new RegexTokenizer()
      .setInputCol("text")      // Input: raw text từ C4 dataset
      .setOutputCol("words")    // Output: array of tokenized words
      .setPattern("\\W")        // Regex: split trên non-word characters
    val tokenizerCreateDuration = (System.nanoTime() - tokenizerCreateStart) / 1e6d
    println(f"RegexTokenizer created in $tokenizerCreateDuration%.2f ms")

    // REQUIREMENT: Use StopWordsRemover to remove stop words
    // Loại bỏ các stop words (the, and, in, of, etc.) không mang nhiều thông tin
    // Giúp giảm noise và tập trung vào các từ có ý nghĩa trong text analysis
    val stopWordsCreateStart = System.nanoTime()
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("words")           // Input: tokenized words từ RegexTokenizer
      .setOutputCol("filtered_words") // Output: words sau khi loại bỏ stop words
      // Sử dụng English stop words list có sẵn trong Spark MLlib
    val stopWordsCreateDuration = (System.nanoTime() - stopWordsCreateStart) / 1e6d
    println(f"StopWordsRemover created in $stopWordsCreateDuration%.2f ms")

    // REQUIREMENT: Use HashingTF and IDF for vectorization
    // HashingTF: Chuyển đổi words thành numerical feature vectors bằng hashing
    // Sử dụng hash function để map words vào fixed-size feature space (20K dimensions)
    val hashingTFCreateStart = System.nanoTime()
    val hashingTF = new HashingTF()
      .setInputCol("filtered_words")  // Input: filtered words (không có stop words)
      .setOutputCol("tf_features")    // Output: Term Frequency vectors
      .setNumFeatures(20000)          // Hash table size: 20K features (balance memory vs accuracy)
    val hashingTFCreateDuration = (System.nanoTime() - hashingTFCreateStart) / 1e6d
    println(f"HashingTF created in $hashingTFCreateDuration%.2f ms")

    // IDF: Inverse Document Frequency - giảm trọng số của words xuất hiện nhiều
    // Tăng trọng số cho rare/distinctive words, giảm trọng số cho common words
    val idfCreateStart = System.nanoTime()
    val idf = new IDF()
      .setInputCol("tf_features")     // Input: TF vectors từ HashingTF
      .setOutputCol("raw_features")   // Output: TF-IDF vectors (chưa normalized)
      // IDF formula: log(numDocs / (docFreq + 1)) + 1
    val idfCreateDuration = (System.nanoTime() - idfCreateStart) / 1e6d
    println(f"IDF created in $idfCreateDuration%.2f ms")

    // REQUIREMENT: Add Normalizer layer to normalize vectors
    // L2 Normalization: Chuẩn hóa vectors để có unit length (L2 norm = 1)
    // Quan trọng cho cosine similarity calculation và machine learning algorithms
    val normalizerCreateStart = System.nanoTime()
    val normalizer = new Normalizer()
      .setInputCol("raw_features")    // Input: TF-IDF vectors chưa chuẩn hóa
      .setOutputCol("features")       // Output: L2-normalized vectors (final features)
      .setP(2.0)                     // L2 norm (Euclidean): sqrt(sum(x_i^2)) = 1
      // Normalization giúp vectors comparable và improve ML algorithm performance
    val normalizerCreateDuration = (System.nanoTime() - normalizerCreateStart) / 1e6d
    println(f"Normalizer (L2) created in $normalizerCreateDuration%.2f ms")

    // REQUIREMENT: Implement a Spark ML Pipeline
    // Tạo ML Pipeline kết hợp 5 transformation stages theo thứ tự:
    // Text → Words → FilteredWords → TF → TF-IDF → NormalizedTF-IDF
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, hashingTF, idf, normalizer))
    
    val pipelineCreateDuration = (System.nanoTime() - pipelineCreateStartTime) / 1e9d
    println(f"Complete pipeline (5 stages) created in $pipelineCreateDuration%.4f seconds")
    println("Pipeline stages: RegexTokenizer -> StopWordsRemover -> HashingTF -> IDF -> Normalizer")

    // =============================================================================
    // STAGE 4: PIPELINE TRAINING (FITTING)
    // =============================================================================
    println("\n=== STAGE 4: Training NLP Pipeline ===")
    val fitStartTime = System.nanoTime()
    
    // REQUIREMENT: Fit the pipeline and transform the data (Part 1: Fitting)
    // Pipeline fitting: học parameters từ training data (IDF weights, vocabulary, etc.)
    // Chỉ IDF stage cần fitting để tính document frequencies
    val pipelineModel = pipeline.fit(df)
    
    // REQUIREMENT: Add detailed performance measurement for stages
    // Đo lường chi tiết thời gian fitting và throughput
    val fitDuration = (System.nanoTime() - fitStartTime) / 1e9d
    val fittingThroughput = recordCount / fitDuration
    println(f"Pipeline training completed in $fitDuration%.2f seconds")
    println(f"Training throughput: ${fittingThroughput}%.0f records/second")
    println(f"Average training time per record: ${(fitDuration * 1000) / recordCount}%.2f ms/record")

    // =============================================================================
    // STAGE 5: DATA TRANSFORMATION (INFERENCE)
    // =============================================================================
    println("\n=== STAGE 5: Applying Pipeline Transformation ===")
    val transformStartTime = System.nanoTime()
    
    // REQUIREMENT: Fit the pipeline and transform the data (Part 2: Transform)
    // Áp dụng trained pipeline để transform raw text thành normalized TF-IDF vectors
    val transformedDF = pipelineModel.transform(df)
    
    val cacheStartTime = System.nanoTime()
    transformedDF.cache() // Cache the result for efficiency
    val transformCount = transformedDF.count() // Force an action to trigger the transformation
    val cacheDuration = (System.nanoTime() - cacheStartTime) / 1e9d
    
    val transformDuration = (System.nanoTime() - transformStartTime) / 1e9d
    val transformThroughput = transformCount / transformDuration
    
    println(f"Data transformation completed in $transformDuration%.2f seconds")
    println(f"Caching and materialization took $cacheDuration%.2f seconds")
    println(f"Transform throughput: ${transformThroughput}%.0f records/second")
    println(f"Average transform time per record: ${(transformDuration * 1000) / transformCount}%.2f ms/record")
    println(f"Total records processed: $transformCount")

    // =============================================================================
    // STAGE 6: VOCABULARY ANALYSIS
    // =============================================================================
    println("\n=== STAGE 6: Analyzing Vocabulary Statistics ===")
    val vocabAnalysisStartTime = System.nanoTime()
    
    // Calculate actual vocabulary size after tokenization and stop word removal
    val actualVocabSize = transformedDF
      .select(explode($"filtered_words").as("word"))
      .filter(length($"word") > 1) // Filter out single-character tokens
      .distinct()
      .count()
    
    val vocabAnalysisDuration = (System.nanoTime() - vocabAnalysisStartTime) / 1e9d
    
    println(f"Vocabulary analysis completed in $vocabAnalysisDuration%.2f seconds")
    println(f"Unique vocabulary size: $actualVocabSize terms")
    println(f"Vocabulary density: ${actualVocabSize.toDouble / recordCount}%.2f unique terms per document")

    // --- Show and Save Results ---
    println("\nSample of transformed data:")
    transformedDF.select("text", "words", "filtered_words", "raw_features", "features").show(5, truncate = 50)

    println("\nDataFrame Schema After Transformation:")
    transformedDF.printSchema()

    println("\nFeature vector information:")
    val featuresInfo = transformedDF.select("raw_features", "features").first()
    val rawFeaturesVector = featuresInfo.getAs[org.apache.spark.ml.linalg.Vector]("raw_features")
    val normalizedFeaturesVector = featuresInfo.getAs[org.apache.spark.ml.linalg.Vector]("features")
    
    println(s"--> Raw TF-IDF vector size: ${rawFeaturesVector.size}")
    println(s"--> Raw TF-IDF vector type: ${rawFeaturesVector.getClass.getSimpleName}")
    println(s"--> Normalized vector size: ${normalizedFeaturesVector.size}")
    println(s"--> Normalized vector type: ${normalizedFeaturesVector.getClass.getSimpleName}")
    
    // Calculate norms for demonstration
    val rawNorm = math.sqrt(rawFeaturesVector.toArray.map(x => x * x).sum)
    val normalizedNorm = math.sqrt(normalizedFeaturesVector.toArray.map(x => x * x).sum)
    println(f"--> Raw vector L2 norm: $rawNorm%.6f")
    println(f"--> Normalized vector L2 norm: $normalizedNorm%.6f (should be approximately 1.0)")

    // =============================================================================
    // STAGE 7: RESULTS COLLECTION AND FILE WRITING
    // =============================================================================
    println("\n=== STAGE 7: Collecting Results and Writing Files ===")
    val resultsCollectionStartTime = System.nanoTime()
    
    // REQUIREMENT: Save the results to a file
    // Collect sample results để write vào output file
    val n_results = 20
    val results = transformedDF.select("text", "words", "filtered_words", "raw_features", "features").take(n_results)
    
    val resultsCollectionDuration = (System.nanoTime() - resultsCollectionStartTime) / 1e9d
    println(f"Results collection completed in $resultsCollectionDuration%.2f seconds")
    println(f"Collected $n_results sample records for output")

    // REQUIREMENT: Log the process
    // Ghi detailed performance metrics và statistics vào log file
    val logWriteStartTime = System.nanoTime()
    val log_path = "log/lab17_metrics.log"
    new File(log_path).getParentFile.mkdirs() // Tạo thư mục nếu chưa tồn tại
    val logWriter = new PrintWriter(new File(log_path))
    try {
      val totalProcessingTime = sparkInitDuration + dataReadDuration + pipelineCreateDuration + fitDuration + transformDuration + vocabAnalysisDuration + resultsCollectionDuration
      
      logWriter.println("=== COMPREHENSIVE NLP PIPELINE PERFORMANCE REPORT ===")
      logWriter.println(f"Generated on: ${new java.util.Date()}")
      logWriter.println(f"Records processed: $recordCount")
      logWriter.println("\n--- DETAILED STAGE-BY-STAGE PERFORMANCE ---")
      logWriter.println(f"Stage 1 - Spark Initialization: $sparkInitDuration%.3f seconds (${(sparkInitDuration/totalProcessingTime*100)}%.1f%%)")
      logWriter.println(f"Stage 2 - Data Reading: $dataReadDuration%.3f seconds (${(dataReadDuration/totalProcessingTime*100)}%.1f%%)")
      logWriter.println(f"Stage 3 - Pipeline Creation: $pipelineCreateDuration%.3f seconds (${(pipelineCreateDuration/totalProcessingTime*100)}%.1f%%)")
      logWriter.println(f"Stage 4 - Pipeline Training: $fitDuration%.3f seconds (${(fitDuration/totalProcessingTime*100)}%.1f%%)")
      logWriter.println(f"Stage 5 - Data Transformation: $transformDuration%.3f seconds (${(transformDuration/totalProcessingTime*100)}%.1f%%)")
      logWriter.println(f"Stage 6 - Vocabulary Analysis: $vocabAnalysisDuration%.3f seconds (${(vocabAnalysisDuration/totalProcessingTime*100)}%.1f%%)")
      logWriter.println(f"Stage 7 - Results Collection: $resultsCollectionDuration%.3f seconds (${(resultsCollectionDuration/totalProcessingTime*100)}%.1f%%)")
      logWriter.println(f"\n--- PERFORMANCE SUMMARY ---")
      logWriter.println(f"Total Processing Time: $totalProcessingTime%.3f seconds")
      logWriter.println(f"Overall Throughput: ${recordCount/totalProcessingTime}%.0f records/second")
      logWriter.println(f"Training Throughput: ${recordCount/fitDuration}%.0f records/second")
      logWriter.println(f"Transform Throughput: ${recordCount/transformDuration}%.0f records/second")
      logWriter.println(f"\n--- DATA STATISTICS ---")
      logWriter.println(s"Vocabulary size (unique terms): $actualVocabSize")
      logWriter.println(s"Feature vector dimensions: ${normalizedFeaturesVector.size}")
      logWriter.println(f"Vocabulary density: ${actualVocabSize.toDouble/recordCount}%.2f unique terms/document")
      logWriter.println(f"Raw TF-IDF L2 norm (sample): $rawNorm%.6f")
      logWriter.println(f"Normalized L2 norm (sample): $normalizedNorm%.6f")
      logWriter.println("Normalization: L2 (Euclidean) normalization applied")
      logWriter.println(f"\n--- FILE LOCATIONS ---")
      logWriter.println(s"Log file: ${new File(log_path).getAbsolutePath}")
      logWriter.println("\nFor real-time monitoring, access Spark UI at: http://localhost:4040")
      
      val logWriteDuration = (System.nanoTime() - logWriteStartTime) / 1e9d
      println(f"Performance log written to $log_path in $logWriteDuration%.3f seconds")
    } finally {
      logWriter.close()
    }

    // Save the results to a file
    val resultFileStartTime = System.nanoTime()
    val result_path = "results/lab17_pipeline_output.txt"
    new File(result_path).getParentFile.mkdirs() // Ensure directory exists
    val resultWriter = new PrintWriter(new File(result_path))
    try {
      resultWriter.println(s"--- NLP Pipeline Output (First $n_results results) ---")
      resultWriter.println(s"Generated on: ${new java.util.Date()}")
      resultWriter.println(s"Output file: ${new File(result_path).getAbsolutePath}\n")
      
      var recordsWritten = 0
      results.foreach { row =>
        val text = row.getAs[String]("text")
        val words = row.getAs[Seq[String]]("words")
        val filteredWords = row.getAs[Seq[String]]("filtered_words")
        val rawFeatures = row.getAs[org.apache.spark.ml.linalg.Vector]("raw_features")
        val normalizedFeatures = row.getAs[org.apache.spark.ml.linalg.Vector]("features")
        
        val rawNormSample = math.sqrt(rawFeatures.toArray.map(x => x * x).sum)
        val normalizedNormSample = math.sqrt(normalizedFeatures.toArray.map(x => x * x).sum)
        
        resultWriter.println("="*80)
        resultWriter.println(s"Record #${recordsWritten + 1}")
        resultWriter.println(s"Original Text: ${text.substring(0, Math.min(text.length, 100))}...")
        resultWriter.println(s"Tokenized Words (${words.length} total): ${words.take(10).mkString(", ")}...")
        resultWriter.println(s"Filtered Words (${filteredWords.length} total): ${filteredWords.take(10).mkString(", ")}...")
        resultWriter.println(f"Raw TF-IDF Vector: ${rawFeatures.size} dimensions, ${rawFeatures.numNonzeros} non-zero, L2 norm = $rawNormSample%.6f")
        resultWriter.println(f"Normalized Vector: ${normalizedFeatures.size} dimensions, ${normalizedFeatures.numNonzeros} non-zero, L2 norm = $normalizedNormSample%.6f")
        resultWriter.println("="*80)
        resultWriter.println()
        recordsWritten += 1
      }
      
      val resultFileWriteDuration = (System.nanoTime() - resultFileStartTime) / 1e9d
      println(f"Results file written to $result_path in $resultFileWriteDuration%.3f seconds")
      println(f"Average writing speed: ${recordsWritten/resultFileWriteDuration}%.1f records/second")
    } finally {
      resultWriter.close()
    }

    // =============================================================================
    // STAGE 8: DOCUMENT SIMILARITY ANALYSIS
    // =============================================================================
    println("\n=== STAGE 8: Finding Similar Documents ===")
    val similarityStartTime = System.nanoTime()
    
    // REQUIREMENT: Search and display top K similar documents
    // Tìm kiếm K documents tương tự nhất với reference document bằng cosine similarity
    
    // Chọn reference document (document đầu tiên làm baseline)
    val referenceDocIndex = 0
    val documentsWithIndex = transformedDF.select(
      monotonically_increasing_id().as("doc_id"),  // Tạo unique ID cho mỗi document
      col("text"),
      col("features")                              // Normalized TF-IDF vectors
    ).cache()  // Cache để tránh recomputation
    
    val referenceDoc = documentsWithIndex.filter(col("doc_id") === referenceDocIndex).first()
    val referenceVector = referenceDoc.getAs[Vector]("features")
    val referenceText = referenceDoc.getAs[String]("text")
    
    println(f"Reference document (ID: $referenceDocIndex):")
    println(f"Text preview: ${referenceText.substring(0, Math.min(referenceText.length, 100))}...")
    
    // Tính cosine similarity giữa reference vector và tất cả documents khác
    // Với L2-normalized vectors: cosine_similarity = dot_product
    val cosineSimilarityUDF = udf((vector: Vector) => {
      val refArray = referenceVector.toArray
      val vecArray = vector.toArray
      // Dot product của normalized vectors = cosine similarity
      refArray.zip(vecArray).map { case (a, b) => a * b }.sum
    })
    
    val documentsWithSimilarity = documentsWithIndex
      .filter(col("doc_id") =!= referenceDocIndex)
      .withColumn("cosine_similarity", cosineSimilarityUDF(col("features")))
      .select(col("doc_id"), col("text"), col("cosine_similarity"))
      .orderBy(desc("cosine_similarity"))
      .limit(5)
      .collect()
    
    val similarityDuration = (System.nanoTime() - similarityStartTime) / 1e9d
    
    println(f"\nTop 5 most similar documents (computed in $similarityDuration%.2f seconds):")
    println("=" * 80)
    
    documentsWithSimilarity.zipWithIndex.foreach { case (row, index) =>
      val docId = row.getAs[Long]("doc_id")
      val text = row.getAs[String]("text")
      val similarity = row.getAs[Double]("cosine_similarity")
      
      println(f"${index + 1}. Document ID: $docId (Similarity: $similarity%.6f)")
      println(f"   Text: ${text.substring(0, Math.min(text.length, 150))}...")
      println("-" * 80)
    }
    
    // Save similarity results
    val similarityResultPath = "results/lab17_similarity_results.txt"
    new File(similarityResultPath).getParentFile.mkdirs()
    val similarityWriter = new PrintWriter(new File(similarityResultPath))
    try {
      similarityWriter.println("=== DOCUMENT SIMILARITY ANALYSIS RESULTS ===")
      similarityWriter.println(f"Generated on: ${new java.util.Date()}")
      similarityWriter.println(f"Analysis completed in: $similarityDuration%.3f seconds\n")
      
      similarityWriter.println(f"REFERENCE DOCUMENT (ID: $referenceDocIndex):")
      similarityWriter.println(f"${referenceText.substring(0, Math.min(referenceText.length, 200))}...\n")
      similarityWriter.println("TOP 5 MOST SIMILAR DOCUMENTS:\n")
      
      documentsWithSimilarity.zipWithIndex.foreach { case (row, index) =>
        val docId = row.getAs[Long]("doc_id")
        val text = row.getAs[String]("text")
        val similarity = row.getAs[Double]("cosine_similarity")
        
        similarityWriter.println(f"${index + 1}. DOCUMENT ID: $docId")
        similarityWriter.println(f"   COSINE SIMILARITY: $similarity%.6f")
        similarityWriter.println(f"   TEXT: ${text.substring(0, Math.min(text.length, 300))}...")
        similarityWriter.println("-" * 80)
      }
      
      similarityWriter.println("\nNOTES:")
      similarityWriter.println("- Cosine similarity ranges from -1 to 1 (1 = identical, 0 = orthogonal)")
      similarityWriter.println("- Similarity calculated using L2-normalized TF-IDF vectors")
      similarityWriter.println("- For normalized vectors: cosine_similarity = dot_product")
      
      println(f"Similarity results saved to: $similarityResultPath")
    } finally {
      similarityWriter.close()
    }
    
    documentsWithIndex.unpersist()

    // =============================================================================
    // FINAL PERFORMANCE SUMMARY
    // =============================================================================
    val totalExecutionTime = (System.nanoTime() - sparkInitStartTime) / 1e9d
    println("\n=== FINAL PERFORMANCE SUMMARY ===")
    println(f"Total Execution Time: $totalExecutionTime%.2f seconds")
    println(f"Records Processed: $recordCount")
    println(f"Overall Throughput: ${recordCount/totalExecutionTime}%.0f records/second")
    println(f"Vocabulary Generated: $actualVocabSize unique terms")
    println(f"Feature Dimensions: ${normalizedFeaturesVector.size}")
    println(f"Vector Normalization: L2 norm approximately ${normalizedNorm}%.3f (normalized)")
    println("\nOutput Files:")
    println(s"   - Log: log/lab17_metrics.log")
    println(s"   - Results: results/lab17_pipeline_output.txt")
    println(s"   - Similarity: results/lab17_similarity_results.txt")
    println("\nAll stages completed successfully!")
    
    spark.stop()
    println("\nSpark Session stopped.")
  }
}
