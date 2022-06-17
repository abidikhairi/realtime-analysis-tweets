from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from utils import tweet_schema, wordcount_to_elasticsearch

spark = SparkSession.builder \
    .appName("WordCount Twitter")\
    .master("spark://localhost:7077")\
    .config("spark.driver.memory","4G")\
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.kryoserializer.buffer.max", "2000M")\
	.getOrCreate()

spark.sparkContext.setLogLevel('WARN')

schema = tweet_schema()

df = spark \
	.readStream \
	.format("kafka") \
	.option("kafka.bootstrap.servers", "localhost:9092") \
  	.option('minOffsetsPerTrigger', '10')\
	.option('maxOffsetsPerTrigger', '50')\
	.option('subscribe', 'tweets')\
	.load()\
	.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)") \
	.select(F.from_json('value', schema).alias('tweet'))\
	.select('tweet.*')
    
wordCountsDF = df.select(F.lower('text').alias('tweet'), 'created_at')\
    .select(F.explode(F.split('tweet', '\W')).alias('word'), 'created_at')\
	.filter(F.length('word') > 3)\
	.withColumn('one', F.lit(1))\
	.withWatermark('created_at', '30 minutes')\
	.groupBy('word', 'created_at')\
	.agg(F.sum('one').alias('count'))\
	.select('word', 'count', 'created_at')\
	.filter(F.col('count') > 1)

wordCountsQuery = wordCountsDF\
	.writeStream \
    .format("console") \
	.outputMode('complete')\
    .option("checkpointLocation", "/home/flursky/Work/realtime-sentiment-analysis/checkpoints/words")\
    .start()

elasticSearchQuery = wordCountsDF\
	.writeStream\
	.foreach(wordcount_to_elasticsearch)\
	.outputMode('complete')\
	.start()

elasticSearchQuery.awaitTermination()
wordCountsQuery.awaitTermination()


spark.stop()
