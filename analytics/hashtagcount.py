from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from utils import tweet_schema, hashtags_to_elasticsearch

spark = SparkSession.builder \
    .appName("Hashtags Count Twitter")\
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
  	.option('minOffsetsPerTrigger', '20')\
	.option('maxOffsetsPerTrigger', '100')\
	.option('subscribe', 'tweets')\
	.load()\
	.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)") \
	.select(F.from_json('value', schema).alias('tweet'))\
	.select('tweet.*')

hashtagsDF = df.select(F.explode('hashtags').alias('hashtag'), 'created_at')\
	.withColumn('hashtag', F.lower('hashtag'))\
	.withColumn('one', F.lit(1))\
	.withWatermark('created_at', '45 minutes')\
	.groupBy('hashtag', 'created_at')\
	.agg(F.sum('one').alias('count')).orderBy('count', ascending=False)\
	.select('hashtag', 'count', 'created_at')\
	.filter(F.col('count') > 1)

hashtagsCountsQuery = hashtagsDF\
	.writeStream \
    .format('console')\
	.outputMode('complete')\
    .option("checkpointLocation", "/home/flursky/Work/realtime-sentiment-analysis/checkpoints/hashtags")\
    .start()

elasticSearchQuery = hashtagsDF\
	.writeStream\
	.foreach(hashtags_to_elasticsearch)\
	.outputMode('complete')\
	.start()

elasticSearchQuery.awaitTermination()
hashtagsCountsQuery.awaitTermination()

spark.stop()
