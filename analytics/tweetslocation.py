from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from utils import tweet_schema, tweetlocation_to_elasticsearch

spark = SparkSession.builder \
    .appName("TweetCount Per Location")\
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
  	.option('minOffsetsPerTrigger', '100')\
	.option('maxOffsetsPerTrigger', '200')\
	.option('subscribe', 'tweets')\
	.option('failOnDataLoss', 'false')\
	.load()\
	.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)") \
	.select(F.from_json('value', schema).alias('tweet'))\
	.select('tweet.*')
    
tweetCountsDF = df.select('place', 'created_at')\
	.filter(F.col('place').isNotNull())\
	.withWatermark('created_at', '5 minutes')\
	.groupBy('place', 'created_at')\
	.count()

wordCountsQuery = tweetCountsDF\
	.writeStream \
    .format("console") \
	.outputMode('complete')\
    .option("checkpointLocation", "/home/flursky/Work/realtime-sentiment-analysis/checkpoints/locations")\
    .start()

elasticSearchQuery = tweetCountsDF\
	.writeStream\
	.foreach(tweetlocation_to_elasticsearch)\
	.outputMode('complete')\
	.start()

elasticSearchQuery.awaitTermination()
wordCountsQuery.awaitTermination()

spark.stop()
