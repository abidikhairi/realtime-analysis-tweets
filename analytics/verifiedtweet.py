from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from utils import author_schema, tweet_schema, verified_tweets_to_elasticsearch

spark = SparkSession.builder \
    .appName("Verified Tweets Count")\
    .master("spark://localhost:7077")\
    .config("spark.driver.memory","4G")\
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.kryoserializer.buffer.max", "2000M")\
	.getOrCreate()

spark.sparkContext.setLogLevel('WARN')

schema = tweet_schema()
author = author_schema()

df = spark \
	.readStream \
	.format("kafka") \
	.option("kafka.bootstrap.servers", "localhost:9092") \
  	.option('minOffsetsPerTrigger', '10')\
	.option('maxOffsetsPerTrigger', '20')\
	.option('subscribe', 'tweets')\
	.load()\
	.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)") \
	.select(F.from_json('value', schema).alias('tweet'))\
	.select('tweet.*')

verifiedTweetsDF = df.withColumn('user', F.from_json('author', author))\
	.select('id', 'user.verified', 'created_at')\
	.withWatermark('created_at', '5 minutes')\
	.groupBy('verified', 'created_at')\
	.count()

verifiedTweetsQuery = verifiedTweetsDF\
	.writeStream \
    .format('console')\
	.outputMode('update')\
    .option("checkpointLocation", "hdfs://localhost:9000/spark/checkpoints/verified-tweets")\
    .start()

elasticSearchQuery = verifiedTweetsDF\
	.writeStream\
	.foreach(verified_tweets_to_elasticsearch)\
	.outputMode('complete')\
	.start()

elasticSearchQuery.awaitTermination()
verifiedTweetsQuery.awaitTermination()

spark.stop()