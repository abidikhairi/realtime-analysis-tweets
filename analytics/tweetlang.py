# importation des objets spark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from utils import author_schema, tweet_schema, lang_tweets_to_elasticsearch

spark = SparkSession.builder \
    .appName("Lang Tweets Count")\
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
  	.option('minOffsetsPerTrigger', '20')\
	.option('maxOffsetsPerTrigger', '50')\
	.option('subscribe', 'tweets')\
	.load()\
	.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)") \
	.select(F.from_json('value', schema).alias('tweet'))\
	.select('tweet.*')

langTweetsDF = df.select('id', 'lang', 'created_at')\
	.withWatermark('created_at', '10 minutes')\
	.withColumn('one', F.lit(1))\
	.groupBy('lang', 'created_at')\
	.agg(F.sum('one').alias('count'))\
	.select('lang', 'count', 'created_at')

# withWatermark -> function spark pour gerer la defaillance en cas de pannes en temps réel

langTweetsQuery = langTweetsDF\
	.writeStream \
    .format('console')\
	.outputMode('update')\
    .option("checkpointLocation", "/home/flursky/Work/realtime-sentiment-analysis/checkpoints/langs")\
    .start()

elasticSearchQuery = langTweetsDF\
	.writeStream\
	.foreach(lang_tweets_to_elasticsearch)\
	.outputMode('complete')\
	.start()

elasticSearchQuery.awaitTermination()

