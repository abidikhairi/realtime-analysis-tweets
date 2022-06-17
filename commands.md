### run docker
```bash
docker-compose up 
```

### collect tweets from twitter and store in kafka
```bash
python load_tweets.py
```

### run streamlit app
```bash
streamlit run streamlit/main.py
```

### kafkadrop
```bash
java --add-opens=java.base/sun.nio.ch=ALL-UNNAMED -jar tools/kafdrop-3.28.0.jar --kafka.brokerConnect=localhost:9092 --server.port=10000
```

### analyse descriptive

#### hashtag count
```bash
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 --py-files analytics/utils.py analytics/hashtagcount.py
```

#### word count
```bash
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 --py-files analytics/utils.py analytics/wordcount.py
```

#### count tweets per language
```bash
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 --py-files analytics/utils.py analytics/tweetlang.py
```

#### count tweets per country
```bash
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 --py-files analytics/utils.py analytics/tweetslocation.py
```

#### ratio of verified tweets
```bash
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0 --py-files analytics/utils.py analytics/verifiedtweet.py
```

### analyse predictive

#### count emotions in tweets
```bash
python analytics/emotioncount.py
```

#### count sentiments in tweets
```bash
python analytics/sentimentanalysis.py
```

#### count emotions in tweets
```bash
python analytics/sentimenttextlength.py
```

### entrainement model predictive

#### model d'analyse sentimentale
```bash
python training/sentiment.py
```

#### model d'analyse emotionelle
```bash
python training/emotion.py
```

### seeders
```bash
python seeders/sentimentlocation.py
```

```bash
python seeders/tweetlocation.py
```