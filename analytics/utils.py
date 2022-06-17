import json
import torch

# importation de la bibliothéque elasticsearch
from elasticsearch import Elasticsearch

# importation des defintions des types
from pyspark.sql import Row
from pyspark.sql.types import BooleanType, StructType, StructField
from pyspark.sql.types import StringType, LongType, ArrayType, TimestampType

from nltk.tokenize import TweetTokenizer

es = Elasticsearch('http://localhost:9200')

def tweet_schema():

	return StructType([
		StructField("id", StringType(), False), 
    	StructField("text", StringType(), False), 
    	StructField("hashtags", ArrayType(StringType(), True), True),
		StructField('lang', StringType()),
		StructField('created_at', TimestampType()),
		StructField('reply_count', LongType()),
		StructField('retweet_count', LongType()),
		StructField('favorite_count', LongType()),
		StructField('place', StringType()),
		StructField('author', StringType(), False)
  	])

def author_schema():
	return StructType([
		StructField("id", StringType()),
		StructField("screen_name", StringType()),
		StructField("followers_count", LongType()),
		StructField("friends_count", LongType()),
		StructField("statuses_count", LongType()),
		StructField("favourites_count", LongType()),
		StructField("protected", BooleanType()),
		StructField("verified", BooleanType()),
		StructField("location", StringType()),
	])

def sentiment_location_to_elastic_search(row: Row):
	doc = {
		'count': row['count'],
		'sentiment': row['sentiment'],
		'country': row['place']
	}

	es.index(index='sentiments-location', document=doc)

def lang_tweets_to_elasticsearch(row: Row):
	doc = {
		'count': row['count'],
		'lang': row['lang'],
		'date': row['created_at']
	}

	idx = str(f'{row["created_at"]}-{row["lang"]}-{row["count"]}')

	es.index(index='tweets-lang', document=doc, id=idx)

def verified_tweets_to_elasticsearch(row: Row):
	doc = {
		'count': row['count'],
		'verified': row['verified'],
		'date': row['created_at']
	}

	idx = row['created_at']

	es.index(index='tweets-verified', document=doc, id=idx)

def sentiment_time_to_elastic_search(row: Row):
	doc = {
		'count': row['count'],
		'sentiment': row['sentiment'],
		'date': row['created_at']
	}

	idx = row['created_at']

	es.index(index='tweets-sentiments', document=doc, id=idx)

def tweetlocation_to_elasticsearch(row: Row):
	doc = {
		'country': row['place'],
		'count': row['count'],
		'date': row['created_at']
	}

	idx = str(f'{row["created_at"]}-{row["place"]}-{row["count"]}')

	es.index(index='locations', document=doc, id=idx)

def wordcount_to_elasticsearch(row: Row):
	doc = {
		'count': row['count'],
		'word': row['word'],
		'created_at': row['created_at']
	}

	idx = str(f'{row["created_at"]}-{row["word"]}-{row["count"]}')
	
	es.index(index='tweets-words', document=doc, id=idx)

def emotion_to_elasticsearch(row: Row):
	doc = {
		'count': row['count'],
		'emotion': row['emotion']
	}

	idx = row['created_at']
	
	es.index(index='tweets-emotions', document=doc, id=idx)

def hashtags_to_elasticsearch(row: Row):

	doc = {
		'count': row['count'],
		'hashtag': row['hashtag'],
		'created_at': row['created_at']
	}

	idx = str(f'{row["created_at"]}-{row["hashtag"]}-{row["count"]}')
	
	es.index(index='tweets-hashtags', document=doc, id=idx)

def load_vocab(path: str):
	with open(path, 'r') as fp:
		vocab = json.load(fp)
		return vocab

def prepare_sequence(seq: str, word2idx):
	tknzr = get_tokenizer()
	words = tknzr.tokenize(seq.lower())

	tokens = []
	for word in words:
		if word in word2idx:
			tokens.append(word2idx[word])
		else:
			tokens.append(word2idx['<UNK>'])
	return torch.tensor(tokens, dtype=torch.long)


def get_tokenizer():
	return TweetTokenizer(preserve_case=False)


sentiment_labels = {
	2: 'positive',
	1: 'neutral',
	0: 'negative'
}

emotion_labels = {
	3: 'sadness',
	2: 'optimism',
	1: 'joy',
	0: 'anger',
}
