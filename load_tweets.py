import os
import json
import time
import tweepy as tw
from kafka import KafkaProducer
from dotenv import load_dotenv

load_dotenv()

# recuperer les variables d'environnement

consumer_key = 'JdG0QpBQ5pSoYBtw7PLCxr3YK'
consumer_secret = '0bdsK4LwCPREU4rBAkxvNogF90HfRK3iJumLqjBAm8MgqrBhdD'
access_key = '1296498183665848320-r7axdEBmMjjwhk5uEAYVJgbjm09Dhw'
access_secret = 'UVfooEK2oZ6MAOivpjExPoV3j1JB20RQyzJ0qbjkuO8Dh'

# Kafka config
bootstrap_servers = ['localhost:9092']
topic = 'tweets'

# config recherche
keywords = ['usa', 'russia', 'ukraine']
langs = ['en', 'fr', 'es', 'ar']


def extract_tweet_information(status):	
	hashtags = list(map(lambda hashtag: hashtag['text'], status.entities['hashtags']))
	
	user = status.user
	text = status.text
	tweet_id = status.id_str
	created_at = str(status.created_at)
	timestamp = status.timestamp_ms
	lang = status.lang
	geo = status.geo
	reply_count = status.reply_count
	retweet_count = status.retweet_count
	favorite_count = status.favorite_count
	
	place = status.place.country if status.place is not None else None

	author = {
		'id': user.id_str,
		'screen_name': user.screen_name,
		'followfavourites_counters_count': user.followers_count,
	   	'friends_count': user.friends_count,
		'statuses_count': user.statuses_count,
		'favourites_count': user.favourites_count,
		'protected': int(user.protected), 
		'location': user.location,
		'verified': user.verified
	}

	return {
		'id': tweet_id,
		'text': text,
		'hashtags': hashtags,
		'created_at': created_at,
		'timestamp': timestamp,
		'lang': lang,
		'geo': geo,
		'reply_count': reply_count,
		'retweet_count': retweet_count,
		'favorite_count': favorite_count,
		'author': author,
		'place': place
	}, tweet_id.encode('utf-8')


class KafkaWriter(tw.Stream):
	def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret, *, chunk_size=512, daemon=False, max_retries=..., proxy=None, verify=True):
		super().__init__(consumer_key, consumer_secret, access_token, access_token_secret, chunk_size=chunk_size, daemon=daemon, max_retries=max_retries, proxy=proxy, verify=verify)
		# initialisation du producer
		self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
	
	def on_status(self, status):
		tweet, tweet_id = extract_tweet_information(status)

		tweet_bytes = json.dumps(tweet).encode('utf-8') # octets json
		
		# ecrire dans kafka
		self.producer.send(topic, value=tweet_bytes, key=tweet_id)
		
		# time.sleep(1)
		return super().on_status(status)

# creation d'un objet d'authentification
auth = tw.OAuthHandler(consumer_key=consumer_key, consumer_secret=consumer_secret)
auth.set_access_token(key=access_key, secret=access_secret)

# initialisation du stream
stream = KafkaWriter(consumer_key, consumer_secret, access_key, access_secret, max_retries=10)

# commencer le stream
stream.filter(track=keywords, languages=langs)
