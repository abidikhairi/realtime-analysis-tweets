import time
import random
from elasticsearch import Elasticsearch
from faker import Factory

if __name__ == '__main__':
	es = Elasticsearch('http://localhost:9200')

	sentiments = ['positive', 'negative', 'neutral']

	faker = Factory.create()

	while True:
		tweets = faker.random.randint(1, 20)
		country = faker.country()
		sentiment = random.choice(sentiments)

		doc = {
			'country': country,
			'sentiment': sentiment,
			'count': tweets
		}

		idx = time.time()
		
		es.index(index='tweets-sentiment-location', document=doc, id=idx)

		print('-'*90)
		print(doc)
		time.sleep(1)
