import time
from elasticsearch import Elasticsearch
from faker import Factory

if __name__ == '__main__':
	es = Elasticsearch('http://localhost:9200')

	faker = Factory.create()

	while True:
		tweets = faker.random.randint(0, 20)
		country = faker.country()
		
		doc = {
			'country': country,
			'count': tweets
		}

		idx = time.time()
		
		es.index(index='locations', document=doc, id=idx)

		print('-'*90)
		print(doc)
		time.sleep(1)
