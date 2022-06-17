import json
import torch
from kafka import KafkaConsumer
from elasticsearch import Elasticsearch
from utils import prepare_sequence, sentiment_labels, load_vocab
from models import LSTMClassifier


if __name__ == '__main__':
	consumer = KafkaConsumer('tweets', bootstrap_servers='localhost:9092')
	es = Elasticsearch('http://localhost:9200')

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	vocab = load_vocab('./data/sentiment-vocab.json')	
	model = LSTMClassifier(vocab_size=len(vocab), embedding_dim=256, hidden_dim=128, num_classes=len(sentiment_labels.keys()))
	model.load_state_dict(torch.load('./data/lstm-sentiment.pt'))
	
	model.to(device)
	model.eval()

	for msg in consumer:
		message = json.loads(msg.value)
		timestamp = message['timestamp']

		seq = prepare_sequence(message['text'], vocab).to(device)

		label = model.inference(seq)
		sentiment = sentiment_labels[label]

		doc = {
			'sentiment': sentiment,
			'count': 1
		}

		idx = timestamp
		es.index(index='tweets-sentiment', document=doc, id=idx)

		if message['place'] is not None:
			doc['country'] = message['place']
			print('-----------------------')
			print(doc)
			print('-----------------------')
			es.index(index='tweets-sentiment-location', document=doc, id=idx)
