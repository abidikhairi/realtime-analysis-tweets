import os
import torch
import json
from nltk.tokenize import TweetTokenizer

from models import LSTMClassifier


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

def load_or_build_vocab(datasets=None, path="./vocab.json"):
	if os.path.exists(path):
		return load_vocab(path)
	
	tknzr = get_tokenizer()

	vocab = {}
	vocab['<UNK>'] = 0
	
	for dataset in datasets:
		for row in dataset:
			for word in tknzr.tokenize(row['text'].lower()): 
				if word not in vocab:
					vocab[word] = len(vocab)

	with open(path, 'w') as fp:
		json.dump(vocab, fp)

	return vocab

def load_vocab(path: str):
	with open(path, 'r') as fp:
		vocab = json.load(fp)
		return vocab
