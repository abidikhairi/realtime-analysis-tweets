import json
import torch

from nltk.tokenize import TweetTokenizer



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


emotion_emoji = {
	'sadness': ':cry:',
	'optimism': ':smile:',
	'joy': ':joy:',
	'anger': ':rage:'
}