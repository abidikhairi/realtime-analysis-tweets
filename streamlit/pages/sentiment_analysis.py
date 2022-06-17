import torch
import streamlit as st
from pages.utils import prepare_sequence, sentiment_labels, load_vocab
from pages.models import LSTMClassifier


def app():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	vocab = load_vocab('./data/sentiment-vocab.json')	
	model = LSTMClassifier(vocab_size=len(vocab), embedding_dim=256, hidden_dim=128, num_classes=len(sentiment_labels.keys()))
	model.load_state_dict(torch.load('./data/lstm-sentiment.pt'))
	
	model.to(device)
	model.eval()

	st.title('Sentiment Analysis')

	text = st.text_area('Text')
	
	if st.button('Predict'):
		result = run_sentiment_analysis(text, model, vocab, device)
		write_fn = st.success if result['sentiment'] == 'positive' else st.warning
		
		write_fn(f"Sentiment: {result['sentiment']}")
		write_fn(f"Confidence: {result['confidence']*100:.2f} %")

def run_sentiment_analysis(text, model, vocab, device):
	seq = prepare_sequence(text, vocab).to(device)

	label, score = model.inference(seq, True)
	sentiment = sentiment_labels[label]

	return {
		'sentiment': sentiment,
		'confidence': score
	}
