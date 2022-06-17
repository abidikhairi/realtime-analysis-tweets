import torch
import emoji
import streamlit as st
from pages.utils import prepare_sequence, emotion_labels, load_vocab, emotion_emoji
from pages.models import LSTMClassifier


def app():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	vocab = load_vocab('./data/emotion-vocab.json')	
	model = LSTMClassifier(vocab_size=len(vocab), embedding_dim=256, hidden_dim=128, num_classes=len(emotion_labels.keys()))
	model.load_state_dict(torch.load('./data/lstm-emotion.pt'))
	
	model.to(device)
	model.eval()

	st.title('Emotion Analysis')

	text = st.text_area('Text')
	
	if st.button('Predict'):
		result = run_emotion_analysis(text, model, vocab, device)
		write_fn = st.success if result['emotion'] == 'positive' else st.warning
		
		write_fn(emoji.emojize(f"Emotion: {result['emotion']} {emotion_emoji[result['emotion']]}"))
		write_fn(f"Confidence: {result['confidence']*100:.2f} %")

def run_emotion_analysis(text, model, vocab, device):
	seq = prepare_sequence(text, vocab).to(device)

	label, score = model.inference(seq, True)
	emotion = emotion_labels[label]
	
	return {
		'emotion': emotion,
		'confidence': score
	}
