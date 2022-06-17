import torch
import torchmetrics.functional as metrics
from datasets import load_dataset

from models import LSTMClassifier
from utils import prepare_sequence, sentiment_labels, load_or_build_vocab


if __name__ == '__main__':
	batch_size = 2048
	num_epochs = 10
	learning_rate = 0.01
	embedding_dim = 256
	hidden_dim = 128

	dataset = load_dataset('tweet_eval', 'sentiment')
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')	
	
	train_data = dataset['train']
	test_data = dataset['test']

	word2idx = load_or_build_vocab([train_data, test_data], './data/sentiment-vocab.json')
	vocab_size = len(word2idx)
	num_classes = len(sentiment_labels.keys())

	model = LSTMClassifier(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_classes=num_classes)
	loss_fn = torch.nn.NLLLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	
	model.to(device)

	epoch_loss = []
	epoch_accuracy = []

	num_batches = len(train_data)

	for epoch in range(num_epochs):
		model.train()

		batch_loss = []
		batch_accuracy = []
		
		preds = []
		labels = []
		
		for batch_idx, row in enumerate(train_data):
			
			seq = prepare_sequence(row['text'], word2idx).to(device)
			label = torch.tensor(row['label']).unsqueeze(-1).to(device)
			
			scores = model(seq)
			loss = loss_fn(scores, label)
			
			loss.backward()

			preds.append(scores.detach().cpu().tolist())
			labels.append(label.detach().cpu().item())

			if batch_idx > 0 and batch_idx % batch_size == 0:
				b_loss = loss_fn(torch.tensor(preds).squeeze(1), torch.tensor(labels)).item()
				b_accuracy = metrics.accuracy(torch.tensor(preds).squeeze(1), torch.tensor(labels)).item()
				
				template_message = f'training:\tepoch: [{epoch + 1}/{num_epochs}]\tprocessed: [{batch_idx}/{num_batches}]\ttrain loss: {b_loss:.4f}\ttrain accuracy: {b_accuracy*100:.2f}%'
				print(template_message, flush=True)
				
				batch_loss.append(b_loss)
				batch_accuracy.append(b_accuracy)

				optimizer.step()
				optimizer.zero_grad()

		with torch.no_grad():
			model.eval()
			test_preds = []
			test_labels = []

			for row in test_data:
				seq = prepare_sequence(row['text'], word2idx).to(device)
				label = torch.tensor(row['label']).unsqueeze(-1).to(device)
			
				scores = model(seq)
				
				test_preds.append(scores.tolist())
				test_labels.append(label.item())
			
			test_loss = loss_fn(torch.tensor(test_preds).squeeze(1), torch.tensor(test_labels)).item()
			test_accuracy = metrics.accuracy(torch.tensor(test_preds).squeeze(1), torch.tensor(test_labels)).item()
			
			template_message = f'testing:\tepoch: {epoch + 1}\ttest loss: {test_loss:.4f}\ttest accuracy: {test_accuracy*100:.2f}%'
			print(template_message, flush=True)

			if test_accuracy > 0.50:
				torch.save(model.state_dict(), f'./data/lstm-sentiment-{test_accuracy*100:.2f}.pt')




