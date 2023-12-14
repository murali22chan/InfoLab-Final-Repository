import torch
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.util import ngrams
from nltk.lm import MLE
import argparse
import seaborn as sns


device = 'cuda'

# Load the pre-trained language model for perplexity calculation
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


#Function to calculate perplexity
def get_perplexity(input_text):
	# Tokenize the input text for perplexity calculation
	input_ids = tokenizer.encode(input_text, truncation = True, return_tensors = 'pt').to(device)
	# Calculate the perplexity of the input text
	with torch.no_grad():
		output = model(input_ids, labels=input_ids)
		loss = output.loss
		perplexity = (torch.pow(2, loss))

	return perplexity.item()

def plot_ppl_distribution(df):
	"""
	Plot perplexity distribution for human and ChatGPT answers.
	"""
	human_data = df[df['label'] == 0]  # Filter human answers
	chatgpt_data = df[df['label'] == 1]  # Filter ChatGPT answers

	human_ppl = human_data['ppl'].values  # Extract perplexity values for human
	chatgpt_ppl = chatgpt_data['ppl'].values  # Extract perplexity values for ChatGPT

	fig, ax = plt.subplots()

	# Plot KDE for human-generated text
	sns.kdeplot(np.log(human_ppl), label='Human', ax=ax)

	# Plot KDE for ChatGPT answers
	sns.kdeplot(np.log(chatgpt_ppl), label='ChatGPT', ax=ax)

	# Set x-axis label
	ax.set_xlabel('Perplexity')

	# Set y-axis label
	ax.set_ylabel('Density')

	# Set title
	ax.set_title('Perplexity Distribution for Human vs ChatGPT')

	# Add legend
	ax.legend()
	# Show the plot
	plt.xlim(0, 8)
	plt.savefig('full_data_kde_plot.png')
	plt.show()



if __name__ == '__main__':
	
	path = "data_full.csv"


	df = pd.read_csv(path)

	# Calculate perplexity for 'answers' column

	df['ppl'] = df['answer'].apply(get_perplexity)

	# Plot perplexity distribution

	plot_ppl_distribution(df)

