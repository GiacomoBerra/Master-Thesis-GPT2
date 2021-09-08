# Complete the sentences with the trained models

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2DoubleHeadsModel
import torch
from time import sleep as sl
from math import ceil, log
from os import path, makedirs
from tqdm import tqdm
from json import load, dump
import csv
import copy

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--job_id", type=int, default=0, help="Job ID of, used when running the code on cluster.")
args = parser.parse_args()

# get the available device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model 
with open(path.join('saved_model', 'job_' + str(args.job_id), 'config.json'), 'r') as json_file:
	json = load(json_file)

# create the folder for saving the results
if not path.exists(path.join('Data', 'job_' + str(args.job_id))):
    makedirs(path.join('Data', 'job_' + str(args.job_id)))

model_name = json['architectures'][0]

print("Testing the model {}.".format(model_name))

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

# add the special tokens
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}
tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)

# Load the sentence to complete 
with open(path.join('Data', 'data_testing_gen.json'), 'r') as json_file:
	sentences = load(json_file)


if 0: #model_name == "GPT2LMHeadModel":
	...

else:
	# dep__dep: sentences depressed completed as depresed 
	# dep__non_dep: sentences depressed completed as non depresed 
	# non_dep__dep: sentences non depresed  completed as depresed 
	# non_dep__non_dep: sentences non depresed  completed as non depresed  
	complete_sent = {'dep__dep':[], 'dep__non_dep':[], 'non_dep__dep':[], 'non_dep__non_dep':[]}


	# fill the new dictionary with the respective sentences and tokenize them, i.e.:
	# depressed sentences in dep__dep and dep__non_dep, 
	# non depressed sentences in non_dep__dep and non_dep__non_dep
	for flag, sents in sentences.items():
		if flag == "depressed":
			for s in sents:
				tokenized_sent = [tokenizer.encode(ATTR_TO_SPECIAL_TOKEN['bos_token'] + ' ' + flag + ' ' + ATTR_TO_SPECIAL_TOKEN['additional_special_tokens'][0] + ' ' + s)]
				complete_sent['dep__dep'] += tokenized_sent
				# change the the lable to non depressed 
				wrong_token_sent = copy.deepcopy(tokenized_sent)
				wrong_token_sent[0][1:3] = [13159, 19095] 
				complete_sent['dep__non_dep'] += wrong_token_sent
		else:
			for s in sents:
				tokenized_sent = [tokenizer.encode(ATTR_TO_SPECIAL_TOKEN['bos_token'] + ' ' + flag + ' ' + ATTR_TO_SPECIAL_TOKEN['additional_special_tokens'][0] + ' ' + s)]
				complete_sent['non_dep__dep'] += tokenized_sent
				wrong_token_sent = copy.deepcopy(tokenized_sent)
				wrong_token_sent[0][1:3] = [10378, 2790]
				complete_sent['non_dep__non_dep'] += wrong_token_sent

	# Load pre-trained model (weights)	
	if model_name == "GPT2LMHeadModel":
		model = GPT2LMHeadModel.from_pretrained(path.join('./saved_model', 'job_' + str(args.job_id), ''))
	else:
		model = GPT2DoubleHeadsModel.from_pretrained(path.join('./saved_model', 'job_' + str(args.job_id), ''))

	# If you have a GPU, put everything on cuda
	model.to(device)

	# Set the model in evaluation mode to deactivate the DropOut modules
	# This is IMPORTANT to have reproducible results during evaluation!
	model.eval()

	# Softmax function to compute the probabilities 
	m = torch.nn.Softmax(dim=1)

	from multiprocessing import Pool, TimeoutError
	import time

	def f (s):
		for i in range(40):
			outputs = model(s)
			# store the prediction
			predictions = outputs[0]
			predictions.to(device)
			# sortt the probabilities of each token that could follows
			v, i = torch.sort(predictions[-1, :], descending=True)
			# set the seedin order to compare the results
			torch.random.manual_seed(2021)
			# select "randomly" the next token from the ns that hava the highest probability
			predicted_index = i[torch.multinomial(i[:3].float(), 1)].item()
			if predicted_index >= 50258:
				break
			# append the prediction 
			s = torch.cat((s, torch.tensor([predicted_index])), 0)
		return tokenizer.decode(s[4:])


	# complete the sentences
	with torch.no_grad():
		if __name__ == '__main__': 
			for flag, sents in tqdm(complete_sent.items()):
				if flag in ['dep__dep', 'dep__non_dep']:
					tonsor_sents = [torch.tensor(s) for s in sents]
					t = time.time()
					with Pool(processes=10) as pool:
					
						res = pool.map(f, tonsor_sents)
						print(f"completed in {time.time() - t}")

						complete_sent[flag] = res

						# Save the complete sentences
						try:
							with open(path.join('Data', 'job_' + str(args.job_id), flag + '.csv'), 'w') as csvfile:
								writer = csv.DictWriter(csvfile, delimiter=',', quotechar='"', fieldnames=['Sentence', 'Number', 'Label']) # fieldnames=['Sentence', 'Number', 'Label'],
								#writer.writeheader()
								predicted_index = 0
								idx = 0
								for sentence in res:
									writer.writerow({'Sentence': sentence, 'Number': 'sentence_' + str(idx), 'Label': flag})
									#writer.writerow([sentence,'sentence_' + str(idx), flag])
									idx += 1
						except IOError:
							print("I/O TimeoutError")

			'''	
			dep__dep = complete_sent['dep__dep']
			dep__non_dep = complete_sent['dep__non_dep']
			non_dep__dep = complete_sent['non_dep__dep']
			non_dep__non_dep = complete_sent['non_dep__non_dep']
			
			
			# Save the complete sentences
			try:
				with open(path.join('Data', 'sentences_completed.csv'), 'w') as csvfile:
					writer = csv.DictWriter(csvfile, fieldnames=['Sentence', 'Number', 'Label'], delimiter=',', quotechar='"')
					writer.writeheader()
					predicted_index = 0
					idx = 0
					for key, value in complete_sent.items():
						for sentence in value:
							writer.writerow({'Sentence': sentence, 'Number': 'sentence_' + str(idx), 'Label': key})
							idx += 1
			except IOError:
				print("I/O TimeoutError")
			'''


# TODO:


# COMPLETED:
# use default python module to save the generated sentences







