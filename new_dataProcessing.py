# Data processing 

"""
NOTES
1) there is no need to right shift the labels when we call the forward method of the model. Indeed the model takes care of right shifting it, so we can set labels=input_ids
"""

from os import path, getcwd, listdir
from pandas import read_csv
from pandas import DataFrame as df
from re import findall
import xml.etree.ElementTree as ET
from time import sleep as sl
from json import dump, load
from re import sub
from nltk import tokenize
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import time
from math import ceil

from transformers import (AdamW, GPT2LMHeadModel, OpenAIGPTTokenizer,
                                  OpenAIGPTLMHeadModel, GPT2Tokenizer)

from ignite.engine import Engine, Events
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from ignite.contrib.handlers import PiecewiseLinear


class DataProcessing():

	def __init__(self, tokenizer, model, optimizer, tokens={'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>'}, path_to_data=[],
				 perc_training=0.8, max_len=280, load_json=True, create_DataLoaders=False, gradient_accumulation_steps=10, batch_size=5, lr=0.0006, n_epochs=10):
		self.tokenizer = tokenizer
		self.tokens = tokens
		self.model = model
		self.optimizer = optimizer
		self.max_len = max_len
		self.lr = lr
		self.n_epochs = n_epochs
		self.perc_training = perc_training
		self.gradient_accumulation_steps = gradient_accumulation_steps
		self.batch_size = batch_size

		self.some_print()

		self.encode_new_tokens()

		if load_json == False:
			self.GetAllSubjectsNumber(path_to_data)
		else:
			self.load_preComputed_json()

		if create_DataLoaders == True:
			self.createDataset()
		else:
			self.load_DataLoaders()

		self.start_training()
		#self.train()


	def encode_new_tokens(self):
		self.new_tokens_encoded = {}
		for k, v in self.tokens.items():
			if not isinstance(v, list):
				self.new_tokens_encoded[k] = self.tokenizer.encode(v)
			else:
				self.new_tokens_encoded[k] = []
				for item in v:
					self.new_tokens_encoded[k] += self.tokenizer.encode(item)



	def some_print(self):
		print("Actual batch size: {} (gradient_accumulation_steps * batch_size)\n".format(self.gradient_accumulation_steps * self.batch_size))
		print("Memory batch size: {} Kbytes\n\n".format(round((8 * self.batch_size * self.max_len) / 1024, 2)))


	def load_preComputed_json(self):
		with open(path.join('data_json.json'), 'r') as json_file:
			self.data = load(json_file)


	def load_DataLoaders(self):
		print("Loading the precomputed DataLoaders")
		self.train_loader = torch.load('train_loader.pth')
		self.test_loader = torch.load('test_loader.pth')


	def GetAllSubjectsNumber(self, path_to_data):
		base_dir = path.join(path.dirname(getcwd()), path_to_data[0], path_to_data[1], path_to_data[2])
		# name of the csf file with all the users and depression flags
		file_csv = [x for x in listdir(base_dir) if x[-3:] == 'csv'][0]
		# directory to the csv file
		dir_file = path.join(base_dir, file_csv)

		# read the csv file
		data_table = read_csv(dir_file, sep=';', names=["file_name", "flag", "n_sent"])

		data_dict = {"depressed": {}, "non depressed": {}}
		dataset = []

		dir_files = path.join(base_dir, path_to_data[3])

		chunks_list = [x for x in listdir(dir_files) if x[0] == 'c']

		# for all the chunks folders
		for chunks_fold in chunks_list:	
			print(chunks_fold)		
			files_xml = [x for x in listdir(path.join(base_dir, path_to_data[3], chunks_fold)) if x[0] != '.']
			# for all the xml files 
			for xml_file in files_xml:
				persona = self.xmlParser(path.join(base_dir, path_to_data[3], chunks_fold), xml_file)
				clean_file_name = findall(r'([\w]+\d+(?=_))', xml_file)[0]

				flag = data_table.loc[data_table['file_name'] == clean_file_name]['flag'].values[0]
				
				if flag == 1:
					if clean_file_name in data_dict["depressed"].keys():
						data_dict["depressed"][clean_file_name] += persona
					else:
						data_dict["depressed"][clean_file_name] = persona
				elif flag == 0:
					if clean_file_name in data_dict["non depressed"].keys():
						data_dict["non depressed"][clean_file_name] += persona
					else:
						data_dict["non depressed"][clean_file_name] = persona
				else:
					raise Exception('Unknown flag: {}'.format(flag))

		with open(path.join('data_json.json'), 'w') as outfile:
			dump(data_dict, outfile)

		self.data = data_dict


	def xmlParser(self, pat, filename):
		tree = ET.parse(path.join(pat, filename))

		persona = []
		for neighbor in tree.iter('TEXT'):
			sentence = neighbor.text.replace('\n', '').replace('\n\n', '').replace('\n\n\n', '').replace('\t', '').replace("\\", '')
			if not sentence.isspace():
				#print(sentence.strip())
				#print(sub(r' +', ' ', sub(r'(?i)([\(|\[\|\{]*(?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'\".,<>?«»~“”‘’]|[\(\[\{])[\)|\]|\}]*)', '', sub(r'[\)|\]|\}]+[\(|\[|\{]+', ' ',sentence.strip()))), end='\n\n')
				#sent = sub(r' +', ' ', sub(r'(?i)([\(|\[\|\{]*(?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'\".,<>?«»~“”‘’]|[\(\[\{])[\)|\]|\}]*)', '', sub(r'[\)|\]|\}]+[\(|\[|\{]+', ' ',sentence.strip())))
				persona.append(sentence.strip())

		return persona

	def pad_dataset(self, dataset, padding=0):
		""" Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
		MODEL_INPUTS = ["input_ids", "attention_mask"]
		new_dataset = {"train":[], "test":[]}
		max_l = max(len(x) for y in dataset.values() for x in y)

		for name in dataset.keys():
			input_ids = []
			attention_mask = []
			for x in dataset[name]:
				input_ids.append(x + [padding] * (max_l - len(x)))
				attention_mask.append([1] * len(x) + [0] * (max_l - len(x)))
			new_dataset[name].append(torch.tensor(input_ids))
			new_dataset[name].append(torch.tensor(attention_mask))

		'''
		for name in dataset.keys():
			dataset[name] = [x + [padding] * (max_l - len(x)) for x in dataset[name]]
		return dataset
		'''
		return new_dataset

	def count_removed_senteces(self, dataset):
		# compute the length of each sentences
		train_sent_train = [len(x.split(' ')) for y in dataset.values() for x in y]
		# count the frequency of each sentences' lengths
		count = Counter(train_sent_train)

		sum_ = 0
		for k, v in count.items():
			if k <= self.max_len:
				sum_ += v

		print(f"Number of sentences with a lengths >= {self.max_len} are: {len(train_sent_train) - sum_}")


	def create_sentences(self, sents, flag):
		# tokenize the sentences of the all review
		tokenized_sent = [self.tokenizer.encode(x) for x in sents]

		# extra length that will be added at the sentence (+3 = +1 for <bos>, +1 for <speaker1> and +1 for <eos>) 
		extra_len = len(self.tokenizer.encode(flag)) + 3
		# length of the complete review
		complete_len = sum([len(x) for x in tokenized_sent])  
			
		new_sents = []
		if complete_len + extra_len <= self.max_len:
			return [' '.join(sents)]
		elif len(sents) == 2:
			return sents
		else:
			new_sents.append([sents[0], len(tokenized_sent[0])])

			sent_1 = False
			for i in range(1, len(sents), 1):
				if new_sents[-1][1] + len(tokenized_sent[i]) + extra_len <= self.max_len:
					new_sents[-1][0] = ' '.join([new_sents[-1][0], sents[i]])
					new_sents[-1][1] += len(tokenized_sent[i]) 
				else: 
					new_sents.append([sents[i], len(tokenized_sent[i])])

			return [x[0] for x in new_sents]
		

	def createDataset(self):
		dataset_non_tokenized = {"depressed": [], "non depressed": []}
		n_sent = {"depressed": 0, "non depressed": 0}
		# for all the flags ("depressed" and "non depressed")
		for flag, persons in self.data.items():
			# for all the subjects name
			for per_id, history in persons.items():
				# for all the comments of a subject
				for sent in history:
					# split the comment into sentences
					sents = tokenize.sent_tokenize(sent)
					#print(sents)
					sents = self.create_sentences(sents, flag)
					#print("#"*40)
					# add the tokens
					for i, s in enumerate(sents):
						new_sent = [self.tokenizer.encode(self.tokens['bos_token'] + ' ' + flag), self.tokenizer.encode(self.tokens['additional_special_tokens'][0] + ' ' + s + ' ' + self.tokens['eos_token'])]
						sents[i] = new_sent
						#dataset[flag] += [self.tokenizer.convert_tokens_to_ids(new_sent.split(' '))]
						dataset_non_tokenized[flag] += [new_sent]

					n_sent[flag] += len(sents)


		print(f"Total number of sentences of depressed subjects: {n_sent['depressed']}")
		print(f"Total number of sentences of non depressed subjects: {n_sent['non depressed']}\n\n")

		print(f"Sample the training and validation datasets\n") 
		# randomly shiffle the data
		random.seed(30)
		random.shuffle(dataset_non_tokenized['depressed'])
		random.seed(30)
		random.shuffle(dataset_non_tokenized['non depressed'])

		# join the two list 
		complete_dataset = dataset_non_tokenized['depressed'] + dataset_non_tokenized['non depressed']
		train_data = complete_dataset[round(len(complete_dataset)*self.perc_training):]
		test_data = complete_dataset[:round(len(complete_dataset)*self.perc_training)]

		dataset_tokenized = {"train": train_data, "test": test_data}
		#self.count_removed_senteces({"train": train_data, "test": test_data})

		print(f"Padding the dataset\n")
		self.tensor_datasets = {"train": [], "test": []}
		for name, dataset_list in dataset_tokenized.items():
			input_ids = []
			token_type_ids = []
			attention_mask = []
			for item in dataset_list:
				item_len = len(item[0] + item[1])
				if item_len <= self.max_len:
					input_ids.append(item[0] + item[1] + self.new_tokens_encoded['pad_token'] * (self.max_len - item_len))
					token_type_ids.append([self.new_tokens_encoded['additional_special_tokens'][0]] * len(item[0]) + [self.new_tokens_encoded['additional_special_tokens'][1]] * len(item[1]) + self.new_tokens_encoded['pad_token'] * (self.max_len - item_len))
					attention_mask.append([1] * len(item[0] + item[1]) + [0] * (self.max_len - item_len))

					a = item[0] + item[1] + self.new_tokens_encoded['pad_token'] * (self.max_len - item_len)
					b = [self.new_tokens_encoded['additional_special_tokens'][0]] * len(item[0]) + [self.new_tokens_encoded['additional_special_tokens'][1]] * len(item[1]) + self.new_tokens_encoded['pad_token'] * (self.max_len - item_len)
					c = [1] * len(item[0] + item[1]) + [0] * (self.max_len - item_len)

					print(a)
					print(b)
					print(c)
					print(len(a))
					print(len(b))
					print(len(c))
					c = 1/0



			self.tensor_datasets[name].append(torch.tensor(input_ids))
			self.tensor_datasets[name].append(torch.tensor(token_type_ids))
			self.tensor_datasets[name].append(torch.tensor(attention_mask))

		# create dataloaders
		train_dataset = TensorDataset(*self.tensor_datasets["train"])
		test_dataset = TensorDataset(*self.tensor_datasets["test"])
		self.train_loader = DataLoader(train_dataset, sampler=None, batch_size=self.batch_size, shuffle=False)
		self.test_loader = DataLoader(test_dataset, sampler=None, batch_size=self.batch_size, shuffle=False)

		# save the dataloaders
		torch.save(self.train_loader, 'train_loader.pth')
		torch.save(self.test_loader, 'test_loader.pth')


	def train(self, engine, batch):
		print(engine.state.iteration)
		model.train()

		batch = tuple(input_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) for input_tensor in batch)
		input_ids, token_type_ids, attention_mask = batch

		#print(f"input_ids: {input_ids}")
		#print(f"token_type_ids: {token_type_ids}")
		#print(f"attention_mask: {attention_mask}", end="\n\n")

		a = self.model(input_ids=input_ids, token_type_ids= token_type_ids, attention_mask=attention_mask, labels=input_ids)

		''' 
		pred = a.logits
		m = torch.nn.Softmax(dim=2)
		sftmx = m(pred)
		sent = []
		pred_sent = []
		for i in range(sftmx.shape[1]):
			pred_sent.append(torch.argmax(sftmx[-1, i, :]).tolist())
			#v, i = torch.sort(sftmx[-1, i, :], descending=True)
			#predicted_index = i[torch.multinomial(i[:3].float(), 1)].item()
			#pred_sent.append(predicted_index)
		
		print(''.join(self.tokenizer.decode(pred_sent)), end="\n\n")
		print(self.tokenizer.decode(input_ids[-1, :]))
		#c = 1/0
		''' 

		loss = a.loss / self.gradient_accumulation_steps
		t = time.time()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
		
		if engine.state.iteration % self.gradient_accumulation_steps == 0:
			self.optimizer.step()
			self.optimizer.zero_grad()

		print("Loss {}\n\n".format(loss.item()))

		self.losses['loss'].append(loss.item())

		return loss.item()



	def start_training(self):
		self.losses = {'loss':[]}
		self.trainer = Engine(self.train)

		# Linearly decrease the learning rate from lr to zero
		scheduler = PiecewiseLinear(self.optimizer, "lr", [(0, self.lr), (self.n_epochs * len(self.train_loader), 0.0)])
		self.trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

		print("\nStart the training")
		self.trainer.run(self.train_loader, max_epochs=20)
		self.model.save_pretrained(getcwd())

		with open(path.join(getcwd(), 'losses.json'), 'w') as outfile:
			dump(self.losses, outfile)







		






def load_models(model_name):
	# select which model we want to use (gpt2 or gpt) 
	model_class = GPT2LMHeadModel if "gpt2" in model_name else OpenAIGPTLMHeadModel
	# load the model
	model = model_class.from_pretrained(model_name)
	# put the model onto the available device
	model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

	tokenizer_class = GPT2Tokenizer if "gpt2" in model_name else OpenAIGPTTokenizer # cant use Autotokenizer because checkpoint could be a Path
	# load the tokenizer 
	tokenizer = tokenizer_class.from_pretrained(model_name)

	return model, tokenizer


def add_special_tokens_(model, tokenizer):
	""" Add special tokens to the tokenizer and the model if they have not already been added. """
	orig_num_tokens = len(tokenizer.encoder)
	num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
	if num_added_tokens > 0:
		model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)




if __name__ == "__main__":
	ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}
	path_to_data = ["data", "depression", "all", "chunks"]
	max_len = 300
	create_DataLoaders = True
	load_json = True
	gradient_accumulation_steps = 10
	batch_size = 1
	n_epochs = 20
	lr = 0.0006
	model_name = "distilgpt2"
	# load the tokenizer 
	model, tokenizer = load_models(model_name)
	add_special_tokens_(model, tokenizer)

	optimizer = AdamW(model.parameters(), lr=lr, correct_bias=True)

	torch.manual_seed(0)

	DataProcessing(path_to_data=path_to_data, tokenizer=tokenizer, model=model, tokens=ATTR_TO_SPECIAL_TOKEN, max_len=max_len, load_json=load_json, create_DataLoaders=create_DataLoaders,
				   optimizer=optimizer, gradient_accumulation_steps=gradient_accumulation_steps, batch_size=batch_size, lr=lr, n_epochs=n_epochs)

	#self.dataset_tokenized["train"] = torch.tensor(self.dataset_tokenized["train"]) 
		#self.dataset_tokenized["test"] = torch.tensor(self.dataset_tokenized["test"]) 

