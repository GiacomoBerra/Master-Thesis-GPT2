# Data processing 

"""
NOTES
1) there is no need to right shift the labels when we call the forward method of the model. Indeed the model takes care of right shifting it, so we can set labels=input_ids
"""

from os import path, getcwd, listdir, rename, makedirs
from pandas import read_csv
from re import findall, sub
import xml.etree.ElementTree as ET
from json import dump, load
from nltk import tokenize
from random import shuffle, seed
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from math import ceil, floor, exp
from argparse import ArgumentParser
from scipy import stats
from cleantext import clean
from copy import deepcopy
from pprint import pformat
from sklearn.metrics import precision_recall_fscore_support

from transformers import (AdamW, GPT2Tokenizer, GPT2DoubleHeadsModel)

from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.engine import Engine, Events
from ignite.contrib.handlers import PiecewiseLinear, ProgressBar


class DataProcessing():

	def __init__(self, optimizer, args, tokenizer, model, path_to_data=[], tokens={'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>'}):

		self.args = args
		self.tokenizer = tokenizer
		self.tokens = tokens
		self.model = model
		self.optimizer = optimizer

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.model.to(self.device)

		# create the directoryin which the file created will be saved
		self.create_dir()

		self.some_print()

		self.encode_new_tokens()

		if self.args.load_json == False:
			self.GetAllSubjectsNumber(path_to_data)
		else:
			self.load_preComputed_json()

		if self.args.create_DataLoaders:
			self.createDataset()
		else:
			self.load_DataLoaders()

		self.start_training()
		#self.train()


	def create_dir(self):
		if self.args.job_id != 0:
			name = 'job_' + str(self.args.job_id)
			if not path.exists(name):
				makedirs(name)
				self.save_dir = path.join(getcwd(), name)
		else:
			self.save_dir = getcwd()


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
		print("Actual batch size: {} (gradient_accumulation_steps * batch_size)\n".format(self.args.gradient_accumulation_steps * self.args.batch_size))
		print("Memory batch size: {} Kbytes\n\n".format(round((8 * self.args.batch_size * self.args.max_len) / 1024, 2)))


	def load_preComputed_json(self):
		with open(path.join('Data', 'data_json.json'), 'r') as json_file:
			self.data = load(json_file)


	def load_DataLoaders(self):
		print("Loading the precomputed DataLoaders")
		train_dataset = torch.load(path.join('Data', 'train_dataset_doule_inverted.pt'))
		test_dataset = torch.load(path.join('Data', 'test_dataset_doule_inverted.pt'))
		self.train_loader = DataLoader(train_dataset, sampler=None, batch_size=self.args.batch_size, shuffle=False)
		self.test_loader = DataLoader(test_dataset, sampler=None, batch_size=self.args.batch_size, shuffle=False)


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

		with open(path.join('Data', 'data_json.json'), 'w') as outfile:
			dump(data_dict, outfile)

		self.data = data_dict


	def xmlParser(self, pat, filename):
		tree = ET.parse(path.join(pat, filename))

		persona = []
		for neighbor in tree.iter('TEXT'):
			sentence = neighbor.text.replace('\n', '').replace('\n\n', '').replace('\n\n\n', '').replace('\t', '').replace("\\", '')
			if not sentence.isspace():
				# link to use clean (https://github.com/jfilter/clean-text)
				sentence = clean(sentence, no_urls=True, no_emails=True, no_phone_numbers=True, replace_with_url="", replace_with_email="", replace_with_phone_number="")
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

		return new_dataset

	def create_sentences(self, sents, flag):
		# tokenize the sentences of the all review
		tokenized_sent = [tokenizer.encode(x) for x in sents]

		# length of the complete review
		complete_len = sum([len(x) for x in tokenized_sent])  
		
		new_sents = []
		if complete_len <= self.args.max_len:
			self.n_split['0'] += 1
			return [' '.join(sents)]
		elif len(sents) == 2:
			self.n_split['1'] += 1
			return [x for i, x in enumerate(sents) if len(tokenized_sent[i]) <= self.args.max_len]
		else:
			for j, s in enumerate(sents):
				if len(tokenized_sent[j]) <= self.args.max_len:
					new_sents.append([s, len(tokenized_sent[j])])
					break

			for i, sent in enumerate(sents[j+1:]):
				if len(tokenized_sent[i+1+j]) <= self.args.max_len:
					#if len(tokenized_sent[i+1+j]) + new_sents[-1][1] <= self.args.max_len:
					l = len(self.tokenizer.encode(new_sents[-1][0] + ' ' + sent))
					if l <= self.args.max_len: 
						new_sents[-1][0] += ' ' + sent
						new_sents[-1][1] += len(tokenized_sent[i+1+j])	
						assert len(self.tokenizer.encode(new_sents[-1][0])) <= self.args.max_len, f'{len(self.tokenizer.encode(new_sents[-1][0]))}'	
					elif len(tokenized_sent[i+1+j]) <= self.args.max_len:
						new_sents.append([sent, len(tokenized_sent[i+1+j])])			

			self.n_split['n'] += 1
			assert all([len(self.tokenizer.encode(x[0])) <= self.args.max_len for x in new_sents]), f'{[x[0] for x in new_sents]} \n {[len(self.tokenizer.encode(x[0])) for x in new_sents]}'
			return [x[0] for x in new_sents]


	def mix_control_positive(self, dataset):
		'''
		depressed_perc: how many depressed and non depressed subject has to contains the dataset:
		- if perc = .5 there will be 50% of depressed subjects and 50% of non depressed subjects
		- if perc = 1 all the depressed subjets are used are used
		Note: perc represents the percentage of depressed subjets to use, since they are less
		'''
		n_dep = len(dataset['depressed'])
		n_non_dep = len(dataset['non depressed']) 

		if self.args.depressed_perc == 0:
			n_dep = 0
		elif self.args.depressed_perc ==1:
			n_non_dep = 0
		elif floor((n_dep * (1 - self.args.depressed_perc)) / self.args.depressed_perc) <= n_non_dep:
			n_non_dep = floor((n_dep * (1 - self.args.depressed_perc)) / self.args.depressed_perc)
		else:
			n_dep = floor((n_non_dep * self.args.depressed_perc) / (1 - self.args.depressed_perc))

		print(f'n_dep: {n_dep}')
		print(f'n_non_dep: {n_non_dep}')

		# join the two lists
		complete_dataset = dataset['depressed'][:n_dep] + dataset['non depressed'][:n_non_dep]

		# randomly shiffle the data
		seed(30)
		shuffle(complete_dataset)

		# 
		train_data = complete_dataset[:round(len(complete_dataset)*self.args.perc_training)]
		print(f"train_data: {len(train_data)}")
		test_data = complete_dataset[round(len(complete_dataset)*self.args.perc_training):]
		print(f"test_data: {len(test_data)}")

		return {"train": train_data, "test": test_data}

	def print_len_n_sent(self, review_len, n_sent):
		stat = stats.describe(review_len)	
		print("Statistics on the sentences's length expressed as number of WORDS:")
		print(f"\t Mean --> {round(stat.mean, 2)}")
		print(f"\t Min --> {stat.minmax[0]}")
		print(f"\t Max --> {stat.minmax[1]}")
		print(f"\t Variance --> {round(stat.variance, 2)}\n\n")

		print("Number of sentences splitted 0 times: {}".format(self.n_split['0']))
		print("Number of sentences splitted 1 times: {}".format(self.n_split['1']))
		print("Number of sentences splitted n times: {}\n\n".format(self.n_split['n']))

		print(f"Total number of sentences of depressed subjects: {n_sent['depressed']}")
		print(f"Total number of sentences of non depressed subjects: {n_sent['non depressed']}\n\n")		


	def createDataset(self):
		# set to True if you need to change the length of the sentence in the batch 
		if False:
			# to keep track the number of times the sentences are splitted 
			self.n_split = {'0':0, '1':0, 'n':0}

			dataset_non_tokenized = {"depressed": [], "non depressed": []}
			n_sent = {"depressed": 0, "non depressed": 0}
			review_len = []
			# for all the flags ("depressed" and "non depressed")
			for flag, persons in self.data.items():
				# for all the subjects name
				for per_id, history in tqdm(persons.items()):
					# for all the comments of a subject
					for sent in history:
						# split the comment into sentences and tokenize it 
						sents = tokenize.sent_tokenize(sent)
						# store the total length of the review
						review_len.append(len(sent.split(' ')))
						# split the sentences of at most phrases of self.args.max_len
						sents = self.create_sentences(sents, flag)
						# add the tokens
						if len(sents) >= 1:
							for i, s in enumerate(sents):
								new_sent = [self.tokenizer.encode(self.tokens['bos_token'] + ' ' + flag), self.tokenizer.encode(self.tokens['additional_special_tokens'][0] + ' ' + s + ' ' + self.tokens['eos_token'])]
								sents[i] = new_sent
								dataset_non_tokenized[flag] += [new_sent]

							n_sent[flag] += len(sents)

			with open(path.join('Data', 'dataset_non_tokenized_double_inverted.json'), 'w') as outfile:
				dump(dataset_non_tokenized, outfile)

			self.print_len_n_sent(review_len, n_sent)

		else:
			with open(path.join('Data', 'dataset_non_tokenized_double_inverted.json'), 'r') as json_file:
				dataset_non_tokenized = load(json_file)

		print(f"Sample the training and validation datasets\n") 
		dataset_tokenized = self.mix_control_positive(dataset_non_tokenized)

		print(f"Padding the dataset\n")
		extra_length = len(tokenizer.encode('non depressed')) + 3
		lengthss = []
		self.tensor_datasets = {"train": [], "test": []}
		for name, dataset_list in dataset_tokenized.items():
			# inputs of the model
			input_ids = []
			# indexes from where the classification
			mc_token_ids = []
			# labels of the language model head
			lm_labels = []
			# labels of the classification head
			mc_labels = []
			# tokens representing the part of the batch sentence (row)
			token_type_ids = []
			# in this list all 0 will represent a non-depressed subject and 1 a depressed subject
			dep_non_dep_lables = []
			for item in tqdm(dataset_list):
				item_len = len(item[0] + item[1])
				lengthss.append(item_len)
				#assert item_len <= self.args.max_len + extra_length, f'length = {item_len}\n{self.tokenizer.decode(item[0] + item[1])}'
				if item_len <= self.args.max_len + extra_length:
					# Inputs
					correct = [item[0] + item[1] + self.new_tokens_encoded['pad_token'] * (self.args.max_len + extra_length - item_len), True]
					# if the current sentence is from a depressed subject 	
					if [10378, 2790] == item[0][1:3]:
						item_0 = deepcopy(item[0])
						# create the wrong label
						item_0[1:3] = [13159, 19095] # non depressed tokenized
						dep_non_dep_lables.append([1])
					else:
						item_0 = deepcopy(item[0])
						# create the wrong label
						item_0[1:3] = [10378, 2790] # depressed tokenized
						dep_non_dep_lables.append([0])
					wrong = [item_0 + item[1] + self.new_tokens_encoded['pad_token'] * (self.args.max_len + extra_length - item_len), False]
					# shuffle randomly the two sentences with the correct and wrong label 
					pre_input_id = [correct, wrong]
					
					shuffle(pre_input_id)
					input_ids.append([pre_input_id[0][0], pre_input_id[1][0]])

					# Index of the classification token in each input sequence
					if self.args.pos_class == 'end_sent':
						mc_token_ids.append([item_len - 1 for i in range(2)])
					elif self.args.pos_class == 'end_lab':
						mc_token_ids.append([len(item[0]) for i in range(2)])

					# Labels for language modeling
					pre_lm_labels = deepcopy(pre_input_id)
					# substitute the padding token with -100. All labels set to -100 are ignored (masked), the loss is only computed for labels in [0, ..., config.vocab_size - 1]
					pre_lm_labels[0][0][item_len:] = [-100 for i in range(55 - item_len)]
					pre_lm_labels[1][0][item_len:] = [-100 for i in range(55 - item_len)]
					lm_labels.append([pre_lm_labels[0][0], pre_lm_labels[1][0]])

					# Labels for computing the multiple choice classification loss
					# position in where the correct label is: 0 first raw, 1 second raw
					mc_labels.append([i for i in range(2) if pre_input_id[i][1]])

					# Segment token indices to indicate first and second portions of the inputs
					local = [self.new_tokens_encoded['additional_special_tokens'][0]] * len(item[0]) + [self.new_tokens_encoded['additional_special_tokens'][1]] * len(item[1]) + self.new_tokens_encoded['pad_token'] * (self.args.max_len + extra_length - item_len)
					token_type_ids.append([local, local])

			self.tensor_datasets[name].append(torch.tensor(input_ids))
			self.tensor_datasets[name].append(torch.tensor(mc_token_ids))
			self.tensor_datasets[name].append(torch.tensor(lm_labels)) 
			self.tensor_datasets[name].append(torch.tensor(mc_labels))
			self.tensor_datasets[name].append(torch.tensor(token_type_ids))
			self.tensor_datasets[name].append(torch.tensor(dep_non_dep_lables))


		# create dataloaders
		train_dataset = TensorDataset(*self.tensor_datasets["train"])
		test_dataset = TensorDataset(*self.tensor_datasets["test"])
		self.train_loader = DataLoader(train_dataset, sampler=None, batch_size=self.args.batch_size, shuffle=False)
		self.test_loader = DataLoader(test_dataset, sampler=None, batch_size=self.args.batch_size, shuffle=False)

		# save the dataloaders
		torch.save(train_dataset, path.join('Data','train_dataset_doule_inverted.pt'))
		torch.save(test_dataset, path.join('Data','test_dataset_doule_inverted.pt'))


	def train(self, engine, batch):

		model.train()

		batch = tuple(input_tensor.to(self.device) for input_tensor in batch)
		input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids, _ = batch

		#print(f'input_ids.shape: {input_ids.shape}')
		#print(f'input_ids: {input_ids[0,:]}\n')
		#print(f'mc_token_ids.shape: {mc_token_ids.shape}')
		#print(f'mc_token_ids: {mc_token_ids}\n')
		#print(f'lm_labels.shape: {lm_labels.shape}')
		#print(f'lm_labels: {lm_labels[0,:]}\n')
		#print(f'mc_labels.shape: {mc_labels.shape}')
		#print(f'mc_labels: {mc_labels}\n')
		#print(f'token_type_ids.shape: {token_type_ids.shape}')
		#print(f'token_type_ids: {token_type_ids[0,:]}')

		#self.check_GPU_free_mem()
		a = model(input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids, mc_labels=mc_labels, labels=lm_labels)

		# delete the data that we will not use 
		del input_ids
		del mc_labels
		del mc_token_ids
		del token_type_ids
		del lm_labels	

		lm_loss = a.loss
		mc_loss = a.mc_loss
		lm_loss.to(self.device)
		mc_loss.to(self.device)

		loss = (lm_loss + mc_loss) / self.args.gradient_accumulation_steps
		loss.to(self.device)

		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
	
		if engine.state.iteration % self.args.gradient_accumulation_steps == 0:
			self.optimizer.step()
			self.optimizer.zero_grad()
		
		if engine.state.iteration % 3000 == 0:
			self.save_training()

		if engine.state.iteration % (floor(engine.state.epoch_length/10)+1) == 0 or engine.state.iteration == 1:
			print('#'*100)
			print("Evaluation")
			print('#'*100, end='\n\n')
			# initialize the tensors in with the predictions and labels for the classification will be stored 
			self.new_init_metrics()
			# evaluate the model
			self.evaluator.run(self.test_loader)
			# compute the matrics and store the values
			self.compute_f1_rec_prec_acc()

		#print("Epoch: {}".format(engine.state.epoch))
		#print("Loss {}\n\n".format(loss.item()))

		self.train_info[engine.state.epoch]['lm_loss'].append(lm_loss.item() / self.args.gradient_accumulation_steps)
		self.train_info[engine.state.epoch]['mc_loss'].append(mc_loss.item() / self.args.gradient_accumulation_steps)
		self.train_info[engine.state.epoch]['total_loss'].append(loss.item())


	# Evaluation function and evaluator (evaluator output is the input of the metrics)
	def inference(self, engine, batch):
		model.eval()
		with torch.no_grad():
			batch = tuple(input_tensor.to(self.device) for input_tensor in batch)
			input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids, dep_non_dep_lables = batch

			a = model(
				input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
			)

			lm_logits = a.logits
			mc_logits = a.mc_logits
			del a
			lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
			lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

			m = torch.nn.Softmax(dim=1)
			a = torch.round(m(mc_logits))
			b = torch.tensor([torch.argmax(i) for i in a]).unsqueeze(1)

			x = ((lm_logits_flat_shifted.to(self.device), b.to(self.device)), (lm_labels_flat_shifted.to(self.device), mc_labels.to(self.device)))

			self.compute_metrics(x, dep_non_dep_lables)


	def compute_f1_rec_prec_acc(self):
		# Accuracy
		correct = float(sum(self.y == self.y_pred))
		total = float(self.y.shape[0])
		self.eval_metrics['mc_accuracy'].append(correct / total) 

		# Precision, Recall and F1
		result = precision_recall_fscore_support(self.y.to('cpu'), self.y_pred.to('cpu'))	
		print(result)
		print(self.eval_metrics['mc_accuracy'][-1])
		precision_NonDep, precision_Dep = result[0][0], result[0][1]
		recall_NonDep, recall_Dep = result[1][0], result[1][1]
		f1_NonDep, f1_Dep = result[2][0], result[2][1]
		self.eval_metrics['precision_0'].append(precision_NonDep)
		self.eval_metrics['precision_1'].append(precision_Dep)
		self.eval_metrics['recall_0'].append(recall_NonDep)
		self.eval_metrics['recall_1'].append(recall_Dep)
		self.eval_metrics['f1_0'].append(f1_NonDep)
		self.eval_metrics['f1_1'].append(f1_Dep)

		# append the metrics of the evaluation to the list
		self.metrics.append(self.eval_metrics)


	def new_init_metrics(self):
		# tensor with the correct labels
		self.y = torch.tensor([])
		self.y = self.y.to(self.device)
		# tensor with the predicted labels of the classification
		self.y_pred = torch.tensor([])
		self.y_pred = self.y_pred.to(self.device)
		# dictionary with the information of the metrics of a single model evaluation
		self.eval_metrics = {'lm_loss':[], 'lm_perplexity':[], 'mc_accuracy':[], 'precision_0':[], 
							 'precision_1':[], 'recall_0':[], 'recall_1':[], 'f1_0':[], 'f1_1':[]}

	def compute_metrics(self, x, dep_non_dep_lables):
		# Loss of the Language Model
		loss = torch.nn.CrossEntropyLoss(ignore_index=-100)
		#print(self.metrics[self.trainer.state.epoch]['lm_loss'])
		self.eval_metrics['lm_loss'].append(float(loss(x[0][0], x[1][0])) / self.args.gradient_accumulation_steps)

		# Perplexity of the Language Model
		self.eval_metrics['lm_perplexity'].append(exp(self.eval_metrics['lm_loss'][-1]))

		# create the correct lables for conputing the score. Since classification lables and prediction do not consider if the subject was
		# depressed or non depressed but instead consider if the correct sentence + label was in the dimention 0 or 1. The loop below create the "correct"
		# predntion based on the mental condition of the subjec. Variable 'dep_non_dep_lables' is a tensor where 0 reprensent that the control subject 
		# while 1 a positive: the position 3 reprenset the sentence + label at position 3 of the batch.
		# E.g: the batch size (BS) is 5 and the person are [dep, non_dep, non_dep, dep, dep], than we have the following vectors
		# vector that represent the COORRECT mental state of a subject
		# dep_non_dep_lables = [1, 0, 0, 1, 1]
		# CORRECT bach size in wich there is the correct pair sentence + label
		# x[1][1] =            [0, 1, 1, 0, 1]
		# PREDICTED bach size in wich there is the correct pair sentence + label
		# x[0][1] =            [0, 0, 1, 1, 0]
		# vector that represent the PREDICTED mental state of a subject
		# y_pred_loc = 		   [1, 1, 0, 0, 0] wich represnet the prediction [dep, dep, non_dep, non_dep, non_dep]

		y_pred_loc = []
		equality = x[1][1] == x[0][1]
		for i, eq in enumerate(equality):
			if eq:
				y_pred_loc.append(dep_non_dep_lables[i])
			else:
				y_pred_loc.append(int(not dep_non_dep_lables[i]))
		y_loc = dep_non_dep_lables
		y_pred_loc = torch.tensor(y_pred_loc)

		self.y = torch.cat((self.y, y_loc.squeeze(1)))
		self.y_pred = torch.cat((self.y_pred, y_pred_loc.to(self.device)))


	def check_GPU_free_mem(self):
		if torch.cuda.is_available():
			t = torch.cuda.get_device_properties(0).total_memory
			r = torch.cuda.memory_reserved(0) 
			a = torch.cuda.memory_allocated(0)
			f = r - a  # free inside reserved
			print(f"Free memory: {f}")


	def save_training(self):
		# save info
		name = 'info_' + str(self.args.job_id) + '.json'
		with open(path.join(self.save_dir, name), 'w') as outfile:
			dump(self.train_info, outfile)

		# save mdodel 
		self.model.save_pretrained(self.save_dir)


	def start_training(self):
		self.train_info = {i+1:{'total_loss':[], 'lm_loss':[], 'mc_loss':[]} for i in range(self.args.n_epochs)}
		self.trainer = Engine(self.train)
		self.evaluator = Engine(self.inference)

		pbar = ProgressBar(persist=True)
		pbar.attach(self.trainer)

		# Linearly decrease the learning rate from lr to zero
		scheduler = PiecewiseLinear(self.optimizer, "lr", [(0, self.args.lr), (self.args.n_epochs * len(self.train_loader), 0.0)])
		self.trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

		# save the model and the information after each epoch
		self.trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: self.save_training())

		self.metrics = []

		# add the metric initialization of the metrics before testing 
		self.trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: self.new_init_metrics())
		# add test to the object trainer 
		self.trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: self.evaluator.run(self.test_loader))
		# compute the metrics
		self.trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: self.compute_f1_rec_prec_acc())

		print("\nStart the training\n")
		self.trainer.run(self.train_loader, max_epochs=self.args.n_epochs) # epoch_length=3

		self.model.save_pretrained(self.save_dir)

		# save the info of the training
		name = 'info_' + str(self.args.job_id) + '.json'
		with open(path.join(self.save_dir, name), 'w') as outfile:
			dump(self.train_info, outfile)

		# save the metrics
		with open(path.join(self.save_dir, 'metrics.json'), 'w') as outfile:
			dump(self.metrics, outfile)

		# move the slurm files error and output in the job's folder 
		if self.args.job_id != 0:
			slurm_out_name = 'slurm_output-' + str(self.args.job_id) + '.out'
			slurm_err_name = 'slurm_error-' + str(self.args.job_id) + '.err'
			rename(path.join(getcwd(), slurm_err_name), path.join(self.save_dir, slurm_err_name))
			rename(path.join(getcwd(), slurm_out_name), path.join(self.save_dir, slurm_out_name))



def load_models(model_name):
	# select which model we want to use (gpt2 or gpt) 
	model_class = GPT2DoubleHeadsModel
	# load the model
	model = model_class.from_pretrained(model_name) # './saved_model/'
	# put the model onto the available device
	model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

	tokenizer_class = GPT2Tokenizer # cant use Autotokenizer because checkpoint could be a Path
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
	def boolean_string(s):
		if s not in {'False', 'True'}:
			raise ValueError('Not a valid boolean string')
		return s == 'True'

	parser = ArgumentParser()
	parser.add_argument("--train", type=boolean_string, default=True, help="True: train the model. False: predict.")
	parser.add_argument("--job_id", type=int, default=0, help="Job ID of, used when running the code on cluster.")
	parser.add_argument("--max_len", type=int, default=50, help="Maximum length of the sentences.")
	parser.add_argument("--create_DataLoaders", type=boolean_string, default=False, help="If set to True it creates new DataLoaders.")
	parser.add_argument("--load_json", type=boolean_string, default=True, help="If set to true, it loads the prercomputed json.")
	parser.add_argument("--gradient_accumulation_steps", type=int, default=60, help="Number of gradiente accumulation steps during training.")
	parser.add_argument("--batch_size", type=int, default=30, help="Batch size.")
	parser.add_argument("--n_epochs", type=int, default=2, help="Number of training epochs.")
	parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate.")
	parser.add_argument("--perc_training", type=float, default=0.8, help="Percentage of the whole data to use for training.")
	parser.add_argument("--depressed_perc", type=float, default=0.5, help="Balance between control and positive subjects, e.g. 0.5 means half control and half positive (thee disribution of the dataset is 0.1112)")
	parser.add_argument("--max_norm", type=float, default=1, help="Clipping gradient norm")
	parser.add_argument("--pos_class", type=str, default='end_lab', help="Where to put the classification index: 'end_lab' after the label or 'end_sent' right after the sentence")

	args = parser.parse_args()

	print(args)

	assert args.pos_class in ['end_lab', 'end_sent'], f"wrong --pos_class {args.pos_class}, possible values [end_lab, end_sent]"

	ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}
	path_to_data = ["data", "depression", "all", "chunks"]

	if args.train:
		model_name = "distilgpt2"
	else:
		model_name =  path.join('./saved_model', 'job_' + str(args.job_id), '')

	# load the tokenizer 
	model, tokenizer = load_models(model_name)
	add_special_tokens_(model, tokenizer)

	optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)

	torch.manual_seed(0)

	DataProcessing(path_to_data=path_to_data, tokenizer=tokenizer, model=model, tokens=ATTR_TO_SPECIAL_TOKEN, optimizer=optimizer, args=args)
