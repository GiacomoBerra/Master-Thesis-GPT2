# Data processing 



from os import path, getcwd, listdir
from pandas import read_csv
from pandas import DataFrame as df
from re import findall
import xml.etree.ElementTree as ET
from time import sleep as sl
from json import dump
from re import sub
from nltk import tokenize
from random import sample
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

from transformers import (AdamW, GPT2LMHeadModel, OpenAIGPTTokenizer,
                                  OpenAIGPTLMHeadModel, GPT2Tokenizer)


class DataProcessing():

	def __init__(self, tokenizer, model, tokens=["<bos>", "<eos>", "<pad>"], path_to_data=[], perc_training=0.8, max_len=280, generate_data=False):
		self.tokenizer = tokenizer
		self.tokens = tokens
		self.model = model
		self.max_len = max_len
		self.perc_training = perc_training

		if generate_data == True:
			self.data = self.GetAllSubjectsNumber(path_to_data)
			self.createDataset()
		else:
			self.load_DataLoaders()

		self.train()


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

		with open(path.join(base_dir, 'data_json.json'), 'w') as outfile:
			dump(data_dict, outfile)

		return data_dict


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
		max_l = max(len(x) for y in dataset.values() for x in y)
		for name in dataset.keys():
			dataset[name] = [x + [padding] * (max_l - len(x)) for x in dataset[name]]
		return dataset

	def count_removed_senteces(self, dataset):
		# compute the length of each sentences
		train_sent_train = [len(x) for y in dataset.values() for x in y]
		# count the frequency of each sentences' lengths
		count = Counter(train_sent_train)

		sum_ = 0
		for k, v in count.items():
			if k <= self.max_len:
				sum_ += v

		print(f"Number of sentences with a lengths >= {self.max_len} are: {len(train_sent_train) - sum_}")
		

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
					# add the tokens
					for i, s in enumerate(sents):
						new_sent = flag + ' ' + self.tokens['bos_token'] + ' ' +  s + ' ' + self.tokens['eos_token']
						sents[i] = new_sent
						#dataset[flag] += [self.tokenizer.convert_tokens_to_ids(new_sent.split(' '))]
						dataset_non_tokenized[flag] += [new_sent]

					n_sent[flag] += len(sents)


		print(f"Total number of sentences of depressed subjects: {n_sent['depressed']}")
		print(f"Total number of sentences of non depressed subjects: {n_sent['non depressed']}\n\n")

		print(dataset_non_tokenized['depressed'][45])
		print(self.tokenizer.convert_tokens_to_ids(dataset_non_tokenized['depressed'][45].split(' ')))
		print(dataset_non_tokenized['non depressed'][45])
		print(self.tokenizer.convert_tokens_to_ids(dataset_non_tokenized['non depressed'][45].split(' ')))

		print(f"Tokenize the dataset")
		train_dep = list(sample(dataset_non_tokenized['depressed'], round(n_sent['depressed']*self.perc_training)))
		train_non_dep = list(sample(dataset_non_tokenized['non depressed'], round(n_sent['non depressed']*self.perc_training)))

		train_data = train_dep + train_non_dep
		test_data = list(set(dataset_non_tokenized['depressed']) - set(train_dep)) + list(set(dataset_non_tokenized['non depressed']) - set(train_non_dep))

		self.count_removed_senteces({"train":train_data,"test":test_data})

		# tokenize the sentences and set the maximim lengths to max_len
		dataset_tokenized = {"train": [], "test": []}
		for sent in train_data:
			sentence = self.tokenizer.convert_tokens_to_ids(sent.split(' '))
			if len(sentence) <= self.max_len:
				dataset_tokenized["train"].append(sentence)
		for sent in test_data:
			sentence = self.tokenizer.convert_tokens_to_ids(sent.split(' '))
			if len(sentence) <= self.max_len:
				dataset_tokenized["test"].append(sentence)

		print(f"Padding the dataset")
		self.dataset_tokenized = self.pad_dataset(dataset_tokenized, padding=tokenizer.convert_tokens_to_ids(ATTR_TO_SPECIAL_TOKEN['pad_token']))

		self.dataset_tokenized["train"] = torch.tensor(self.dataset_tokenized["train"]) 
		self.dataset_tokenized["test"] = torch.tensor(self.dataset_tokenized["test"]) 

		# create dataloaders
		train_dataset = TensorDataset(self.dataset_tokenized["train"])
		test_dataset = TensorDataset(self.dataset_tokenized["test"])
		self.train_loader = DataLoader(train_dataset, sampler=None, batch_size=5, shuffle=False)
		self.test_loader = DataLoader(test_dataset, sampler=None, batch_size=5, shuffle=False)

		# save the dataloaders
		torch.save(self.train_loader, 'train_loader.pth')
		torch.save(self.test_loader, 'test_loader.pth')


	def train(self):
		optimizer = AdamW(model.parameters(), lr=0.001, correct_bias=True)
		for i in self.train_loader:
			a = model(i[0], labels=i[0])
			loss = a.loss
			print(loss)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			optimizer.zero_grad()




		






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
	ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>'}
	path_to_data = ["data", "depression", "all", "chunks"]
	max_len = 300
	generate_data = False

	# load the tokenizer 
	model, tokenizer = load_models("gpt2")
	add_special_tokens_(model, tokenizer)

	DataProcessing(path_to_data=path_to_data, tokenizer=tokenizer, model=model, tokens=ATTR_TO_SPECIAL_TOKEN, max_len=max_len, generate_data=generate_data)

