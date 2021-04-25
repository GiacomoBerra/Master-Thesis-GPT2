# Analysis on the sentences

from os import path, getcwd, listdir, rename, makedirs
from json import dump, load
from tqdm import tqdm
import re
from statistics import mean

from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

if 0:
	# load all the sentences
	with open(path.join('data_json.json'), 'r') as json_file:
		data = load(json_file)

	# extraxt all the words in for each flag (depressed and non depressed)
	words = {'non depressed': {'pronominalisation':[], 'readiness_to_action':[], 'punctation':[]},\
			 'depressed': {'pronominalisation':[], 'readiness_to_action':[], 'punctation':[]}}
	for flag, persons in data.items():
		# for all the subjects name
		n_pron = 0
		n_noun = 0
		n_verb = 0
		n_sent = 0
		n_punct = 0
		for _, history in tqdm(persons.items()):
			for sent in history:
				c = sent.split(' ')
				match_list = re.findall(r"\b([i|I][\\]*?\'[a-zA-Z]+)", sent, re.IGNORECASE | re.DOTALL)
				# if there is a pronound "i" lower make it capital
				if 'i' in c or 'i' in [x[0] for x in match_list]:
					for i, w in enumerate(c):
						# if the pronoun is alone
						if w == 'i':
							c[i] = c[i].upper()
						# if the pronoun is within a contraction
						elif re.search(r"\b([i|I][\\]*?\'[a-zA-Z]+)", w, re.IGNORECASE | re.DOTALL) is not None:
							c_split = list(w)
							c_split[0] = c_split[0].upper()
							c[i] = ''.join(c_split)
						# detect also the cases in which there is a "went.i was..."

					sent = ' '.join(c)

				# tokenize and tag each word
				tag = pos_tag(word_tokenize(sent))
				# count all the pronouns
				n_pron += sum([1 for x in tag if x[1][:3]=='PRP'])
				# count all the nouns
				n_noun += sum([1 for x in tag if x[1][:2]=='NN'])
				# count all the verbs
				n_verb += sum([1 for x in tag if x[1][:2]=='VB'])
				# count all the punctation
				n_punct += len(re.findall(r"(!+|\.+|\?+|,+|:+|;+|\'|\*+|\(+|\)+|\[+|\]+|/+|\(+|-+|\(+|–+|\"+)", sent, re.IGNORECASE | re.DOTALL))
				# count the number of sentences
				n_sent += 1
				

			words[flag]['pronominalisation'] += [n_pron/n_noun]
			words[flag]['readiness_to_action'] += [n_verb/n_noun]
			words[flag]['punctation'] += [n_punct/n_sent]

	print('pronominalisation')
	print(f"\t non dep: {mean(words['non depressed']['pronominalisation'])}")
	print(f"\t dep: {mean(words['depressed']['pronominalisation'])}")
	print('readiness_to_action')
	print(f"\t non dep: {mean(words['non depressed']['readiness_to_action'])}")
	print(f"\t dep: {mean(words['depressed']['readiness_to_action'])}")
	print('punctation')
	print(f"\t non dep: {mean(words['non depressed']['punctation'])}")
	print(f"\t dep: {mean(words['depressed']['punctation'])}")

	# load all the sentences
	with open(path.join('sentences_anal.json'), 'w') as json_file:
		dump(words, json_file)


with open(path.join('sentences_anal.json'), 'r') as json_file:
		data = load(json_file)

import matplotlib.pyplot as plt
import pandas as pd 
from pandas.plotting import table
import numpy as np

dep = pd.DataFrame(data['depressed'])
non_dep = pd.DataFrame(data['non depressed'])

for name in ['pronominalisation', 'readiness_to_action', 'punctation']: 
	fig, axes = plt.subplots(3, 2, figsize=(12,8))
	axe = axes.ravel()

	if name == 'pronominalisation':
		fig.suptitle(r'Distribution on the Pronominalisation $\left(\frac{n_{pron}}{n_{noun}}\right)$')
		#table(axe[2], np.round(dep[name].describe(), 3), loc="upper right", colWidths=[0.3]);
		#table(axe[3], np.round(non_dep[name].describe(), 3), loc="upper right", colWidths=[0.3]);
	elif name == 'readiness_to_action':
		fig.suptitle(r'Distribution on the Readiness to action $\left(\frac{n_{verb}}{n_{noun}}\right)$')
		#table(axe[2], np.round(dep[name].describe(), 3), loc="upper right", colWidths=[0.3]);
		#table(axe[3], np.round(non_dep[name].describe(), 3), loc="upper right", colWidths=[0.3]);
	else:
		fig.suptitle(r'Distribution on the Punctation $\left(\frac{n_{punct}}{n_{sent}}\right)$')
		#table(axe[2], np.round(dep[name].describe(), 3), loc="right", colWidths=[0.2]);
		#table(axe[3], np.round(non_dep[name].describe(), 3), loc="right", colWidths=[0.2]);

	axe[0].title.set_text('Positive subjects')
	dep[name].hist(bins=100, ax=axe[0])
	axe[1].title.set_text('Control subjects')
	non_dep[name].hist(bins=100, ax=axe[1])
	dep[name].plot(kind='density', ax=axe[2])
	non_dep[name].plot(kind='density', ax=axe[3])

	dep.boxplot(column=[name], ax=axe[4], vert=False, rot=90, showmeans=True, meanline=True, showfliers=False)
	axe[4].set_yticklabels([])
	non_dep.boxplot(column=[name], ax=axe[5], vert=False, rot=90, showmeans=True, meanline=True, showfliers=False)
	axe[5].set_yticklabels([])
	plt.show()
























































