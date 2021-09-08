# Extract 500 sentences of depressed and non depressed subjects

from json import dump, load
from os import path
from random import shuffle, seed
from tqdm import tqdm

# load precomputed json
with open(path.join('Data', 'data_json.json'), 'r') as json_file:
	data = load(json_file)


dataset = {"depressed": [], "non depressed": []}
review_len = []
# for all the flags ("depressed" and "non depressed")
for flag, persons in data.items():
	# for all the subjects name
	for per_id, history in tqdm(persons.items()):
		# for all the comments of a subject
		for sent in history:
			# keep only the sentences with more than 5 words
			splitted_sent = sent.split(' ')
			if len(splitted_sent) >= 5:
				dataset[flag].append(' '.join(splitted_sent[:5]))

seed(30)
shuffle(dataset["depressed"])
seed(30)
shuffle(dataset["non depressed"])

data_testing_gen = {"depressed": dataset["depressed"][:500], "non depressed": dataset["non depressed"][:500]}

# save the sentences
with open(path.join('Data', 'data_testing_gen.json'), 'w') as json_file:
	dump(data_testing_gen, json_file)
