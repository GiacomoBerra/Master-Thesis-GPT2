# plot the output of the model

import matplotlib.pyplot as plt
from json import load 
from argparse import ArgumentParser
from collections import Counter
from os import path
from json import load
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd 
import numpy as np

parser = ArgumentParser()
parser.add_argument("--job_id", type=int, default=0, help="Job ID of, used when running the code on cluster.")
args = parser.parse_args()

print(args)
name = 'info_' + str(args.job_id) + '.json'
with open(path.join('saved_model', 'job_' + str(args.job_id), name), 'r') as json_file:
	data = load(json_file)

with open(path.join('saved_model', 'job_' + str(args.job_id), 'config.json'), 'r') as json_file:
	json = load(json_file)

with open(path.join('saved_model', 'job_' + str(args.job_id), 'metrics.json'), 'r') as json_file:
		metrics = load(json_file)

model_name = json['architectures'][0]

print("Results of the model {}.".format(model_name))


if model_name == "GPT2LMHeadModel":
	train_loss =  data['loss'] 
	train_loss_df = pd.DataFrame([train_loss]).T
	col_names = ['train_loss']
	train_loss_df.rename(columns = {i:j for i, j in enumerate(col_names)}, inplace = True)

	test_loss = []
	test_perplexity = []

	for dict_ in metrics:
		test_loss += dict_['lm_loss']
		test_perplexity += dict_['lm_perplexity']

	test_losses_df = pd.DataFrame([test_loss, test_perplexity]).T
	col_names = ['test_loss', 'test_perplexity']
	test_losses_df.rename(columns = {i:j for i, j in enumerate(col_names)}, inplace = True)

	b = round(train_loss_df.shape[0]/len(metrics))
	a = round(test_losses_df.shape[0]/len(metrics))

	fig, axes = plt.subplots(3, 1, figsize=(15,8.7))
	axe = axes.ravel()

	# add the title to the whole plot
	fig.suptitle('Job ' + str(args.job_id))

	axe[0].title.set_text('Test Perplexity')
	test_losses_df['test_perplexity'].plot(ax=axe[0], markevery=a, marker='o', markerfacecolor='red')
	axe[1].title.set_text('Test Loss')
	test_losses_df['test_loss'].plot(ax=axe[1], markevery=a, marker='o', markerfacecolor='red')
	axe[2].title.set_text('Train Loss')
	train_loss_df['train_loss'].plot(ax=axe[2], markevery=b, marker='o', markerfacecolor='red')
	plt.show()

	'''
	loss = data['loss']

	print(data['iteration'])

	plt.figure(0)
	plt.plot(loss)
	plt.title('Plot of the loss')
	plt.ylabel('loss values')
	plt.xlabel('Iteration')
	plt.show()
	'''

else:
	#with open(path.join('saved_model', 'job_' + str(args.job_id), 'metrics.json'), 'r') as json_file:
	#	metrics = load(json_file)

	if args.job_id <= 1307291:

		for (epoch_data, values_data), (epoch_metrics, values_metrics) in zip(data.items(), metrics.items()):

			df_values_data = pd.DataFrame(values_data)
			df_values_metrics = pd.DataFrame(values_metrics)
			fig, axes = plt.subplots(2, 3, figsize=(12,8))
			axe = axes.ravel()

			# add the title to the whole plot
			fig.suptitle('Results of the training and testiong after the epoch {}'.format(epoch_data))

			axe[0].title.set_text('Aggregated Loss (Train)')
			df_values_data['total_loss'].plot(ax=axe[0])
			axe[1].title.set_text('Language Modeling Loss (Train)')
			df_values_data['lm_loss'].plot(ax=axe[1])
			axe[2].title.set_text('Classification Loss (Train)')
			df_values_data['mc_loss'].plot(ax=axe[2])

			axe[3].title.set_text('Language Modeling Loss (Test)')
			df_values_metrics['lm_loss'].plot(ax=axe[3])
			axe[4].title.set_text('Language Modeling Perplexity (Test)')
			df_values_metrics['lm_perplexity'].plot(ax=axe[4])
			axe[5].title.set_text('Classification Accuracy (Test)')
			df_values_metrics['mc_accuracy'].plot(ax=axe[5])
			plt.show()

	elif args.job_id > 1307291:
		with open(path.join('saved_model', 'job_' + str(args.job_id), 'metrics.json'), 'r') as json_file:
			metrics = load(json_file)

		# load the losses on the training dataset
		with open(path.join('saved_model', 'job_' + str(args.job_id), 'info_' + str(args.job_id) + '.json'), 'r') as json_file:
			train_losses = load(json_file)


		lm_loss = []
		lm_perplexity = []
		mc_accuracy = []
		precision_NonDep = []
		precision_Dep = []
		recall_NonDep = []
		recall_Dep = []
		f1_NonDep = []
		f1_Dep = []

		# iterate through all the result of every evalution and every time skip the first two metrics, i.e. loss and perplexity
		for i, dict_ in enumerate(metrics):
			#lm_loss.append([dict_['lm_loss']])
			#lm_perplexity.append([dict_['lm_perplexity']])
			mc_accuracy += dict_['mc_accuracy']
			precision_NonDep += dict_['precision_0']
			precision_Dep += dict_['precision_1']
			recall_NonDep += dict_['recall_0']
			recall_Dep += dict_['recall_1']
			f1_NonDep += dict_['f1_0']
			f1_Dep += dict_['f1_1']

		lm_loss_df = pd.DataFrame([i['lm_loss'] for i in metrics]).T
		lm_perplexity_df = pd.DataFrame([i['lm_perplexity'] for i in metrics]).T
		metrics_df = pd.DataFrame([mc_accuracy, precision_NonDep, precision_Dep, recall_NonDep, recall_Dep, f1_NonDep, f1_Dep]).T
		col_names = ['mc_accuracy', 'precision_NonDep', 'precision_Dep', 'recall_NonDep', 'recall_Dep', 'f1_NonDep', 'f1_Dep']
		metrics_df.rename(columns = {i:j for i, j in enumerate(col_names)}, inplace = True)

		'''
		fig, axes = plt.subplots(3, 3, figsize=(15,8.7))
		axe = axes.ravel()

		# add the title to the whole plot
		fig.suptitle('Job ' + str(args.job_id))
		#fig.suptitle('Metrics of the Classification of the model evaluation')

		axe[0].title.set_text('Accuracy (Test)')
		metrics_df['mc_accuracy'].plot(ax=axe[0])
		axe[1].title.set_text('Precision for the Non Depressed label (Test)')
		metrics_df['precision_NonDep'].plot(ax=axe[1])
		axe[2].title.set_text('Precision for the Depressed label (Test)')
		metrics_df['precision_Dep'].plot(ax=axe[2])
		axe[3].title.set_text('Recall for the Non Depressed label (Test)')
		metrics_df['recall_NonDep'].plot(ax=axe[3])
		axe[4].title.set_text('Recall for the Depressed label (Test)')
		metrics_df['recall_Dep'].plot(ax=axe[4])
		axe[5].title.set_text('F1 for the Non Depressed label (Test)')
		metrics_df['f1_NonDep'].plot(ax=axe[5])
		axe[6].title.set_text('F1 for the Depressed label (Test)')
		metrics_df['f1_Dep'].plot(ax=axe[6])
		plt.show()
		'''
		def set_ticks(axes):
			axes.xaxis.set_major_locator(plt.MaxNLocator(12))
			labels = [item.get_text() for item in axes.get_xticklabels()]
			labels = labels[:1] + [str(i)+"%" for i in range(0,110,10)]
			return labels

		plt.figure(figsize=(15,8.7))
		G = gridspec.GridSpec(4, 2)

		axes_1 = plt.subplot(G[0, :])
		axes_1.title.set_text('Accuracy (Test)')
		metrics_df['mc_accuracy'].plot(ax=axes_1)
		axes_1.set_xticklabels(set_ticks(axes_1))
		axes_1.set_xlabel('% of the training')

		axes_2 = plt.subplot(G[1, 0])
		axes_2.title.set_text('Precision for the Non Depressed label (Test)')
		metrics_df['precision_NonDep'].plot(ax=axes_2)
		axes_2.set_xticklabels(set_ticks(axes_2))

		axes_3 = plt.subplot(G[1, 1])
		axes_3.title.set_text('Precision for the Depressed label (Test)')
		metrics_df['precision_Dep'].plot(ax=axes_3)
		axes_3.set_xticklabels(set_ticks(axes_3))

		axes_4 = plt.subplot(G[2, 0])
		axes_4.title.set_text('Recall for the Non Depressed label (Test)')
		metrics_df['recall_NonDep'].plot(ax=axes_4)
		axes_4.set_xticklabels(set_ticks(axes_4))

		axes_5 = plt.subplot(G[2, 1])
		axes_5.title.set_text('Recall for the Depressed label (Test)')
		metrics_df['recall_Dep'].plot(ax=axes_5)
		axes_5.set_xticklabels(set_ticks(axes_5))

		axes_6 = plt.subplot(G[3, 0])
		axes_6.title.set_text('F1 for the Non Depressed label (Test)')
		metrics_df['f1_NonDep'].plot(ax=axes_6)
		axes_6.set_xticklabels(set_ticks(axes_6))
		axes_6.set_xlabel('% of the training')

		axes_7 = plt.subplot(G[3, 1])
		axes_7.title.set_text('Recall for the Depressed label (Test)')
		metrics_df['f1_Dep'].plot(ax=axes_7)
		axes_7.set_xticklabels(set_ticks(axes_7))
		axes_7.set_xlabel('% of the training')

		plt.tight_layout()
		plt.show()



		# plot thr loss and the perplexity
		new_lm_loss_df = []
		new_lm_perplexity_df = []
		for i in range(lm_loss_df.shape[1]):
			new_lm_loss_df += list(lm_loss_df[i])
			new_lm_perplexity_df += list(lm_perplexity_df[i])


		df_ = pd.DataFrame([new_lm_loss_df, new_lm_perplexity_df]).T
		col_names = ['lm_loss', 'lm_perp']
		df_.rename(columns = {i:j for i, j in enumerate(col_names)}, inplace = True)

		df_train = pd.DataFrame(train_losses["1"]["lm_loss"])
		df_train.rename(columns = {0:"lm_loss"}, inplace = True)


		fig, axes = plt.subplots(3, 1, figsize=(12,8.7))
		axe = axes.ravel()

		a = int(len(new_lm_loss_df)/lm_loss_df.shape[1])-1
		b = round(df_train.shape[0]/lm_loss_df.shape[1])-1

		# add the title to the whole plot
		fig.suptitle('Job ' + str(args.job_id))

		axe[0].title.set_text('Language Modeling Perplexity (Test)')
		df_['lm_perp'].plot(ax=axe[0], markevery=a, marker='o', markerfacecolor='red') #
		axe[0].set_ylabel('Perplexity')
		axe[1].title.set_text('Language Modeling Loss (Test)')
		df_['lm_loss'].plot(ax=axe[1], markevery=a, marker='o', markerfacecolor='red')
		axe[1].set_ylabel('Loss')
		axe[2].title.set_text('Language Modeling Loss (Train)')
		df_train['lm_loss'].plot(ax=axe[2], markevery=b, marker='o', markerfacecolor='red')
		axe[2].set_ylabel('Loss')
		fig.text(0.5, 0.06, 'Iteration', ha='center')
		#fig.text(0.08, 0.5, '', va='center', rotation='vertical')
		plt.show()

		
		
















