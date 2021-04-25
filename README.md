# Master-Thesis-GPT2

## Installation
To install and use the training and inference scripts please install the requirements:
```bash
pip install -r requirements.txt
```

## Fine-tuning the model

The training script can be used in single GPU or on the CPU (not recommended, too slow)

```bash
python train_predict.py
```

The training script accept several arguments to tweak the training:

Argument | Type | Default value | Description
---------|------|---------------|------------
train | `str` | `'True'` | True: train the model. False: predict
job_id | `int` | `0` | Job ID of, used when running the code on cluster
max_len | `int` | `50` | Maximum length of the sentences (excluding the special tokens)
create_DataLoaders | `str` | `False` | If set to True it creates new DataLoaders
load_json | `str` | `True` | If set to true, it loads the prercomputed json vih the data
gradient_accumulation_steps | `int` | `40` |  Accumulate gradients on several steps during the training to overcome the small match size due to small GPU memory
batch_size | `int` | `2` | Batch size for validation
n_epochs | `int` | `8` | Number of training epochs
lr | `float` | `6.25e-5` | Learning rate
perc_training | `float` | `0.8` | Percentage of the whole data to use for training
depressed_perc | `float` | `0.5` | Balance between control and positive subjects, e.g. 0.5 means half control and half positive
max_norm | `float` | `1.0` | Clipping gradient norm

