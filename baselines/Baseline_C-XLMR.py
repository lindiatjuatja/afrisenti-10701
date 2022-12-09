#!/usr/bin/env python
# coding: utf-8

# <center>
# <img src="https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/main/afrisenti-logo.png" width="30%" />
# </center>

# In[19]:


import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--lang_code",
                        default='am',
                        type=str,
                        help="TRAINED on this data, evaluated on all. Valid codes: 'am', 'dz', 'ha', 'ig', 'ma', 'pcm', 'pt', 'sw', 'yo'")
parser.add_argument("--use_en", 
                        action="store_true",
                        help="Enable to use english data for zero shot rather than the original language codes")
parser.add_argument("--seed",
                        default=42069,
                        type=int,
                        help="Random seed")

args = parser.parse_args()

LANGUAGE_CODE = args.lang_code
USE_EN = args.use_en

# LANGUAGE_CODE = 'am'
# USE_EN = True


# In[2]:


print("Language Code: ", LANGUAGE_CODE)


# In[3]:


import pandas as pd
import numpy as np

# Please don not edit anything here
languages = ['am', 'dz', 'ha', 'ig', 'ma', 'pcm', 'pt', 'sw', 'yo']

colab = False


TASK = 'SubtaskA'


# In[4]:


# import os

# if colab:
#     from google.colab import drive
#     drive.mount('/content/drive')
#     proj_folder = '/content/drive/MyDrive'
# else:
#     proj_folder = os.getcwd()

# %cd {proj_folder}


# PROJECT_DIR = f'{proj_folder}/afrisent-semeval-2023'
# if not os.path.isdir(PROJECT_DIR):
#   %run Make_Datasets.py


# In[5]:


# MODEL_NAME_OR_PATH = 'Davlan/afro-xlmr-mini'
MODEL_NAME_OR_PATH = 'xlm-roberta-base'
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 5e-5
NUMBER_OF_TRAINING_EPOCHS = 5
MAXIMUM_SEQUENCE_LENGTH = 128
SAVE_STEPS = -1


# 
# 
# ####Starter Code: Datasets, etc
# 

# In[6]:


from types import SimpleNamespace
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import warnings
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim

import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
import pandas
import pandas as pd
from datasets import load_dataset

import evaluate
import transformers
from transformers.trainer_callback import ProgressCallback
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    PreTrainedTokenizerFast,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from tokenizers import SentencePieceBPETokenizer
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from datasets import Features, Value, ClassLabel, load_dataset, Dataset

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
set_seed(args.seed)


# In[7]:


# folder = ''


# if colab:
#     from google.colab import drive
#     drive.mount('/content/drive')
#     proj_folder = '/content/drive/MyDrive'
# else:
#     proj_folder = os.getcwd()

# %cd {proj_folder}


# PROJECT_DIR = f'{proj_folder}/afrisent-semeval-2023'


PROJECT_DIR = 'afrisent-semeval-2023'

TRAINING_DATA_DIR = os.path.join(PROJECT_DIR, TASK, 'train')
FORMATTED_TRAIN_DATA = os.path.join(TRAINING_DATA_DIR, 'formatted-train-data')

TRAINING_DATA_DIR


# In[8]:


DATA_DIR = os.path.join(TRAINING_DATA_DIR, 'splitted-train-dev-test', LANGUAGE_CODE)


# In[9]:


LANGUAGE_CODE


# In[11]:


# Set seed before initializing model.



# In[23]:



# obtain train data
if USE_EN:
    print('Using EN for zero shot')
    df = pd.read_csv('../adapter_notebooks/data/en_all.csv')[['text', 'labels']].rename(
        columns={'labels':'label'})
    
    X_train, X_test = train_test_split(df, test_size=.3)
    eval_dataset = Dataset.from_pandas(X_test)
    label_list = df['label'].unique().tolist()
    train_dataset = Dataset.from_pandas(X_train)
    
    # Labels
    num_labels = len(label_list)
    print(label_list)
else:
    
    
    # obtain dev data
    df = pd.concat([pd.read_csv(DATA_DIR + '/dev.tsv', sep='\t'), pd.read_csv(DATA_DIR + '/test.tsv', sep='\t')])
    df = df.dropna()
    eval_dataset = Dataset.from_pandas(df)
    label_list = df['label'].unique().tolist()
    print('dev data:', eval_dataset)

    # Labels
    num_labels = len(label_list)
    print(label_list)

    df = pd.read_csv(DATA_DIR + '/train.tsv', sep='\t')
    df = df.dropna()
    eval_dataset = Dataset.from_pandas(df)
    train_dataset = Dataset.from_pandas(df)
    print('train data:', train_dataset)

# # Labels
# num_labels = len(label_list)
# print(label_list)

# ####Tokenization

# In[9]:


config = AutoConfig.from_pretrained(
    MODEL_NAME_OR_PATH,
    num_labels=num_labels,
    cache_dir=None,
    revision='main',
    use_auth_token=None,
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME_OR_PATH,
    do_lower_case=None,
    cache_dir=None,
    use_fast=True,
    revision='main',
    use_auth_token=None,
)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME_OR_PATH,
    from_tf=False,
    config=config,
    cache_dir=None,
    revision="main",
    use_auth_token=None,
    ignore_mismatched_sizes=False,
)


# In[10]:


# Preprocessing the datasets
# Padding strategy
padding = "max_length"


label_to_id = None
label_to_id = {v: i for i, v in enumerate(label_list)}


# In[11]:


def preprocess_function(examples):
    texts =(examples['text'],)
    result = tokenizer(*texts, padding=padding, max_length=MAXIMUM_SEQUENCE_LENGTH, truncation=True)
    
    if label_to_id is not None and "label" in examples:
        result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
    
    result['length'], result["tokenized"] = [], []
    for input_ids in result['input_ids']:
        toks = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)
        result['length'].append(len(toks)+2)
        result['tokenized'].append(' '.join(toks))
    return result

train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
#     desc="Running tokenizer on train dataset",
)

eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
#     desc="Running tokenizer on validation dataset",
)


# In[12]:


train_dataset, eval_dataset


# In[13]:


train_text, train_labels = train_dataset['tokenized'], train_dataset['label']


# In[14]:


eval_text, eval_labels = eval_dataset['tokenized'], eval_dataset['label']


# In[15]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[16]:


# Get the metric function
metric = evaluate.load("accuracy")

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    metrics = metric.compute(predictions=preds, references=p.label_ids)
    metrics['f1'] = f1_score(p.label_ids, preds, average='weighted')
    metrics['bal_acc'] = balanced_accuracy_score(p.label_ids, preds)
    return metrics


data_collator = default_data_collator

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args = TrainingArguments(
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUMBER_OF_TRAINING_EPOCHS,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        output_dir='tmp_trainer',
        save_steps=SAVE_STEPS,
        overwrite_output_dir=True
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
# trainer.remove_callback(ProgressCallback)
# Training

train_result = trainer.train(resume_from_checkpoint=None)
metrics = train_result.metrics
metrics["train_samples"] = len(train_dataset) 


# Evaluation

metrics = trainer.evaluate(eval_dataset=eval_dataset)

metrics["eval_samples"] = len(eval_dataset)

print('metrics on dev set')
for key,value in metrics.items():
    print(key, ':', value)

# splitted_A = os.path.join(PROJECT_DIR, 'SubtaskA', 'train', 'splitted-train-dev-test')

# try:
#     LANGUAGE_CODE
# except NameError:
#     LANGUAGE_CODE = 'combined'
# else:
#     pass
for lang in languages:

    lang_data_dir = os.path.join(TRAINING_DATA_DIR, 'splitted-train-dev-test', lang)
    print('Testing on', lang, 'dev + test set')
    lang_test = pd.concat([pd.read_csv(lang_data_dir + '/dev.tsv', sep='\t'), pd.read_csv(lang_data_dir + '/test.tsv', sep='\t')])
    lang_test = Dataset.from_pandas(lang_test)
    lang_test = lang_test.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=False,
        desc="Running tokenizer on validation dataset",
    )
    predictions, labels, metrics = trainer.predict(lang_test, metric_key_prefix="eval")
    f1 = f1_score(labels, np.argmax(predictions, axis=1), average='weighted')
    bal_acc = balanced_accuracy_score( labels, np.argmax(predictions, axis=1))
    print(f'{lang} results:      F1: {f1:.3}     balanced accuracy: {bal_acc:.3}')





