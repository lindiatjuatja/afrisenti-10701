#!/usr/bin/env python
# coding: utf-8

# <center>
# <img src="https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/main/afrisenti-logo.png" width="30%" />
# </center>

# In[1]:


# MODEL_NAME_OR_PATH = 'Davlan/afro-xlmr-mini'
MODEL_NAME_OR_PATH = 'xlm-roberta-base'
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 5e-5
NUMBER_OF_TRAINING_EPOCHS = 5
MAXIMUM_SEQUENCE_LENGTH = 128
SAVE_STEPS = -1


# In[2]:


# Please don not edit anything here
languages = ['am', 'dz', 'ha', 'ig', 'ma', 'pcm', 'pt', 'sw', 'yo']

colab = False


TASK = 'SubtaskB'


# In[ ]:


import os

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


# In[3]:


from types import SimpleNamespace
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import warnings
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, balanced_accuracy_score
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim

import logging
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
    PreTrainedTokenizer,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
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

logger = logging.getLogger(__name__)


with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")


np.random.seed(420)
torch.manual_seed(69);


# In[4]:


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

TRAINING_DATA_DIR = os.path.join(PROJECT_DIR, TASK)
FORMATTED_TRAIN_DATA = os.path.join(TRAINING_DATA_DIR, 'formatted-train-data')

TRAINING_DATA_DIR


# In[5]:

DATA_DIR = os.path.join(TRAINING_DATA_DIR, 'splitted-train-dev-test', 'multilingual')
EVAL_DIR = os.path.join(PROJECT_DIR, TASK, 'dev')


# In[6]:


# Set seed before initializing model.
set_seed(42069)

# obtain train data
df = pd.read_csv(DATA_DIR + '/train.tsv', sep='\t')
df = df.dropna()
train_dataset = Dataset.from_pandas(df)
label_list = df['label'].unique().tolist()

# obtain dev data
df = pd.concat([pd.read_csv(DATA_DIR + '/dev.tsv', sep='\t'), pd.read_csv(DATA_DIR + '/test.tsv', sep='\t')])
df = df.dropna()
eval_dataset = Dataset.from_pandas(df)
label_list = df['label'].unique().tolist()

# Labels
num_labels = len(label_list)
print(label_list)


# ####Tokenization

# In[7]:


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


# In[8]:


# Preprocessing the datasets
# Padding strategy
padding = "max_length"


label_to_id = None
label_to_id = {v: i for i, v in enumerate(label_list)}


# In[9]:


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
    desc="Running tokenizer on train dataset",
)

eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    desc="Running tokenizer on validation dataset",
)


# In[10]:


train_dataset, eval_dataset


# In[11]:


train_text, train_labels = train_dataset['tokenized'], train_dataset['label']


# In[12]:


eval_text, eval_labels = eval_dataset['tokenized'], eval_dataset['label']


# In[13]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[14]:


# Get the metric function
metric = evaluate.load("accuracy")

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return metric.compute(predictions=preds, references=p.label_ids)


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
trainer.remove_callback(ProgressCallback)

# Training

    
train_result = trainer.train(resume_from_checkpoint=None)

metrics = train_result.metrics
metrics["train_samples"] = len(train_dataset) 

# Evaluation

metrics = trainer.evaluate(eval_dataset=eval_dataset)

metrics["eval_samples"] = len(eval_dataset)

for key,value in metrics.items():
    print(key, ':', value)

splitted_A = os.path.join(PROJECT_DIR, 'SubtaskA', 'train', 'splitted-train-dev-test')

try:
    LANGUAGE_CODE
except NameError:
    LANGUAGE_CODE = 'combined'
else:
    pass

data = []
f1, bal_acc = 0, 0
for lang in languages:
    eval_path = os.path.join(splitted_A, lang)
    df = pd.read_csv(eval_path + '/dev.tsv', sep='\t')
    df = df.dropna()
    lang_eval = Dataset.from_pandas(df)
    lang_eval = lang_eval.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
        desc="Running tokenizer on validation dataset",
    )

    predictions, labels, metrics = trainer.predict(lang_eval, metric_key_prefix="eval")

    if LANGUAGE_CODE == lang:
        f1 = (f1_score(labels, np.argmax(predictions, axis=1), average='weighted'))
        bal_acc = balanced_accuracy_score(labels, np.argmax(predictions, axis=1))

    data.append([LANGUAGE_CODE, lang, str(list(predictions)), str(list(labels))])
    gc.collect()
df = pd.DataFrame(data, columns=['source', 'target', 'predictions', 'labels'])
df.to_csv(f'{LANGUAGE_CODE}_preds.csv', index=False)


trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
print(f"f1 score: {f1:.3}, balanced acc: {bal_acc:.3}")


# In[15]:


# import os
# get_ipython().run_line_magic('cd', '{PROJECT_DIR}')

# kinya = 'jean-paul/KinyaBERT-small'

# DATA_DIR = os.path.join(TRAINING_DATA_DIR, 'splitted-train-dev-test', 'multilingual')
# OUTPUT_DIR = os.path.join(PROJECT_DIR, 'models', 'multilingual')

# get_ipython().system("python starter_kit/run_textclass.py   --model_name_or_path {kinya}   --data_dir {DATA_DIR}   --do_train   --do_eval   --do_predict   --per_device_train_batch_size {BATCH_SIZE}   --learning_rate {MAXIMUM_SEQUENCE_LENGTH}   --num_train_epochs {NUMBER_OF_TRAINING_EPOCHS}   --max_seq_length {MAXIMUM_SEQUENCE_LENGTH}   --output_dir {'tmp_trainer'}   --save_steps {SAVE_STEPS}   --overwrite_output_dir")


# In[ ]:




