#!/usr/bin/env python
# coding: utf-8

# <center>
# <img src="https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/main/afrisenti-logo.png" width="30%" />
# </center>

# In[1]:


import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--lang_code",
                        default='am',
                        type=str,
                        help="Valid codes: 'am', 'dz', 'ha', 'ig', 'ma', 'pcm', 'pt', 'sw', 'yo'")

args = parser.parse_args()

LANGUAGE_CODE = args.lang_code

# LANGUAGE_CODE = 'am'


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


# 
# 
# ####Starter Code: Datasets, etc
# 

# In[5]:


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
from sklearn.metrics import f1_score, balanced_accuracy_score, balanced_accuracy_score, average_precision_score
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

np.random.seed(420)
torch.manual_seed(69);


# In[6]:


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


# In[7]:


MAXIMUM_SEQUENCE_LENGTH = 500
DATA_DIR = os.path.join(TRAINING_DATA_DIR, 'splitted-train-dev-test', LANGUAGE_CODE)
EVAL_DIR = os.path.join(PROJECT_DIR, TASK, 'dev')


# In[8]:


LANGUAGE_CODE


# In[9]:


# Set seed before initializing model.
set_seed(42069)

# obtain train data
df = pd.read_csv(DATA_DIR + '/train.tsv', sep='\t')
df = df.dropna()
train_dataset = Dataset.from_pandas(df)
label_list = df['label'].unique().tolist()

# obtain dev data
df = pd.read_csv(DATA_DIR + '/dev.tsv', sep='\t') #pd.concat([, pd.read_csv(DATA_DIR + '/test.tsv', sep='\t')])
df = df.dropna()
eval_dataset = Dataset.from_pandas(df)

# obtain test data
df = pd.read_csv(DATA_DIR + '/test.tsv', sep='\t')
df = df.dropna()
test_dataset = Dataset.from_pandas(df)



# Labels
num_labels = len(label_list)
print(label_list)


# ####Tokenization

# In[10]:


tokenizer = SentencePieceBPETokenizer()
tokenizer.train_from_iterator(
    train_dataset['text'] + eval_dataset['text'],
    vocab_size=100000,
    min_frequency=5,
    show_progress=True,
    limit_alphabet=500,
)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer
)
tokenizer.add_special_tokens({'eos_token': '[EOS]'})
tokenizer.pad_token = tokenizer.eos_token


# In[11]:


# Preprocessing the datasets
# Padding strategy
padding = "max_length"


label_to_id = None
label_to_id = {v: i for i, v in enumerate(label_list)}


# In[12]:


def preprocess_function(examples):
    texts =(examples['text'],)
    result = tokenizer(*texts, padding=padding, max_length=MAXIMUM_SEQUENCE_LENGTH)
    
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

test_dataset = test_dataset.map(
    preprocess_function,
    batched=True,
    desc="Running tokenizer on validation dataset",
)


# In[13]:


train_dataset, eval_dataset, test_dataset


# In[14]:


train_text, train_labels = train_dataset['tokenized'], train_dataset['label']


# In[15]:


eval_text, eval_labels = eval_dataset['tokenized'], eval_dataset['label']


# In[16]:


test_text, test_labels = test_dataset['tokenized'], test_dataset['label']


# #### Simple Baselines
# 
# Class Proportions (AKA Constant / Majority Classifier Results)

# In[ ]:





# In[17]:


for l in np.unique(test_labels):
  print(np.mean(np.array(test_labels) == l), 
        f1_score(test_labels, [l] * len(test_labels), average='weighted'), 
        balanced_accuracy_score(test_labels, [l] * len(test_labels)))


# In[18]:


# set up unigram, bigram, trigram BOW and TF-IDF

unigram_vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=100000, min_df=7, max_df=0.8)
unigram_vectorizer.fit(train_text)
X_train_unigram = unigram_vectorizer.transform(train_text)
X_eval_unigram = unigram_vectorizer.transform(eval_text)
X_test_unigram = unigram_vectorizer.transform(test_text)

unigram_tf_idf_transformer = TfidfTransformer()
unigram_tf_idf_transformer.fit(X_train_unigram)
X_train_unigram_tf_idf = unigram_tf_idf_transformer.transform(X_train_unigram)
X_eval_unigram_tf_idf = unigram_tf_idf_transformer.transform(X_eval_unigram)
X_test_unigram_tf_idf = unigram_tf_idf_transformer.transform(X_test_unigram)

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=100000, min_df=7, max_df=0.8)
bigram_vectorizer.fit(train_text)
X_train_bigram = bigram_vectorizer.transform(train_text)
X_eval_bigram = bigram_vectorizer.transform(eval_text)
X_test_bigram = bigram_vectorizer.transform(test_text)

bigram_tf_idf_transformer = TfidfTransformer()
bigram_tf_idf_transformer.fit(X_train_bigram)
X_train_bigram_tf_idf = bigram_tf_idf_transformer.transform(X_train_bigram)
X_eval_bigram_tf_idf = bigram_tf_idf_transformer.transform(X_eval_bigram)
X_test_bigram_tf_idf = bigram_tf_idf_transformer.transform(X_test_bigram)

trigram_vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=100000, min_df=7, max_df=0.8)
trigram_vectorizer.fit(train_text)
X_train_trigram = trigram_vectorizer.transform(train_text)
X_eval_trigram = trigram_vectorizer.transform(eval_text)
X_test_trigram = trigram_vectorizer.transform(test_text)

trigram_tf_idf_transformer = TfidfTransformer()
trigram_tf_idf_transformer.fit(X_train_trigram)
X_train_trigram_tf_idf =trigram_tf_idf_transformer.transform(X_train_trigram)
X_eval_trigram_tf_idf =trigram_tf_idf_transformer.transform(X_eval_trigram)
X_test_trigram_tf_idf =trigram_tf_idf_transformer.transform(X_test_trigram)


# In[19]:


def AdaDTC(**params):
    base_params = {}
    for k, v in list(params.items()):
        if k[:16] == 'base_estimator__':
            base_params[k[16:]] = v
            del params[k]
    return AdaBoostClassifier(base_estimator=DecisionTreeClassifier(**base_params), **params)

classifiers = [
    (MultinomialNB, [
        {'alpha': 1e-3}, {'alpha': 0.1}, {'alpha': 1}, {'alpha': 10}, {'alpha': 1e3},
    ]), 
    (SVC, [
        {'C': 1, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': 1000},
        {'C': 0.1, 'gamma': 'auto', 'kernel': 'rbf', 'max_iter': 1000},
        {'C': 10, 'gamma': 'auto', 'kernel': 'rbf', 'max_iter': 1000},
        {'C': 1, 'gamma': 'scale', 'kernel': 'linear', 'max_iter': 1000},
        {'C': 0.1, 'gamma': 'auto', 'kernel': 'linear', 'max_iter': 1000},
        {'C': 10, 'gamma': 'auto', 'kernel': 'linear', 'max_iter': 1000},
    ]), 
    (LogisticRegression, [
        {'C': 0.1, 'max_iter': 1000, 'penalty': 'l2'},
        {'C': 1, 'max_iter': 1000, 'penalty': 'l2'},
        {'C': 10, 'max_iter': 1000, 'penalty': 'l2'}
    ]), 
    (RandomForestClassifier, [
        {'criterion': 'gini', 'max_depth': 5, 'n_estimators': 500},
        {'criterion': 'gini', 'max_depth': 2, 'n_estimators': 1000},
    ]), 
    (AdaDTC, [
        {'base_estimator__criterion': 'gini', 'base_estimator__max_depth': 5, 
          'base_estimator__splitter': 'best', 'learning_rate': 0.1, 'n_estimators': 500},
        {'base_estimator__criterion': 'entropy', 'base_estimator__max_depth': 3, 
         'base_estimator__splitter': 'random', 'learning_rate': 1, 'n_estimators': 500},
        {'base_estimator__criterion': 'entropy', 'base_estimator__max_depth': 2, 
         'base_estimator__splitter': 'random', 'learning_rate': 1, 'n_estimators': 1000}
    ]),  
    (MLPClassifier, [
        {'activation': 'relu', 'alpha': 1, 'hidden_layer_sizes': (100,), 'learning_rate': 'constant'},
        {'activation': 'tanh', 'alpha': 1, 'hidden_layer_sizes': (100,), 'learning_rate': 'adaptive'}
    ])
    ]

X_train_data = {'unigram counts': X_train_unigram,
                'unigram tf-idf': X_train_unigram_tf_idf,
                'bigram counts': X_train_bigram, 
                'bigram tf-idf': X_train_bigram_tf_idf,
                'trigram counts': X_train_trigram, 
                'trigram tf-idf': X_train_trigram_tf_idf}

X_eval_data = {'unigram counts': X_eval_unigram,
                'unigram tf-idf': X_eval_unigram_tf_idf,
                'bigram counts': X_eval_bigram, 
                'bigram tf-idf': X_eval_bigram_tf_idf,
                'trigram counts': X_eval_trigram, 
                'trigram tf-idf': X_eval_trigram_tf_idf}

X_test_data = {'unigram counts': X_test_unigram,
                'unigram tf-idf': X_test_unigram_tf_idf,
                'bigram counts': X_test_bigram, 
                'bigram tf-idf': X_test_bigram_tf_idf,
                'trigram counts': X_test_trigram, 
                'trigram tf-idf': X_test_trigram_tf_idf}

def train_and_show_scores(title, model, parameters, k=5):

    X_train, X_test = X_train_data[title], X_test_data[title]
    y_train, y_test = train_labels, test_labels
    X_eval, y_eval = X_eval_data[title], eval_labels
    
    
    best_train, best_test, best_f1, best_balanced_accuracy, best_params = 0, 0, 0, 0, None
    best_valid_f1 = 0
    for params in parameters:
        train_score, test_score, test_f1, test_balanced_accuracy = 0, 0, 0, 0
        valid_f1 = 0
        for _ in range(k):
            clf = model(**params)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                clf.fit(X_train, y_train)
                train_score += clf.score(X_train, y_train) / k
#                 eval_score += clf.score(X_eval, y_eval) / k
                test_score += clf.score(X_test, y_test) / k

                preds = clf.predict(X_test)
    #             print(preds, y_test)
                test_f1 += f1_score(y_test, preds, average='weighted') / k
                test_balanced_accuracy += balanced_accuracy_score(y_test, preds) / k
            
                preds = clf.predict(X_eval)
                valid_f1 += f1_score(y_eval, preds, average='weighted') / k
        
        if valid_f1 > best_valid_f1:
            best_train, best_test, best_f1, best_balanced_accuracy, best_params, best_valid_f1 =                 train_score, test_score, test_f1, test_balanced_accuracy, params, valid_f1
    
    return best_valid_f1, f"""
    {title}
    Train score: {best_train:.3}
    Eval score: {best_test:.3}
    Balanced Accuracy: {best_balanced_accuracy:.3}
    Weighted Test F1: {best_f1:.3}
    Params: {best_params}
    """


# In[20]:


# Get scores for multiple different models
for model, parameters in classifiers:
    print(model)
    best_valid, best_scores = -1, ''
    for title in ['unigram counts', 'unigram tf-idf', 'bigram counts', 'bigram tf-idf', 'trigram counts', 'trigram tf-idf']:
        valid, scores = train_and_show_scores(title, model, parameters)
        if valid > best_valid:
            best_valid, best_scores = valid, scores
    print(best_scores)


# In[ ]:


# Clear Memory
del unigram_vectorizer
del X_train_unigram 
del X_test_unigram

del unigram_tf_idf_transformer
del X_train_unigram_tf_idf
del X_test_unigram_tf_idf
                                                             
del bigram_vectorizer
del X_train_bigram
del X_test_bigram

del bigram_tf_idf_transformer
del X_train_bigram_tf_idf
del X_test_bigram_tf_idf

del trigram_vectorizer
del X_train_trigram
del X_test_trigram

del trigram_tf_idf_transformer
del X_train_trigram_tf_idf
del X_test_trigram_tf_idf

del X_train_data
del X_test_data 

del train_text
del eval_text


# In[ ]:




