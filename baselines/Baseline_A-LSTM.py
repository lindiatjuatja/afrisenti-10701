#!/usr/bin/env python
# coding: utf-8

# <center>
# <img src="https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/main/afrisenti-logo.png" width="30%" />
# </center>

# In[1]:


import argparse
import os

# parser = argparse.ArgumentParser()
# parser.add_argument("--lang_code",
#                         default='am',
#                         type=str,
#                         help="Valid codes: 'am', 'dz', 'ha', 'ig', 'ma', 'pcm', 'pt', 'sw', 'yo'")

# args = parser.parse_args()

# LANGUAGE_CODE = args.lang_code

LANGUAGE_CODE = 'am'


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
from tokenizers import SentencePieceBPETokenizer, ByteLevelBPETokenizer
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
df = pd.read_csv(DATA_DIR + '/dev.tsv', sep='\t')
df = df.dropna()
eval_dataset = Dataset.from_pandas(df)
label_list = df['label'].unique().tolist()

df = pd.read_csv(DATA_DIR + '/test.tsv', sep='\t')
df = df.dropna()
test_dataset = Dataset.from_pandas(df)

# Labels
num_labels = len(label_list)
print(label_list)


# ####Tokenization

# In[10]:


tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(
    train_dataset['text'] + eval_dataset['text'],
    vocab_size=100000,
    min_frequency=5,
    show_progress=True,
#     limit_alphabet=500,
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


train_dataset, eval_dataset


# In[14]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[15]:


class LSTM(nn.Module):
    def __init__(self, hidden_dim=128, emb_dim=300, num_layers=1, dropout=0.5, lstm_dropout=0.0):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(len(tokenizer), emb_dim)
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=emb_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=lstm_dropout)
        
        self.drop = nn.Dropout(p=dropout)

        self.fc = nn.Linear(2*hidden_dim, 3)

    def forward(self, text, text_len):

        text_emb = self.embedding(text)

        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), text_len - 1, :self.hidden_dim]
        out_reverse = output[:, 0, self.hidden_dim:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = text_fea
        return text_out


# In[16]:


num_pts = len(train_dataset)
# shuffled_ids = np.arange(num_pts, dtype=int)
# np.random.shuffle(shuffled_ids)

valid_ids = torch.LongTensor(np.array(eval_dataset['input_ids']))
valid_lengths = torch.LongTensor(np.array(eval_dataset['length'])).cpu()
valid_labels = torch.LongTensor(np.array(eval_dataset['label']))

train_ids = torch.LongTensor(np.array(train_dataset['input_ids']))
train_lengths = torch.LongTensor(np.array(train_dataset['length'])).cpu()
train_labels = torch.LongTensor(np.array(train_dataset['label']))

eval_ids = torch.LongTensor(test_dataset['input_ids'])
eval_lengths = torch.LongTensor(test_dataset['length']).cpu()
eval_labels = torch.LongTensor(test_dataset['label'])

len(valid_ids), len(train_ids), len(eval_ids)


# In[19]:


def train(criterion = nn.CrossEntropyLoss(),
          batch_size = 32,
          num_epochs = 30,
          eval_every = 4,
          params={},
          lr=0.0005,
          leave=False):
    model = LSTM(**params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    eval_every = (len(train_ids) / batch_size) // eval_every
    best_valid_acc = 0
    best_test_f1 = 0
    best_test_balanced_acc = 0
    model.train()
    best_preds = []
    for epoch in range(num_epochs):
        pbar = tqdm(range(0, len(train_ids), batch_size), leave=leave, desc=f'Epoch {epoch+1}/{num_epochs}')
        total_train_loss = 0.0
        total_points = 0
        for i in pbar:           
            labels = train_labels[i:i+batch_size].to(device)
            inps = train_ids[i:i+batch_size].to(device)
            lengths = train_lengths[i:i+batch_size]#.to(device)
            
            optimizer.zero_grad()
            output = model(inps, lengths)
            loss = criterion(output, labels)
            total_train_loss += loss.item()
            total_points += labels.size(0)
            loss.backward()
            optimizer.step()

            if (i + batch_size) % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    num_correct = 0
                    eval_batch_size = 100
                    preds = np.array([])
                    for j in range(0, len(valid_ids), eval_batch_size):
                        labels = valid_labels[j:j+eval_batch_size].to(device)
                        inps = valid_ids[j:j+eval_batch_size].to(device)
                        lengths = valid_lengths[j:j+eval_batch_size]#.to(device)
                        output = model(inps, lengths)
                        output = torch.argmax(output, -1)
                        preds = np.append(preds, output.cpu().numpy())
#                         num_correct += torch.sum(output == labels).cpu().numpy()

                    valid_accuracy = balanced_accuracy_score(valid_labels, preds)

                    accuracy = "N/A"
                    preds = np.array([])
                    if valid_accuracy > best_valid_acc:
                        num_correct = 0
                        eval_batch_size = 100
                        for j in range(0, len(eval_ids), eval_batch_size):
                            labels = eval_labels[j:j+eval_batch_size].to(device)
                            inps = eval_ids[j:j+eval_batch_size].to(device)
                            lengths = eval_lengths[j:j+eval_batch_size]#.to(device)
                            output = model(inps, lengths)
                            output = torch.argmax(output, -1)
                            preds = np.append(preds, output.cpu().numpy())
                            num_correct += torch.sum(output == labels).cpu().numpy()
#                         accuracy = num_correct / len(eval_ids)
                        best_test_f1 = f1_score(eval_labels, preds, average='weighted')
                        best_test_balanced_acc = balanced_accuracy_score(eval_labels, preds)
                        best_preds = preds
                        best_valid_acc = valid_accuracy
                    pbar.set_description(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {(total_train_loss / total_points):.3} ' +                                          f'Curent Bal Val Acc: {valid_accuracy:.3}, Bal Test Acc @ Best Ever Val {best_test_balanced_acc:.3}, Test F1: {best_test_f1:.3}')
                model.train()
    return best_valid_acc, best_test_f1, best_test_balanced_acc, preds, eval_labels




# acc, f1, preds, labels = train(num_epochs=15, params={
#     'hidden_dim': 128, 'emb_dim': 300, 'num_layers': 2, 'dropout': 0.0, 'lstm_dropout': 0.5},
#       lr=0.001, leave=True)


# In[ ]:


dropout = 0.0
lstm_dropout = 0.5

best_val, best_f1, best_bal_ac, best_params = 0, 0, 0, {}
for hidden_dim in [200, 300]:
    for emb_dim in [300, 500]:
        for num_layers in [2, 3]:
            for lr in [1e-3, 5e-4, 1e-4]:
                my_f1, my_balanced_acc, my_val = 0, 0, 0
                params = { 'hidden_dim': hidden_dim, 'emb_dim': emb_dim, 
                    'num_layers': num_layers, 'dropout': dropout, 'lstm_dropout': lstm_dropout}
                
                k = 3
                for i in range(k):
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter('always')
                        acc, f1, balanced_acc, preds, labels = train(num_epochs=15, params=params, lr=lr, leave=False)
                        my_f1 += f1 / k
                        my_balanced_acc += balanced_acc / k
                        my_val += acc / k
                        
                if my_val > best_val:
                    params['lr'] = lr
                    best_val, best_f1, best_bal_ac, best_params = my_val, my_f1, balanced_acc, params
                    print(f"balanced acc: {balanced_acc:.3}, f1: {my_f1:.3}, params: {params}")
# best_f1, best_params          

print(f'''
Best Weighted F1: {best_f1}
Best Balanced Accuracy: {best_bal_ac}
Params: {best_params}
''')


# In[ ]:



del train_ids
del train_lengths
del train_labels

del eval_ids
del eval_lengths
del eval_labels

