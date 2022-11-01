#!/usr/bin/env python
# coding: utf-8

# <center>
# <img src="https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/main/afrisenti-logo.png" width="30%" />
# </center>

# In[1]:


import os
import pandas as pd
import numpy as np

languages = ['am', 'dz', 'ha', 'ig', 'ma', 'pcm', 'pt', 'sw', 'yo']

folder = ''

colab = False

if colab:
    from google.colab import drive
    drive.mount('/content/drive')
    proj_folder = '/content/drive/MyDrive'
else:
    proj_folder = os.getcwd()


# In[2]:


get_ipython().run_line_magic('cd', '{proj_folder}')


PROJECT_DIR = f'{proj_folder}/afrisent-semeval-2023'
PROJECT_GITHUB_URL = 'https://github.com/afrisenti-semeval/afrisent-semeval-2023.git'

if not os.path.isdir(PROJECT_DIR):
  get_ipython().system('git clone {PROJECT_GITHUB_URL}')
else:
  get_ipython().run_line_magic('cd', '{PROJECT_DIR}')
  get_ipython().system('git pull {PROJECT_GITHUB_URL}')


# In[3]:


get_ipython().run_cell_magic('capture', '', '\n%cd {PROJECT_DIR}\n\nif os.path.isdir(PROJECT_DIR):\n  #The requirements file should be in PROJECT_DIR\n  if os.path.isfile(os.path.join(PROJECT_DIR, \'starter_kit/requirements.txt\')):\n    !pip install -r starter_kit/requirements.txt\n  else:\n    print(\'requirements.txt file not found\')\n\nelse:\n  print("Project directory not found, please check again.")')


# In[4]:


# Training Data Paths

TASK2 = 'SubtaskA'
TRAINING_DATA_DIR = os.path.join(PROJECT_DIR, TASK2, 'train')
FORMATTED_TRAIN_DATA = os.path.join(TRAINING_DATA_DIR, 'formatted-train-data')

if os.path.isdir(TRAINING_DATA_DIR):
  print('Data directory found.')
  if not os.path.isdir(FORMATTED_TRAIN_DATA):
    print('Creating directory to store formatted data.')
    os.mkdir(FORMATTED_TRAIN_DATA)
else:
  print(TRAINING_DATA_DIR + ' is not a valid directory or does not exist!')


# In[5]:


get_ipython().run_line_magic('cd', '{TRAINING_DATA_DIR}')

training_files = os.listdir()

if len(training_files) > 0:
  for training_file in training_files:
    if training_file.endswith('.tsv'):

      data = training_file.split('_')[0]
      if not os.path.isdir(os.path.join(FORMATTED_TRAIN_DATA, data)):
        print(data, 'Creating directory to store train, dev and test splits.')
        os.mkdir(os.path.join(FORMATTED_TRAIN_DATA, data))
      
      df = pd.read_csv(training_file, sep='\t', names=['ID', 'text', 'label'], header=0)
      df[['text', 'label']].to_csv(os.path.join(FORMATTED_TRAIN_DATA, data, 'train.tsv'), sep='\t', index=False)
    else:
      print(training_file + ' skipped!')
else:
  print('No files are found in this directory!')


# In[6]:


if os.path.isdir(FORMATTED_TRAIN_DATA):
  print('Data directory found.')
  SPLITTED_DATA = os.path.join(TRAINING_DATA_DIR, 'splitted-train-dev-test')
  if not os.path.isdir(SPLITTED_DATA):
    print('Creating directory to store train, dev and test splits.')
    os.mkdir(SPLITTED_DATA)
else:
  print(FORMATTED_TRAIN_DATA + ' is not a valid directory or does not exist!')

get_ipython().run_line_magic('cd', '{FORMATTED_TRAIN_DATA}')
formatted_training_files = os.listdir()

if len(formatted_training_files) > 0:
  for data_name in formatted_training_files:
    formatted_training_file = os.path.join(data_name, 'train.tsv')
    if os.path.isfile(formatted_training_file):
      labeled_tweets = pd.read_csv(formatted_training_file, sep='\t', names=['text', 'label'], header=0)
      train, dev, test = np.split(labeled_tweets.sample(frac=1, random_state=42), [int(.7*len(labeled_tweets)), int(.8*len(labeled_tweets))])

      if not os.path.isdir(os.path.join(SPLITTED_DATA, data_name)):
        print(data_name, 'Creating directory to store train, dev and test splits.')
        os.mkdir(os.path.join(SPLITTED_DATA, data_name))

      train.sample(frac=1).to_csv(os.path.join(SPLITTED_DATA, data_name, 'train.tsv'), sep='\t', index=False)
      dev.sample(frac=1).to_csv(os.path.join(SPLITTED_DATA, data_name, 'dev.tsv'), sep='\t', index=False)
      test.sample(frac=1).to_csv(os.path.join(SPLITTED_DATA, data_name,'test.tsv'), sep='\t', index=False)
    else:
      print(training_file + ' is not a supported file!')
else:
  print('No files are found in this directory!')

get_ipython().run_line_magic('cd', '{PROJECT_DIR}')


# In[7]:


# Training Data Paths

TASK2 = 'SubtaskB'
TRAINING_DATA_DIR = os.path.join(PROJECT_DIR, TASK2)
FORMATTED_TRAIN_DATA = os.path.join(TRAINING_DATA_DIR, 'formatted-train-data')

if os.path.isdir(TRAINING_DATA_DIR):
  print('Data directory found.')
  if not os.path.isdir(FORMATTED_TRAIN_DATA):
    print('Creating directory to store formatted data.')
    os.mkdir(FORMATTED_TRAIN_DATA)
else:
  print(TRAINING_DATA_DIR + ' is not a valid directory or does not exist!')


# In[8]:


get_ipython().run_line_magic('cd', '{TRAINING_DATA_DIR}')

training_files = os.listdir()

if len(training_files) > 0:
  for training_file in training_files:
    if training_file.endswith('train.tsv'):

      data = training_file.split('_')[0]
      if not os.path.isdir(os.path.join(FORMATTED_TRAIN_DATA, data)):
        print(data, 'Creating directory to store train, dev and test splits.')
        os.mkdir(os.path.join(FORMATTED_TRAIN_DATA, data))
      
      df = pd.read_csv(training_file, sep='\t', names=['ID', 'text', 'label'], header=0)
      df[['text', 'label']].to_csv(os.path.join(FORMATTED_TRAIN_DATA, data, 'train.tsv'), sep='\t', index=False)
    else:
      print(training_file + ' skipped!')
else:
  print('No files are found in this directory!')


# In[9]:


if os.path.isdir(FORMATTED_TRAIN_DATA):
  print('Data directory found.')
  SPLITTED_DATA = os.path.join(TRAINING_DATA_DIR, 'splitted-train-dev-test')
  if not os.path.isdir(SPLITTED_DATA):
    print('Creating directory to store train, dev and test splits.')
    os.mkdir(SPLITTED_DATA)
else:
  print(FORMATTED_TRAIN_DATA + ' is not a valid directory or does not exist!')

get_ipython().run_line_magic('cd', '{FORMATTED_TRAIN_DATA}')
formatted_training_files = os.listdir()

if len(formatted_training_files) > 0:
  for data_name in formatted_training_files:
    formatted_training_file = os.path.join(data_name, 'train.tsv')
    if os.path.isfile(formatted_training_file):
      labeled_tweets = pd.read_csv(formatted_training_file, sep='\t', names=['text', 'label'], header=0)
      train, dev, test = np.split(labeled_tweets.sample(frac=1, random_state=42), [int(.7*len(labeled_tweets)), int(.8*len(labeled_tweets))])

      if not os.path.isdir(os.path.join(SPLITTED_DATA, data_name)):
        print(data_name, 'Creating directory to store train, dev and test splits.')
        os.mkdir(os.path.join(SPLITTED_DATA, data_name))

      train.sample(frac=1).to_csv(os.path.join(SPLITTED_DATA, data_name, 'train.tsv'), sep='\t', index=False)
      dev.sample(frac=1).to_csv(os.path.join(SPLITTED_DATA, data_name, 'dev.tsv'), sep='\t', index=False)
      test.sample(frac=1).to_csv(os.path.join(SPLITTED_DATA, data_name,'test.tsv'), sep='\t', index=False)
    else:
      print(training_file + ' is not a supported file!')
else:
  print('No files are found in this directory!')

get_ipython().run_line_magic('cd', '{proj_folder}')

