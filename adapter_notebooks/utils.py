import pandas as pd
import numpy as np

from transformers import (
    AutoModelForMaskedLM,
    AutoAdapterModel,
)
from transformers.adapters.configuration import AdapterConfig
from transformers.adapters.composition import Stack, Parallel

train_file = 'data/multilingual_combined.csv'
en_file = 'data/en_all.csv'

def oversample(df):
    size = df['labels'].value_counts().max()
    dfs = [df]
    for i, group in df.groupby('labels'):
        dfs.append(group.sample(size-len(group), replace=True))
    return pd.concat(dfs)

def get_target_data(args, test=False, lang_code=None):
    am_train = pd.read_csv(train_file)

    if lang_code is None:
        lang_code = args.lang_code

    if lang_code != 'all':
        am_train = am_train[am_train.tag == lang_code]

    if args.translate_low_resource:
        am_train['text'] = am_train['en_translated']

    am_train = am_train[['text', 'labels', 'tag']]

    if test:
        train, dev, test = np.split(am_train.sample(frac=1, random_state=42),
         [int(.7*len(am_train)), int(.8*len(am_train))])
        if args.oversample:
            train = oversample(train)
        return train.sample(frac=1, random_state=42), dev, test
    else:
        train, test = np.split(am_train.sample(frac=1, random_state=42), [int(.7*len(am_train))])
        if args.oversample:
            train = oversample(train)
        return train.sample(frac=1, random_state=42), test

def get_source_data(args, dev=True):
    if args.source_lang_code == 'en':
        en_train = pd.read_csv(en_file)
        if args.translate_en_to_lang:
            assert args.lang_code in ['am', 'ig', 'ha', 'sw', 'yo'], "English data cannot be translated to this language"
            en_train['text'] = en_train[args.lang_code]
    else:
        en_train = pd.read_csv(train_file)
        if args.source_lang_code == 'all':
            en_train = en_train[en_train.tag != args.lang_code]
        else:
            en_train = en_train[en_train.tag == args.source_lang_code]

    en_train = en_train[['text', 'labels', 'tag']]

    if dev:
        train, dev = np.split(en_train.sample(frac=1, random_state=42), [int(.8*len(en_train))])
        if args.oversample:
            train = oversample(train)
        return train.sample(frac=1, random_state=42), dev
    if args.oversample:
        en_train = oversample(en_train)
    return en_train.sample(frac=1, random_state=42)


