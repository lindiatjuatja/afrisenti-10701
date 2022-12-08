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

def get_target_data(args, test=False):
    am_train = pd.read_csv(train_file)

    if args.lang_code != 'all':
        am_train = am_train[am_train.tag == args.lang_code]

    if args.translate_low_resource:
        am_train['text'] = am_train['en_translated']

    am_train = am_train[['text', 'labels', 'tag']]

    if test:
        return np.split(am_train.sample(frac=1, random_state=42),
         [int(.7*len(am_train)), int(.8*len(am_train))])
    else:
        return np.split(am_train.sample(frac=1, random_state=42), [int(.7*len(am_train))])

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
        return np.split(en_train.sample(frac=1, random_state=42), [int(.8*len(en_train))])
    return en_train.sample(frac=1, random_state=42)

def slot_in_adapter(args, model, adapter_loc, task_name='sa'):
    adapter = model.load_adapter(adapter_loc, config=AdapterConfig.load(args.adapter_type, reduction_factor=2))
    model.active_adapters = Stack(adapter, task_name)


def make_model(args, 
    lm=False, 
    lm_adapter=None, 
    load_head=None, 
    add_head=True,
    freeze_head=False,
    parallel=None,
    stack=None,
    task_name="sa"):

    if lm:
        model = AutoModelForMaskedLM.from_pretrained(args.base_model)
        adapter_config = AdapterConfig.load(args.adapter_type)
        model.add_adapter(task_name, config=adapter_config)
        model.train_adapter([task_name])
        model.set_active_adapters(task_name)
        return model

    adapter_config = AdapterConfig.load(
        args.adapter_type, reduction_factor=2 
        if lm_adapter is not None and args.finetune_style == 'stack' 
        else 1)

    model = AutoAdapterModel.from_pretrained(args.base_model)
    model.add_adapter(task_name, config=adapter_config)
    model.train_adapter(task_name)
    model.set_active_adapters(task_name)

    if add_head:
        if load_head:
            model.load_head(load_head)
            if freeze_head:
                for p in model.heads[task_name].parameters():
                    p.requires_grad = False
                    p.train = False
        else:
            model.add_classification_head(task_name, num_labels=3)

    if lm_adapter:
            
        if args.finetune_style == 'stack':
            slot_in_adapter(args, model, lm_adapter, task_name)
        if args.finetune_style == 'load':
            model.delete_adapter(task_name)
            loaded = model.add_adapter(lm_adapter, 
            config=AdapterConfig.load(args.adapter_type, reduction_factor=2))
            model.train_adapter(loaded)
            model.set_active_adapters(loaded)
    elif parallel is not None:

        src_adapter, tgt_adapter = parallel
        src_adapter = model.load_adapter(src_adapter, config=adapter_config)
        tgt_adapter = model.load_adapter(tgt_adapter, config=adapter_config)

        model.active_adapters = Parallel(
            Stack(src_adapter, task_name), 
            Stack(tgt_adapter, task_name))
    elif stack is not None:
        first, second = stack
        model.delete_adapter(task_name)
        first = model.load_adapter(first, config=adapter_config)
        second = model.load_adapter(second, config=adapter_config)
        model.active_adapters = Stack(first, second)
        model.train_adapter([first, second])

    return model

