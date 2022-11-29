import pandas as pd
from transformers import AutoAdapterModel, AdapterConfig, AutoTokenizer, TrainingArguments, AdapterTrainer, EvalPrediction, set_seed
import torch
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score
from transformers.adapters.composition import Stack
import warnings
from train_lm import train_wiki_lm_and_save
import gc
from transformers.trainer_callback import PrinterCallback, ProgressCallback

import argparse

def main():
    warnings.filterwarnings(action='ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file",
                            default='data/am/am_train_translated.csv',
                            type=str,
                            help="Location of the training data for low resource language")
    parser.add_argument("--en_file",
                            default='data/en_all.csv',
                            type=str,
                            help="Location of the training data for English language")
    parser.add_argument("--lang_code",
                            default='am',
                            type=str,
                            help="Code for the target low resource language")
    parser.add_argument("--base_model",
                            default='xlm-roberta-base',
                            type=str,
                            help="Base Transformer model")
    parser.add_argument("--adapter_type",
                            default='pfeiffer',
                            type=str,
                            help="Base Adapter Type")
    parser.add_argument("--tmp_folder",
                            default='tmp/',
                            type=str,
                            help="Folder in which to save temporary components")
    parser.add_argument("--show_bar", action="store_true")

    parser.add_argument("--use_en_train", action="store_true",
                            help="Whether to use the English dataset (in English) for training")
    parser.add_argument("--use_en_translated", action="store_true",
                            help="Whether to use the translated English dataset (auto translated into the lang_code variable) for training (only for single task)")
    parser.add_argument("--frozen_en_head", action="store_true",
                            help="Whether to train a head in English, then use the frozen head")
    parser.add_argument("--translate_low_resource", action="store_true",
                            help="Use the low resource language translated into English")


    parser.add_argument("--train_lm", action="store_true",
                            help="Train an LM using Wikipedia data")
    parser.add_argument("--lm_zero_shot", action="store_true",
                            help="See results for zero shot ")
    parser.add_argument("--lm_date",
                            default='20221120',
                            type=str,
                            help="Date for Wikipedia")
    parser.add_argument("--lm_lr", default=1e-4, type=float,
                            help="Learning rate for LM adapter training")
    parser.add_argument("--lm_epochs", default=10, type=int,
                            help="Epochs for LM adapter training")
    parser.add_argument("--lm_gradient_accumulation_steps", default=4, type=int,
                            help="Gradient accumulation steps for LM training")
    parser.add_argument("--lm_per_device_batch_size", default=2, type=int,
                            help="Batch size for device")
    parser.add_argument("--mlm_probability", default=0.15, type=float,
                            help="Mask probability for MLM")


    parser.add_argument("--lr", default=1e-4, type=float,
                            help="Learning rate for adapter training")
    parser.add_argument("--train_epochs", default=6, type=int,
                            help="Epochs for adapter training")
    parser.add_argument("--per_device_batch_size", default=32, type=int,
                            help="Batch size for device")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                            help="What it sounds like")
    parser.add_argument("--seed", default=42069, type=int,
                            help="Random seed. Set to -1 for random")


    args = parser.parse_args()

    for arg in vars(args):
        print(f"{arg} = {getattr(args, arg)}")
    print('\n', '='*50, '\n')

    if args.seed != -1:
        torch.manual_seed(args.seed)
        set_seed(args.seed)
        np.random.seed(args.seed)


    if args.train_lm or args.lm_zero_shot:
        lm_adapter_location = train_wiki_lm_and_save(args)
        lang_adapter_config = AdapterConfig.load(args.adapter_type, reduction_factor=2)


    label2id = {"positive":0, "neutral":1, 'negative':2}
    id2label = {0:"positive", 1:"neutral", 2:'negative'}

    training_args = TrainingArguments(
        args.tmp_folder,
        learning_rate=args.lr,
        num_train_epochs=args.train_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=200
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def encode_batch(row):
        if type(row.text) != str:
            return -1
        text = ' '.join(filter(lambda x:x[0]!='@', row.text.split()))
        out = tokenizer(text, max_length=100, truncation=True, padding="max_length", return_tensors='pt')
        out['labels'] = torch.LongTensor([label2id[row.labels]])[0]
        return out

    test_split_lengths = []
    def compute_scores(p: EvalPrediction, test_split_lengths=test_split_lengths):
        preds = np.argmax(p.predictions, axis=1)
        i, output = 0, dict()
        for name, split_length in test_split_lengths:
            s = np.s_[i:i+split_length]
            split_preds = preds[s]
            split_labels = p.label_ids[s]
            output[f'{name}_acc'] = (split_preds==split_labels).mean()
            output[f'{name}_weighted_f1'] = f1_score(split_preds, split_labels, average='weighted')
            output[f'{name}_balanced_accurancy'] = balanced_accuracy_score(split_preds, split_labels)
            i += split_length
        return output

    am_train = pd.read_csv(args.train_file)
    am_train, am_dev = np.split(
        am_train.sample(frac=1, random_state=42), [int(.7*len(am_train))])

    combined_train, combined_test = [], []

    if args.translate_low_resource:
        combined_train.append(am_train[['eng_translated', 'label']].rename(columns={'eng_translated':'text', 'label':'labels'}))
        combined_test.append(am_dev[['eng_translated', 'label']].rename(columns={'eng_translated':'text', 'label':'labels'}))
    else:
        combined_train.append(am_train[['tweet', 'label']].rename(columns={'tweet':'text', 'label':'labels'}))
        combined_test.append(am_dev[['tweet', 'label']].rename(columns={'tweet':'text', 'label':'labels'}))
    test_split_lengths.append(('am_dev', len(am_dev)))


    gc.collect()

    if args.use_en_train or args.use_en_translated or args.frozen_en_head or args.lm_zero_shot:
        en_train = pd.read_csv(args.en_file)
        en_train, en_test = np.split(en_train.sample(frac=1, random_state=42), [int(.8*len(en_train))])

        if args.frozen_en_head or args.lm_zero_shot:
            # train an en model, and save it somewhere. Supports frozen classification heads or LMs
            en_freeze_train = en_train[['text', 'labels']].apply(encode_batch, axis=1).reset_index()[0]
            en_freeze_test = en_test[['text', 'labels']].apply(encode_batch, axis=1).reset_index()[0]

            freeze_model = AutoAdapterModel.from_pretrained(args.base_model)
            adapter_config = AdapterConfig.load(args.adapter_type)
            freeze_model.add_adapter("sa", config=adapter_config)
            freeze_model.train_adapter("sa")

            if args.lm_zero_shot:
                en = freeze_model.load_adapter("en/wiki@ukp", config=lang_adapter_config)
                task_adapter = freeze_model.load_adapter(lm_adapter_location, config=lang_adapter_config)
                freeze_model.add_classification_head("sa", num_labels=3)
                freeze_model.set_active_adapters("sa")
                freeze_model.active_adapters = Stack(en, "sa")
            elif args.frozen_en_head:
                freeze_model.add_classification_head("sa", num_labels=3)
                freeze_model.set_active_adapters("sa")
            else:
                assert False, "How did you get this error"

            freeze_trainer = AdapterTrainer(
                model=freeze_model,
                args=training_args,
                train_dataset=en_freeze_train,
                eval_dataset=en_freeze_test,
                compute_metrics=lambda p: compute_scores(p, [('en_test', len(en_freeze_test))]),
            )
            if not args.show_bar:
                freeze_trainer.remove_callback(ProgressCallback)
            freeze_trainer.train()
            print('EN performance')
            for key,value in freeze_trainer.evaluate().items():
                print(key, ':', value)
            print('\n', '='*50, '\n')

            if args.lm_zero_shot:
                freeze_model.active_adapters = Stack(task_adapter, "sa")
                eval_trainer = AdapterTrainer(
                    model=freeze_model,
                    eval_dataset=combined_test[0].apply(encode_batch, axis=1),
                    compute_metrics=compute_scores
                )
                if not args.show_bar:
                    eval_trainer.remove_callback(ProgressCallback)
                print(f'Zero Shot performance on {args.lang_code}:')
                
                for key, value in eval_trainer.evaluate().items():
                    print(key, ':', value)
                print('\n', '='*50, '\n')

            freeze_model.save_head(args.tmp_folder+'head', 'sa')

            del en_freeze_train
            del en_freeze_test
            del freeze_model
            del freeze_trainer
            gc.collect()

        elif args.use_en_train:
            combined_train.append(en_train[['text', 'labels']])
            combined_test.append(en_test[['text', 'labels']])
        elif args.use_en_translated:
            combined_train.append(en_train[[args.lang_code, 'labels']].rename(columns={args.lang_code:'text'}))
            combined_test.append(en_test[[args.lang_code, 'labels']].rename(columns={args.lang_code:'text'}))
        test_split_lengths.append(('en_test', len(en_test)))
        del en_train
        del en_test


    combined_train = pd.concat(combined_train)
    combined_test = pd.concat(combined_test)

    train = combined_train.apply(encode_batch, axis=1)
    test = combined_test.apply(encode_batch, axis=1)
    train = train[train != -1].reset_index()[0]
    test = test[test != -1].reset_index()[0]

    del am_train
    del am_dev
    del combined_train
    del combined_test
    gc.collect()


    print(f"{len(train)} training samples, {len(test)} test samples")

    model = AutoAdapterModel.from_pretrained(args.base_model)
    adapter_config = AdapterConfig.load(args.adapter_type)

    if args.train_lm:
        lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
        task_adapter = model.load_adapter(lm_adapter_location, config=lang_adapter_config)
        model.add_adapter("sa", config=lang_adapter_config)
        model.train_adapter("sa")
    else:
        model.add_adapter("sa", config=adapter_config)
        model.train_adapter("sa")

    if args.frozen_en_head:
        model.load_head(args.tmp_folder+'head/')
        model.set_active_adapters("sa")
        for p in model.heads['sa'].parameters():
            p.requires_grad = False
            p.train = False
    else:
        model.add_classification_head("sa", num_labels=3)
        model.set_active_adapters("sa")

    if args.train_lm:
        model.active_adapters = Stack(task_adapter, "sa")

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        compute_metrics=compute_scores,
    )
    if not args.show_bar:
        trainer.remove_callback(ProgressCallback)

    trainer.train()
    print('Final evaluation data')


    for key,value in trainer.evaluate().items():
        print(key, ':', value)
    
    for _ in range(10):
        print()

if __name__ == '__main__':
    main()
