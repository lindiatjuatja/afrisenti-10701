from transformers import AutoAdapterModel, AdapterConfig, AutoTokenizer, TrainingArguments, AdapterTrainer, EvalPrediction, set_seed
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score
from transformers.adapters.composition import Stack
import warnings
from train_lm import train_wiki_lm_and_save
import gc
from transformers.trainer_callback import ProgressCallback
from train_da import run_da_experiment

from utils import get_source_data, get_target_data, make_model, slot_in_adapter

import argparse

def main():
    warnings.filterwarnings(action='ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_code",
                            default='am',
                            type=str,
                            help="Code for the target low resource language. set to 'all' to run on all african data")
    parser.add_argument("--source_lang_code",
                            default='en',
                            type=str,
                            help="Code for the source language. set to 'all' to run on all data except whatever the lang_code is")
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

    parser.add_argument("--use_source_train", action="store_true",
                            help="Whether to use the source dataset (default English) for training")
    parser.add_argument("--translate_en_to_lang", action="store_true",
                            help="Whether to use the English dataset auto translated into the lang_code variable for training (only for single task)")
    parser.add_argument("--freeze_head", action="store_true",
                            help="Whether to train a head in source language, then use the frozen head")
    parser.add_argument("--translate_low_resource", action="store_true",
                            help="Use the low resource language translated into English")

    parser.add_argument("--train_da", action="store_true",
                            help="Use an unsupervised domain adaptation method")
    parser.add_argument("--da_method", default='dann', type=str,
                            help="Which unsupervised domain adaptation method to use. Supports dann, adda, cdan, coral, dc (Domain Confusion), gan, itl")
    parser.add_argument("--da_Ch", default=256, type=int,
                            help="Hidden layer size for classifier")
    parser.add_argument("--da_Dh", default=256, type=int,
                            help="Hidden layer size for discriminator")
    parser.add_argument("--da_lr", default=1e-4, type=float,
                            help="Learning rate for discrimintor and classifier")
    parser.add_argument("--finetune_style", default='stack', type=str,
                            help="How to fine-tune after training LM or DA. Options: stack, load, False")

    parser.add_argument("--train_lm", action="store_true",
                            help="Train an LM using Wikipedia data")
    # parser.add_argument("--lm_adapter_type", default='pfeiffer+inv',
    #                         type=str,
    #                         help="Adapter Type for LM")
    parser.add_argument("--lm_zero_shot", action="store_true",
                            help="See results for zero shot ")
    parser.add_argument("--lm_src_data", nargs=2, default=(None, None),
                            help='language code and date for wikipedia dataset for source: i.e. --lm_src_data sw 20221120')
    parser.add_argument("--lm_tgt_data", nargs=2, default=('am', '20221120'),
                            help='language code and date for wikipedia dataset for target: i.e. --lm_src_data am 20221120')
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
    parser.add_argument("--lm_zero_src_lm_adapter", default="en/wiki@ukp", type=str,
                            help="Name or location of LM adapter. Leave as 'train' to train a new one")
    parser.add_argument("--lm_zero_tgt_lm_adapter", default="train", type=str,
                            help="Name or location of LM adapter. Leave as 'train' to train a new one")

    # parser.add_argument("--load_adapter", default='', type=str, 
    #                         help="location of an additional adapter for use in the model. cannot be used with LM.")
    parser.add_argument("--lr", default=1e-5, type=float,
                            help="Learning rate for adapter training")
    parser.add_argument("--train_epochs", default=14, type=int,
                            help="Epochs for adapter training")
    parser.add_argument("--per_device_batch_size", default=16, type=int,
                            help="Batch size for device")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                            help="What it sounds like")
    parser.add_argument("--seed", default=42069, type=int,
                            help="Random seed. Set to -1 for random")


    args = parser.parse_args()

    available_langs = ['am', 'dz', 'ha', 'ig', 'ma', 'pcm', 'pt', 'sw', 'yo', 'all']
    assert args.lang_code in available_langs

    for arg in vars(args):
        print(f"{arg} = {getattr(args, arg)}")
    print('\n', '='*50, '\n')

    if args.seed != -1:
        torch.manual_seed(args.seed)
        set_seed(args.seed)
        np.random.seed(args.seed)

    if args.train_lm or args.lm_zero_shot:
        if args.lm_zero_src_lm_adapter == 'train':
            args.lm_zero_src_lm_adapter = train_wiki_lm_and_save(
                args, args.lang_code+'src_lm_adapter', *args.lm_src_data)

        if args.lm_zero_tgt_lm_adapter == 'train':
            args.lm_zero_tgt_lm_adapter = train_wiki_lm_and_save(
                args, args.lang_code+'tgt_lm_adapter', *args.lm_tgt_data)


    label2id = {"positive":0, "neutral":1, 'negative':2}
    id2label = {0:"positive", 1:"neutral", 2:'negative'}

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    use_model = None

    def encode_batch(row):
        if type(row.text) != str:
            return -1
        text = ' '.join(filter(lambda x:x[0]!='@', row.text.split()))
        out = tokenizer(text, max_length=100, truncation=True, padding="max_length", return_tensors='pt')
        out['labels'] = torch.LongTensor([label2id[row.labels]])[0]
        return out

    if args.train_da:
        da_loc = run_da_experiment(args, encode_batch, args.train_lm,
        args.lm_zero_src_lm_adapter, args.lm_zero_tgt_lm_adapter)

        args.load_adapter = da_loc
        if args.finetune_style not in ['stack', 'load']:
            return
    
    training_args = TrainingArguments(
        args.tmp_folder,
        learning_rate=args.lr,
        num_train_epochs=args.train_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=500,
        save_steps=-1,
        seed=args.seed
    )


    test_split_lengths = []
    def compute_scores(p: EvalPrediction, test_split_lengths=test_split_lengths):
        preds = np.argmax(p.predictions, axis=1)
        i, output = 0, dict()
        for name, split_length in test_split_lengths:
            s = np.s_[i:i+split_length]
            split_preds = preds[s]
            split_labels = p.label_ids[s]
            output[f'{name}_acc'] = (split_preds==split_labels).mean()
            output[f'{name}_weighted_f1'] = f1_score(split_labels, split_preds, average='weighted')
            output[f'{name}_balanced_accurancy'] = balanced_accuracy_score(split_labels, split_preds)
            i += split_length
        return output

    am_train, am_dev = get_target_data(args, test=False)

    combined_train, combined_test = [am_train], [am_dev]
    print(f'loaded {len(am_train)} target train samples and {len(am_dev)} target test samples')
    test_split_lengths.append((f'{args.lang_code}_dev', len(am_dev)))

    gc.collect()

    if args.use_source_train or args.translate_en_to_lang or args.freeze_head or args.lm_zero_shot:

        en_train, en_test = get_source_data(args, dev=True)
        print(f'loaded {len(en_train)} source train samples and {len(en_test)} source test samples')
        if args.freeze_head or args.lm_zero_shot:
            # train an en model, and save it somewhere. Supports frozen classification heads or LMs
            en_freeze_train = en_train.apply(encode_batch, axis=1).reset_index()[0]
            en_freeze_test = en_test.apply(encode_batch, axis=1).reset_index()[0]

            
            freeze_model = make_model(args, 
                lm_adapter=args.lm_zero_shot and args.lm_zero_src_lm_adapter)
            print('training source model')
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
            print('Source performance')
            for key,value in freeze_trainer.evaluate().items():
                print(key, ':', value)
            print('\n', '='*50, '\n')

            if args.lm_zero_shot:
                print('slotting in adapter for zero shot')
                stacked = slot_in_adapter(args, freeze_model, args.lm_zero_tgt_lm_adapter)
                eval_trainer = AdapterTrainer(
                    model=freeze_model,
                    args=training_args,
                    eval_dataset=combined_test[0].apply(encode_batch, axis=1).reset_index()[0],
                    compute_metrics=compute_scores
                )
                if not args.show_bar:
                    eval_trainer.remove_callback(ProgressCallback)
                print(f'Zero Shot performance on {args.lang_code}:')

                use_model = freeze_model
                
                for key, value in eval_trainer.evaluate().items():
                    print(key, ':', value)
                print('\n', '='*50, '\n')

            freeze_model.save_head(args.tmp_folder+'head', 'sa')

            del en_freeze_train
            del en_freeze_test
            del freeze_model
            del freeze_trainer
            gc.collect()

        elif args.use_source_train or args.translate_en_to_lang:
            combined_train.append(en_train)
            combined_test.append(en_test)
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

    if use_model is not None:
        model = use_model
    elif args.train_lm and args.train_da:
        model = make_model(args, 
                stack=(args.lm_zero_tgt_lm_adapter, da_loc),
                load_head=args.tmp_folder+'head' if args.freeze_head else None,
                freeze_head=args.freeze_head)
    else:
        lm_adapter = None
        if args.train_lm:
            lm_adapter = args.lm_zero_tgt_lm_adapter
        elif args.train_da:
            lm_adapter = da_loc
        model = make_model(args, 
                lm_adapter=lm_adapter,
                load_head=args.tmp_folder+'head' if args.freeze_head else None,
                freeze_head=args.freeze_head)

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
