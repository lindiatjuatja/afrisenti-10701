""" 
Script to use LM probabilities to classify tweets
Can vary number of examples for in-context learning (multiples of three to balance labels in examples)
Uses 70/30 train-dev split
"""
import argparse
import pandas as pd
import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)   
torch.cuda.empty_cache()
tqdm.pandas()

label_list = ['positive', 'neutral', 'negative']
label_to_id = {v: i for i, v in enumerate(label_list)}
id_to_label = {i: v for v, i in label_to_id.items()}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_dir",
        type=str,
        default="/home/ltjuatja/afrisent-semeval-2023",
        help="project dir"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="pcm",
        help="language code for data"
    ),
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="cache dir"
    ),
    parser.add_argument(
        "--num_example_sets",
        type=int,
        default=20,
        help="number of example sets in prefix"
    ),
    parser.add_argument(
	"--random_seed",
	type=int,
	default=42,
	help="random seed"
    )
    args = parser.parse_args()
    return args

def format_data(data_path):
    df = pd.read_csv(data_path, sep='\t')
    return df

def make_template(tweet_text, label):
    tweet = "Tweet: " + tweet_text + "\n"
    question = "Is the above tweet positive, neutral, or negative?: "
    answer = label
    template = tweet + question + answer
    return template
    
def make_context(df, num_example_sets, random_seed):
    pos_examples = df.loc[df['label'] == "positive"].sample(n=num_example_sets, random_state=42)
    neu_examples = df.loc[df['label'] == "neutral"].sample(n=num_example_sets, random_state=42)
    neg_examples = df.loc[df['label'] == "negative"].sample(n=num_example_sets, random_state=42)
    examples = pd.concat([neu_examples, pos_examples, neg_examples])
    shuffled = examples.sample(frac=1, random_state=random_seed)
    ordered_examples = dict(shuffled.values)
    context = ""
    for tweet_text, label in ordered_examples.items():
        template = make_template(tweet_text, label) + "\n"
        context += template
    return context

def score_label(tweet, label, context, model, tokenizer):
    prompt = context + make_template(tweet, "")
    completed = prompt + label
    # enc = tokenizer(completed, return_tensors="pt")
    enc = tokenizer(completed, return_tensors="pt").to(device)
    ans_len = tokenizer(label, return_tensors="pt").input_ids.size(1)
    input_ids = enc.input_ids[:, 0:]
    target_ids = input_ids.clone()
    target_ids[:, :-ans_len] = -100
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        LL = outputs[0] * -1
    return LL.item()
    
# def predict_label(tweet, context, model, tokenizer):
#     prompt = context + make_template(tweet, "")
#     # print(prompt)
#     scores = np.zeros(3)
#     for label in label_list:
#         completed = prompt + label
#         enc = tokenizer(completed, return_tensors="pt")
#         # enc = tokenizer(completed, return_tensors="pt").to(device)
#         ans_len = tokenizer(label, return_tensors="pt").input_ids.size(1)
#         input_ids = enc.input_ids[:, 0:]
#         target_ids = input_ids.clone()
#         target_ids[:, :-ans_len] = -100
#         with torch.no_grad():
#             outputs = model(input_ids, labels=target_ids)
#         scores[label_to_id.get(label)] = outputs[0] * -1
#     label = id_to_label.get(np.argmax(scores))   
#     print(label)
#     return label

def main():
    args = parse_args()
    PROJECT_DIR = args.project_dir
    LANGUAGE = args.language
    NUM_EXAMPLE_SETS = args.num_example_sets
    RANDOM_SEED = args.random_seed
    cache_path = args.cache_dir
    
    DATA_DIR = os.path.join(PROJECT_DIR, 'SubtaskA/train/split-train-dev/')
    LANG_DIR = os.path.join(DATA_DIR, LANGUAGE)
    EXP_DIR = os.path.join(PROJECT_DIR, 'SubtaskA/generative_exp/', LANGUAGE)
    if not os.path.isdir(EXP_DIR):
        os.makedirs(EXP_DIR)
    
    tokenizer = AutoTokenizer.from_pretrained("bonadossou/afrolm_active_learning")
    model = AutoModelForMaskedLM.from_pretrained("bonadossou/afrolm_active_learning")      

    train_df = format_data(os.path.join(LANG_DIR, 'train.tsv'))
    dev_df = format_data(os.path.join(LANG_DIR, 'dev.tsv'))
    context = make_context(train_df, NUM_EXAMPLE_SETS, RANDOM_SEED)
    print(context)
    
    for label in label_list:
        dev_df[label] = dev_df['text'].progress_apply(score_label, label=label, model=model, tokenizer=tokenizer, context=context)
    
    dev_df["predicted label"] = dev_df[label_list].idxmax(axis=1)
    print(dev_df.head())
    output_file = LANGUAGE + '-afroLM-' + str(NUM_EXAMPLE_SETS) + '-' + str(RANDOM_SEED) + '.tsv'
    dev_df.to_csv(os.path.join(EXP_DIR, output_file), sep='\t')
    
if __name__ == "__main__":
    main()
