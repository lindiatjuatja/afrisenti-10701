from itertools import chain
from datasets import load_dataset, load_metric

import transformers.adapters.composition as ac
from transformers import (
    AdapterTrainer,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from transformers.adapters.configuration import AdapterConfig
from transformers.trainer_callback import PrinterCallback, ProgressCallback
import math

import warnings
warnings.filterwarnings(action='ignore')

def train_wiki_lm_and_save(
    args
    ):
    valid_percentage = 10
    language_code = args.lang_code
    task_name = language_code + "_mlm"
    date = args.lm_date
    adapter_location = args.tmp_folder+language_code+'_lm_adapter'

    training_args = TrainingArguments(
        args.tmp_folder+'lm/',
        learning_rate=args.lm_lr,
        num_train_epochs=args.lm_epochs, report_to="all", 
        gradient_accumulation_steps=args.lm_gradient_accumulation_steps,
        per_device_train_batch_size=args.lm_per_device_batch_size, 
        per_device_eval_batch_size=args.lm_per_device_batch_size)

    raw_datasets = load_dataset(
        "wikipedia", language=language_code, date=date, beam_runner='DirectRunner'
    )
    raw_datasets["validation"] = load_dataset(
        "wikipedia", language=language_code, date=date, 
        split=f"train[:{valid_percentage}%]", beam_runner='DirectRunner'
    )
    raw_datasets["train"] = load_dataset(
        "wikipedia", language=language_code, date=date, 
        split=f"train[{valid_percentage}%:]", beam_runner='DirectRunner'
    )
    print("Training LM Adapter")
    print(f'LM Train size: {len(raw_datasets["train"])}, LM Val size: {len(raw_datasets["validation"])}')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    model = AutoModelForMaskedLM.from_pretrained(args.base_model)
    adapter_config = AdapterConfig.load(args.adapter_type)
    model.add_adapter(task_name, config=adapter_config)

    model.train_adapter([task_name])
    model.set_active_adapters(task_name)

    column_names = raw_datasets["train"].column_names
    text_column_name = "text"
    max_seq_length = tokenizer.model_max_length

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names
    )

    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = load_metric("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        return metric.compute(predictions=preds, references=labels)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability)

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    if not args.show_bar:
        trainer.remove_callback(ProgressCallback)

    trainer.train()
    metrics = trainer.evaluate()
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")

    metrics["perplexity"] = perplexity
    print('LM Metrics:', '\n')
    for key,value in metrics.items():
        print(key, ':', value)
    
    model.save_adapter(adapter_location, task_name)
    return adapter_location