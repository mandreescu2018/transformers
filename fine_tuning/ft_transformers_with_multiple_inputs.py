from datasets import load_dataset, load_metric
import numpy as np
from pprint import pprint
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score
import utils


def tokenize_fn(batch):
    return tokenizer(batch['sentence1'], batch['sentence2'], truncation=True)


raw_datasets = load_dataset("glue", "rte")

print(raw_datasets)
print("\n====== features in dataset ==========")
pprint(raw_datasets['train'].features)

print("\n====== 10 sentences ==========")
print(raw_datasets['train']['sentence1'][:10])

checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

print("\n====== tokenized ==========")
print("---- raw_datasets['train']['sentence1'][0] ---")
print(raw_datasets['train']['sentence1'][0])

print("--- raw_datasets['train']['sentence2'][0] ---")
print(raw_datasets['train']['sentence2'][0])

res = tokenizer(raw_datasets['train']['sentence1'][0],
                raw_datasets['train']['sentence2'][0])
print("-- keys ---")
print(res.keys())
print("\n====== tokenizer decode ==========")
print(tokenizer.decode(res['input_ids']))

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

training_args = TrainingArguments(
    output_dir='training_dir_multiple',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    logging_steps=150  # otherwise, 'no log will appear under training loss'
)

metric = load_metric("glue", "rte")

# test metric
print(metric.compute(predictions=[1, 0, 1], references=[1, 0, 0]))

tokenized_dataset = raw_datasets.map(tokenize_fn, batched=True)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=utils.compute_metrics
)

trainer.train()
