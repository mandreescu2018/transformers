import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

import torch
from torchinfo import summary
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    pipeline, AutoConfig

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from datasets import load_dataset
import utils


def tokenize_fn(batch):
    return tokenizer(batch['sentence'], truncation=True)


def compute_metrics(logits_and_labels):
    logits, labels = logits_and_labels
    predictions = np.argmax(logits, axis=-1)
    acc = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average='macro')
    return {'accuracy': acc, 'f1': f1}


def get_label(d):
    return int(d['label'].split('_')[1])


df_ = pd.read_csv("../beginners/AirlineTweets.csv")
print(df_.head())

df = df_[['airline_sentiment', 'text']]
# print(df.head())
# df['airline_sentiment'].hist()
# plt.show()

target_map ={'positive': 1, 'negative': 0, 'neutral': 2}
df['target'] = df['airline_sentiment'].map(target_map)
print(df.head())

df2 = df[['text', 'target']]
df2.columns = ['sentence', 'label']
df2.to_csv('data.csv', index=None)

print("\n========= raw dataset =====")
raw_dataset = load_dataset('csv', data_files='data.csv')
print(raw_dataset)

# if you have multiple csv files:
# datafiles=[file1.csv, file2.csv]

print("\n========= split dataset =====")
split = raw_dataset['train'].train_test_split(test_size=0.3, seed=42)
print(split)

checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# ==================
# AutoConfig
# ===================
print("\n========= config =====")
config = AutoConfig.from_pretrained(checkpoint)
print(config)

# print(config.id2label)
# print(config.label2id)

config.id2label = {v: k for k, v in target_map.items()}
config.label2id = target_map

# exit(0)


tokenized_dataset = split.map(tokenize_fn, batched=True)

# model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config=config)

# print model
summary(model)

training_args = TrainingArguments(
    output_dir='training_dir',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# trainer.train()

savedmodel = pipeline("text-classification", model='training_dir/checkpoint-641', device=0)

print(split['test'])

test_pred = savedmodel(split['test']['sentence'])

# print(test_pred)
test_pred = [get_label(d) for d in test_pred]
print(test_pred)

print("acc:", accuracy_score(split['test']['label'], test_pred))
print("f1:", f1_score(split['test']['label'], test_pred, average='macro'))

cm = confusion_matrix(split['test']['label'], test_pred, normalize='true')
classes = ['negative', 'positive', 'neutral']
utils.plot_cm(cm, classes=classes)



