from datasets import load_dataset, load_metric
import numpy as np
from pprint import pprint
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification, Trainer, pipeline
from torchinfo import summary
import json

def tokenize_fn(batch):
    return tokenizer(batch['sentence'], truncation=True)


def compute_metrics(logits_and_labels):
    logits, labels = logits_and_labels
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


raw_datasets = load_dataset("glue", "sst2")
print(dir(raw_datasets['train']))
pprint(raw_datasets['train'].data)
print("\n========= raw_datasets['train'][0] =====")
pprint(raw_datasets['train'][0])

print("\n========= raw_datasets['train'][50000:50003] =====")
pprint(raw_datasets['train'][50000:50003])

print("\n========= raw_datasets['train'].features =====")
pprint(raw_datasets['train'].features)

checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

print("\n========= tokenized_sentences =====")
tokenized_sentences = tokenizer(raw_datasets['train'][0:3]['sentence'])
pprint(tokenized_sentences)

tokenized_dataset = raw_datasets.map(tokenize_fn, batched=True)
print(tokenized_dataset)
training_args = TrainingArguments(
    'my_trainer',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=1
)

print("\n========= model type =====")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
print(type(model))
print(summary(model))

print("\n========= params before =====")
params_before = []
for name, p in model.named_parameters():
    params_before.append(p.detach().cpu().numpy())
print(params_before)

print("\n========= metric =====")
metric = load_metric("glue", "sst2")
res = metric.compute(predictions=[1, 0, 1], references=[1, 0, 0])
print(res)

# trainer = Trainer(
#     model,
#     training_args,
#     train_dataset=tokenized_dataset['train'],
#     eval_dataset=tokenized_dataset['validation'],
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )
#
# trainer.train()

# trainer.save_model('may_saved_model')

# newmodel = pipeline('text-classification', model='fine_tuning/may_saved_model')
newmodel = pipeline('text-classification',
                    model='D:\\work\\PythonProjects\\transformers_learn\\transformers\\fine_tuning\\may_saved_model')

print("\n========= evaluate with pipeline =====")
print("This movie is great")
res = newmodel('This movie is great')
print(res)
print("This movie sucks")
res = newmodel('This movie sucks')
print(res)
