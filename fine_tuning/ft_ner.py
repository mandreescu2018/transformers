from datasets import load_dataset
from pprint import pprint
from transformers import AutoTokenizer
import re
import utils

data = load_dataset("conll2003")
print("========= dataset ==========")
print(data)

print("\n========= data['train'][0] ==========")
pprint(data['train'][0])

print("\n========= data['train'].features ==========")
pprint(data['train'].features)

print("\n========= data['train'].features['ner_tags'] ==========")
pprint(data['train'].features['ner_tags'])

print("\n========= data['train'].features['ner_tags'].feature.names ==========")
pprint(data['train'].features['ner_tags'].feature.names)
label_names = data['train'].features['ner_tags'].feature.names
print("\n========= label_names ==========")
print(label_names)

checkpoint = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

idx = 0
t = tokenizer(data["train"][idx]["tokens"], is_split_into_words=True)
print("\n============ tokenizer - data - tokens ===========")
pprint(t)
print(type(t))
print("\n============ tokens() ===========")
pprint(t.tokens())
print("\n============ word_ids() ===========")
# value of i indicates it is the i'th word
# in the input sentence (counting from 0)
pprint(t.word_ids())

# ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
begin2inside = {
    1: 2,
    3: 4,
    5: 6,
    7: 8
}

# try NER functions
labels = data["train"][idx]["ner_tags"]
word_ids = t.word_ids()
aligned_targets = utils.align_targets(labels, word_ids, begin2inside)

print("\n============ aligned_targets ===========")
print(aligned_targets)

print("\n========= label_names ==========")
print(label_names)

print("\n============ aligned_labels ===========")
aligned_labels = [label_names[t] if t >= 0 else None for t in aligned_targets]
for x, y in zip(t.tokens(), aligned_labels):
    print(f"\t{x}\t\t{y}")

# make up a fake input just to test it
words = ['[CLS]', 'Ger', '##man', 'call', 'to', 'boycott', 'Micro', '##soft', '[SEP]']
word_ids = [None, 0, 0, 1, 2, 3, 4, 4, None]
labels = [7, 0, 0, 0, 3]
aligned_targets = utils.align_targets(labels, word_ids, begin2inside)

print("\n============ aligned_labels 2 ===========")
aligned_labels = [label_names[t] if t >= 0 else None for t in aligned_targets]
for x, y in zip(words, aligned_labels):
    print(f"\t{x}\t\t{y}")
