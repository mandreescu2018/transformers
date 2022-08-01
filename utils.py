import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import textwrap
from sklearn.metrics import f1_score
import numpy as np


def plot_cm(cm, classes=None):
    # classes = ['negative', 'positive']
    if classes is None:
        classes = ['negative', 'positive']
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    ax = sn.heatmap(df_cm, annot=True, fmt='g')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    plt.show()


def wrap(x):
    return textwrap.fill(x, replace_whitespace=False, fix_sentence_endings=True)


def tokenize_fn(batch, tokenizer, batch_str='sentence'):
    return tokenizer(batch[batch_str], truncation=True)


def compute_metrics(logits_and_labels):
    logits, labels = logits_and_labels
    predictions = np.argmax(logits, axis=-1)
    acc = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average='macro')
    return {'accuracy': acc, 'f1': f1}


# for NER
def align_targets(labels, word_ids, begin2inside):
    aligned_labels = []
    last_word = None
    for word in word_ids:
        if word is None:
            label = -100  # it's a token like [CLS]
        elif word != last_word:
            label = labels[word]  # it is a new word
        else:
            label = labels[word]  # it is the same word as before
            # change  B-<tag> to I-<tag> if necessary
            if label in begin2inside:
                label = begin2inside[label]
        aligned_labels.append(label)
        last_word = word
    return aligned_labels


# for NER
def ner_tokenize_fn(batch, tokenizer):
    # tokenize the input sequence first
    # this populates input_ids. attention mask, etc.
    tokenized_inputs = tokenizer(batch['tokens'], truncation=True, is_split_into_words=True)
    labels_batch = batch['ner_tags']  # original targets
    aligned_labels_batch = []  # aligned targets
    for i, labels in enumerate(labels_batch):
        word_ids = tokenized_inputs.word_ids(i)
        aligned_labels_batch.append(align_targets(labels, word_ids))
    # the 'target' must be stored in key called 'labels'
    tokenized_inputs['labels'] = aligned_labels_batch

    return tokenized_inputs
