from transformers import pipeline

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import textwrap
from pprint import pprint
import utils

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

classifier = pipeline("zero-shot-classification", device=0)


# res = classifier("This is a great movie", candidate_labels=["positive", "negative"])
# print(res)
#
# text = "Due to the presence of isoforms of its components, there are " \
#        "12 versions of AMPK in mammals, each of which can have different tissue " \
#        "localizations, and different functions under different conditions. " \
#        "AMPK is regulated allosterically and by post-translational " \
#        "modification, which work together."
#
# cand_labels = ['biology', 'math', 'geology']
#
# res = classifier(text, candidate_labels=cand_labels)
# print(cand_labels[np.argmax(res['scores'])])

df = pd.read_csv('bbc_text_cls.csv')
print(len(df))
print(df.sample(frac=1).head())
labels = list(set(df['labels']))
print(labels)
print('\n--------')
print(df.iloc[1024]['labels'])
print('\n--------')
print(utils.wrap(df.iloc[1024]['text']))

preds = classifier(df.iloc[1024]['text'], candidate_labels=labels)
pprint(preds['labels'][np.argmax(preds['scores'])])
