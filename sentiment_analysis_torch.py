from transformers import pipeline
import pandas as pd
import torch
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

import utils

# res = torch.cuda.is_available()
# print(res)

# res = torch.cuda.current_device()
# print(res)

classifier = pipeline("sentiment-analysis", device=0)
# classifier = pipeline("sentiment-analysis")

df_ = pd.read_csv("AirlineTweets.csv")
print(df_.head())

df = df_[['airline_sentiment', 'text']].copy()

# plt.hist(df['airline_sentiment'])
# plt.show()
df = df[df.airline_sentiment != 'neutral']

target_map = {'positive': 1, 'negative': 0}
df['target'] = df['airline_sentiment'].map(target_map)

print(len(df))

start_time = time.perf_counter()

texts = df['text'].tolist()
predictions = classifier(texts)

end_time = time.perf_counter()
# print(predictions)

print(f"Duration: {end_time - start_time} seconds")

probs = [d['score'] if d['label'].startswith('P') else 1 - d['score'] for d in predictions]
print(probs[:16])

preds = [1 if d['label'].startswith('P') else 0 for d in predictions]

preds = np.array(preds)

print("acc: ", np.mean(df['target'] == preds))
cm = confusion_matrix(df['target'], preds, normalize='true')

utils.plot_cm(cm)

f1score = f1_score(df['target'], preds)
print('f1_score:', f1score)
f1score = f1_score(1 - df['target'], 1 - preds)
print('f1_score invert:', f1score)

rocscore = roc_auc_score(df['target'], probs)
print('roc_auc_score:', rocscore)
rocscore = roc_auc_score(1 - df['target'], 1 - np.array(probs))
print('roc_auc_score inverted:', rocscore)
