from transformers import pipeline
import pandas as pd
import numpy as np
import textwrap

import utils

df = pd.read_csv("bbc_text_cls.csv")
# print(df.head())

doc = df[df.labels == 'business']['text'].sample(random_state=42)

print('\n--sample--')
print(utils.wrap(doc.iloc[0]))

summarizer = pipeline("summarization")
# res = summarizer(doc.iloc[0].split("\n", 1)[1])
# print(utils.wrap(res[0]['summary_text']))

print('\n ---- entertainment ----')
doc = df[df.labels == 'entertainment']['text'].sample(random_state=123)
print(utils.wrap(doc.iloc[0]))

print('\n ---- result ----')
res = summarizer(doc.iloc[0].split("\n", 1)[1])
print(utils.wrap(res[0]['summary_text']))

