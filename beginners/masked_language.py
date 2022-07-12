from transformers import pipeline
import numpy as np
import pandas as pd
import textwrap
from pprint import pprint
import utils

# import wget
# csv_url = "https://lazyprogrammer.me/course_files/nlp/bbc_text_cls.csv"
# wget.download(csv_url)

df = pd.read_csv("bbc_text_cls.csv")
print(df.head())

labels = set(df['labels'])
print("--- Labels -----")
print(labels)

# pick a label
label = "business"

texts = df[df['labels'] == label]['text']
print(texts.head())

np.random.seed(1234)

i = np.random.choice(texts.shape[0])
doc = texts.iloc[i]
print("\n--- Article -----\n")
print(utils.wrap(doc))

# initiate transformer
mlm = pipeline('fill-mask')

res = mlm('Bombardier chief to leave <mask>')
pprint(res)

text = "Shares in <mask> and plane-making " \
       "giant Bombardier have fallen to a 10-year low following the departure " \
       "of its chief executive and two members of the board."
print('\n--------------')
res = mlm(text)
pprint(res)

text = "Shares in train and plane-making " \
       "giant Bombardier have fallen to a 10-year low following the <mask> " \
       "of its chief executive and two members of the board."
print('\n-------------')
res = mlm(text)
pprint(res)



