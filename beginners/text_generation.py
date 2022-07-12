from transformers import pipeline, set_seed
import pandas as pd

# import torch
import matplotlib.pyplot as plt
import time
import numpy as np
# from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import utils


from pprint import pprint
import textwrap

# import wget
# txt_url= "https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/robert_frost.txt"
# wget.download(txt_url)

lines = [line.strip() for line in open("robert_frost.txt")]
lines = [line for line in lines if len(line) > 0]

# print(lines)
gen = pipeline("text-generation")
set_seed(1234)

print(lines[0])
res = gen(lines[0], max_length=20, num_return_sequences=3)
pprint(res)

out = gen(lines[0], max_length=30)
print(utils.wrap(out[0]['generated_text']))

prev = 'Two roads diverged in a yellow wood, and as a cloud of dust fell on' \
       ' the ground the little girl looked down when she saw the three girls.'

print('----')
out = gen(prev + '\n' + lines[2], max_langth=60)
print(utils.wrap(out[0]['generated_text']))

prev = "Two roads diverged in a yellow wood, and as a cloud of dust fell on" \
       " the ground the little girl looked down when she saw the three girls." \
       " And be one traveler, long I stood before that green mountain where a " \
       "sun would shine to"

print('--------')
out = gen(prev + '\n' + lines[4], max_langth=90)
print(utils.wrap(out[0]['generated_text']))

print('--------')
prompt = "Neural networks with attention have been used with great success in" \
         " natural language processing"

out = gen(prompt, max_length=300)
print(utils.wrap(out[0]['generated_text']))



