from transformers import pipeline

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import textwrap

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

classifier = pipeline("zero-shot-classification")

res = classifier("This is a great movie", candidate_labels=["positive", "negative"])
print(res)
