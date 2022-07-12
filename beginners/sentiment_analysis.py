from transformers import pipeline

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# import wget
#
# csv_url = "https://lazyprogrammer.me/course_files/AirlineTweets.csv"
# wget.download(csv_url)

classifier = pipeline("sentiment-analysis")

print(type(classifier))

res = classifier("This is such a great movie!")
print(res)
res = classifier("This show was not interesting!")
print(res)

res = classifier("This show was interesting!")
print(res)

res = classifier("This show was not bad at all!")
print(res)

res = classifier("I can't say this was a good movie!")
print(res)

res = classifier(["This course is just what I need.!",
                  "I can't understand any of this. Instructor kept me telling me to meet the prerequisites. "
                  "What are prerequisites? Why does he keep saying that?"])
print(res)
