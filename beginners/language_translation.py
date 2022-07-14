
from transformers import pipeline
from pprint import pprint
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import numpy as np


# from zipfile import ZipFile
#
# path_zip = "spa-eng.zip"
# zip_obj = ZipFile("spa-eng.zip")
# zip_obj.extractall()

# compile eng-spa translation

def print_random_translation(en_phrases_subset=None, translations=None):
    i = np.random.choice(len(en_phrases_subset))
    eng = en_phrases_subset[i]
    print('EN:', eng)

    translation = translations[i]['translation_text']
    print('ES translation:', translation)

    matches = eng2spa[eng]
    print('Matches:', matches)

eng2spa = {}
with open("spa-eng/spa.txt", encoding='utf-8') as f:
    for line in f.readlines():
        line = line.rstrip()
        eng, spa = line.split("\t")
        if eng not in eng2spa:
            eng2spa[eng] = []
            eng2spa[eng].append(spa)

pprint({k: eng2spa[k] for k in list(eng2spa)[:10]})

tokenizer = RegexpTokenizer(r'\w+')

print('\n---- tokenized ----')
tokens = tokenizer.tokenize('¿Qué me cuentas?'.lower())
print(tokens)

print('\n---- sentence ----')

smoother = SmoothingFunction()

res = sentence_bleu([tokens], tokens, smoothing_function=smoother.method4)
print(res)

eng2spa_tokens = {}
for eng, spa_list in eng2spa.items():
    spa_list_tokens = []
    for text in spa_list:
        tokens = tokenizer.tokenize(text.lower())
        spa_list_tokens.append(tokens)
    eng2spa_tokens[eng] = spa_list_tokens

translator = pipeline("translation", model='Helsinki-NLP/opus-mt-en-es', device=0)
res = translator("I like you and your sister")
print(res)

print('\n------- eng phrases -----')
eng_phrases = list(eng2spa.keys())
print(len(eng_phrases))

eng_phrases_subset = eng_phrases[20_000:21_000]
translations = translator(eng_phrases_subset)
print(translations[0])

scores = []
for eng, pred in zip(eng_phrases_subset, translations):
    matches = eng2spa_tokens[eng]

    # tokenize translation
    spa_pred = tokenizer.tokenize(pred['translation_text'].lower())
    score = sentence_bleu(matches, spa_pred, smoothing_function=smoother.method4)
    scores.append(score)

plt.hist(scores, bins=50)
plt.show()

mean_res = np.mean(scores)
print(mean_res)

np.random.seed(1)

print('\n----------- translation 1 --------')
print_random_translation(eng_phrases_subset, translations=translations)
print('\n----------- translation 2 --------')
print_random_translation(eng_phrases_subset, translations=translations)
print('\n----------- translation 1 --------')
print_random_translation(eng_phrases_subset, translations=translations)
