
from transformers import pipeline
from pprint import pprint
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import RegexpTokenizer


# from zipfile import ZipFile
#
# path_zip = "spa-eng.zip"
# zip_obj = ZipFile("spa-eng.zip")
# zip_obj.extractall()

# compile eng-spa translation
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

translator = pipeline("translation", model='')