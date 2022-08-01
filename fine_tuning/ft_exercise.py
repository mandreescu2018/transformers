from transformers import AutoTokenizer
import re

checkpoint = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

t = tokenizer(re.split('\s+', 'This lamb is little'), is_split_into_words=True)
print(t.word_ids())
