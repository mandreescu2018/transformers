from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pprint import pprint
# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()

str_text = "hello world"
checkpoint = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# pprint(tokenizer)
pprint(tokenizer(str_text))

# model inputs
# model_inputs = tokenizer(str_text)

print('\n---- how it works -----')
tokens = tokenizer.tokenize("hello world")
print(f'\ntokens:\n {tokens}')
ids = tokenizer.convert_tokens_to_ids(tokens)
print(f'ids:\n {ids}')

print('\n---- back to tokens -----')
res = tokenizer.convert_ids_to_tokens(ids)
print(f"tokens:\n{res}")
print('-- to string directly --')
res = tokenizer.decode(ids)
print(f"string:\n{res}")
print('\n----- use encode -----')
ids = tokenizer.encode(str_text)
print(f'ids:\n {ids}')

print(f'back to tokens:\n {tokenizer.convert_ids_to_tokens(ids)}')
print(f'now decode:\n {tokenizer.decode(ids)}')

# ======== AutoModelForSequenceClassification ======
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# model inputs
model_inputs = tokenizer(str_text, return_tensors='pt')
pprint(f'model_inputs:\n {model_inputs}')
outputs = model(**model_inputs)
pprint(f'outputs:\n {outputs}')

# ======== create another model ======
print('\n======== create another model ======')
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)
outputs = model(**model_inputs)
print(f'outputs:\n {outputs}')
print(f"logits:\n {outputs.logits}, {outputs['logits']}, {outputs[0]}")
print("\nConvert to a numpy arr")
print(outputs.logits.detach().cpu().numpy())

print("\n==== multiline ====")
data = [
    "I like cats.",
    "Do you like cats too?"
]
model_inputs = tokenizer(data, padding=True, truncation=True, return_tensors='pt')
print(model_inputs)

outputs = model(**model_inputs)
print(f'outputs:\n {outputs}')

