from transformers import pipeline
import pickle
from pprint import pprint
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.metrics import accuracy_score, f1_score


def compute_prediction(tokens, input_, ner_result):
    # map hugging face ner result to list of tags for later performance assessment
    # tokens is the original tokenized sentence
    # input_ is the detokenized string

    predicted_tags = []
    state = 'O'  # keep the track of the state, so if O ---> B, if B ---> I, if I ---> I
    current_index = 0
    for token in tokens:
        # find token in text
        index = input_.find(token)
        assert (index >= 0)
        current_index += index

        # check if this index belong to an entity and assig label
        tag = 'O'
        for entity in ner_result:
            if entity['start'] <= current_index < entity['end']:
                # then this token belongs to an entity
                if state == 'O':
                    state = 'B'
                else:
                    state = 'I'
                tag = f"{state}-{entity['entity_group']}"
                break
        if tag == 'O':
            # reset the state
            state = 'O'
        predicted_tags.append(tag)

        # remove the token from input_
        input_ = input_[index + len(token):]

        # update current index
        current_index += len(token)

    # sanity check
    assert (len(predicted_tags) == len(tokens))
    return predicted_tags


def flatten(list_of_lists):
    flattened = [val for sublist in list_of_lists for val in sublist]
    return flattened


# initiate the transformer
# ner = pipeline("ner", aggregation_strategy='simple', device=0)
ner = pipeline("ner", aggregation_strategy='simple')

with open('ner_train.pkl', 'rb') as f:
    corpus_train = pickle.load(f)

with open('ner_test.pkl', 'rb') as f:
    corpus_test = pickle.load(f)

# pprint(corpus_test)

inputs = []
targets = []

for sentence_tag_pairs in corpus_test:
    tokens = []
    target = []
    for token, tag in sentence_tag_pairs:
        tokens.append(token)
        target.append(tag)
    inputs.append(tokens)
    targets.append(target)

# pprint(inputs[9])
# pprint(targets[9])

detokenizer = TreebankWordDetokenizer()

res = detokenizer.detokenize(inputs[9])
print(res)

# test transformer model
res = ner(res)
pprint(res)

input_ = detokenizer.detokenize(inputs[9])
ner_result = ner(input_)
ptags = compute_prediction(inputs[9], input_, ner_result)
print(accuracy_score(targets[9], ptags))

# get detokenized inputs to pass into ner transformer model
detok_inputs = []
for tokens in inputs:
    text = detokenizer.detokenize(tokens)
    detok_inputs.append(text)

# 17 min on CPU, 3 min on GPU
ner_result = ner(detok_inputs)

predictions = []
for tokens, text, ner_result in zip(inputs, detok_inputs, ner_result):
    pred = compute_prediction(tokens, text, ner_result)
    predictions.append(pred)

flat_predictions = flatten(predictions)
flat_targets = flatten(targets)

print("\naccuracy_score")
print(accuracy_score(flat_targets, flat_predictions))
print("\nf1_score")
print(f1_score(flat_targets, flat_predictions, average='macro'))

