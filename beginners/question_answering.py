from transformers import pipeline
from pprint import pprint

qa = pipeline("question-answering")

context = "Today I went to the store to purchase a carton of milk."
question = "What did I buy?"

res = qa(context=context, question=question)
pprint(res)

context = "Out of all colors, I like blue the best."
question = "What is my favourite color?"

res = qa(context=context, question=question)
pprint(res)

print(context[26:30])

context = "Albert Einstein (14 March 1879 – 18 April 1955) was a " \
          "German-born theoretical physicist,[7] widely acknowledged to be one of the " \
          "greatest and most influential physicists of all time. Einstein is best known for developing " \
          "the theory of relativity, but he also made important contributions to " \
          "the development of the theory of quantum mechanics. Relativity and " \
          "quantum mechanics are together the two pillars of modern physics."


question = "When was Albert Einstein born?"
print("\n--------- ", question," ----------")

res = qa(context=context, question=question)
# pprint(res)
print("answer: ", res['answer'])



question = "What was Albert Einstein's occupation?"
print("\n--------- ", question," ----------")



res = qa(context=context, question=question)
# pprint(res)
print("answer: ", res['answer'])

question = "What is Albert Einstein best known for?"
print("\n--------- ", question," ----------")

res = qa(context=context, question=question)
# pprint(res)
print("answer: ", res['answer'])

question = "What else has Albert Einstein contributed to?"
print("\n--------- ", question," ----------")

res = qa(context=context, question=question)
# pprint(res)
print("answer: ", res['answer'])

question = "Where was Albert Einstein born?"
print("\n--------- ", question," ----------")

res = qa(context=context, question=question)
# pprint(res)
print("answer: ", res['answer'])

question = "What is peanut butter made of?"
print("\n--------- ", question," ----------")

res = qa(context=context, question=question)
# pprint(res)
print("answer: ", res['answer'])

