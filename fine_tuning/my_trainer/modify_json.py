import json

config_path = "D:\\work\\PythonProjects\\transformers_learn\\transformers\\fine_tuning\\may_saved_model\\config.json"
with open(config_path) as f:
    j = json.load(f)

j['id2label'] = {0:'negative', 1: 'positive'}

with open(config_path, 'w') as f:
    json.dump(j, f, indent=2)
