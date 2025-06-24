
import random
import pandas as pd
import torch
import pickle
from functions_experiments import classify_external_sentence, classification, get_dimension_from_label
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from functions_cem import generate_prompts, llama_gen
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter
from tqdm import tqdm

device = 'cuda:1'
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(type(tokenizer))
model = AutoModelForCausalLM.from_pretrained(model_id, device_map= device, torch_dtype=torch.bfloat16)
model = model.to(device)
classifier = pipeline("text-generation", model = model, tokenizer=tokenizer)

y_true = []
y_pred = []
y_original = []

interventions_0 = [[0]] * 2
interventions_1 = [[1]] * 2
interventions_2 = [[2]] * 2
intervention = interventions_0 + interventions_1 + interventions_2

c_true_00 = c_true_10 = c_true_20 = [[0, 0, 0]] 
c_true_01 = [[1, 0, 0]] 
c_true_11 = [[0, 1, 0]]
c_true_21 = [[0, 0, 1]]
c_true = c_true_00 + c_true_01 + c_true_10 + c_true_11 + c_true_20 + c_true_21

for i in tqdm(range(150)):
    prompt_list = generate_prompts(n = 1)
    for inter in range(len(intervention)):
        res_llama = llama_gen(prompt_list, tokenizer, model, hidden_layer = {10 : False, 20 : False, 'last' : True}, n_gen=100)
        res = class_iii('parameter_last.pkl', 'best_cem_last.pth', tokenizer, model, c_true[inter], intervention[inter], res_llama, hidden_layer=-1, device = device)
        print(f'res {res}')
        if res[0]:
            y_true_temp = res[0]
            print(y_true_temp)
            dim = get_dimension_from_label(y_true_temp[-1][-1])
            original = classify_external_sentence(prompt_list, classifier, dim)
            if original:
                y_original.append(original)
                y_true.append(res[0][-1][-1])
                y_pred.append(res[1][-1])
                print(f'y_original {y_original}')
                print(f'y_true {y_true}')
                print(f'y pred {y_pred}')
labels_cm = sorted(['hateful', 'kind', 'truthful', 'lie', 'formal', 'informal'])
cm = confusion_matrix(y_true, y_pred, labels=labels_cm)
df_cm = pd.DataFrame(cm, index=[f"True {label}" for label in labels_cm],
                     columns=[f"Pred {label}" for label in labels_cm])

original_counts = Counter(y_original)
df_cm["original"] = [original_counts.get(label, 0) for label in labels_cm]

with open('experiment_II_original_30_iii.pkl', 'wb') as f:
    pickle.dump(df_cm, f)

for i in range(300):
    prompt_list = generate_prompts(n = 1)
    for inter in range(len(intervention)):
        res_llama = llama_gen(prompt_list, tokenizer, model, hidden_layer = {10 : False, 20 : False, 'last' : True}, n_gen=100)
        res = class_iii('parameter_last_prompt.pkl', 'best_cem_last_prompt.pth', tokenizer, model, c_true[inter], intervention[inter], res_llama, hidden_layer=-1, device = device)
        print(f'res {res}')
        if res[0]:
            y_true_temp = res[0]
            print(y_true_temp)
            dim = get_dimension_from_label(y_true_temp[-1][-1])
            original = classify_external_sentence(prompt_list, classifier, dim)
            if original:
                y_original.append(original)
                y_true.append(res[0][-1][-1])
                y_pred.append(res[1][-1])
                print(f'y_original {y_original}')
                print(f'y_true {y_true}')
                print(f'y pred {y_pred}')
labels_cm = sorted(['hateful', 'kind', 'truthful', 'lie', 'formal', 'informal'])
cm = confusion_matrix(y_true, y_pred, labels=labels_cm)
df_cm = pd.DataFrame(cm, index=[f"True {label}" for label in labels_cm],
                     columns=[f"Pred {label}" for label in labels_cm])

original_counts = Counter(y_original)
df_cm["original"] = [original_counts.get(label, 0) for label in labels_cm]

with open('experiment_II_original_prompt_30_ii.pkl', 'wb') as f:
    pickle.dump(df_cm, f)
  
