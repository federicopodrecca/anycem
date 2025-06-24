from functions_cem import emb_rep, generate_prompts, llama_gen
from models import GenerativeCEM
from huggingface_hub import login
import pickle
import pandas as pd
import numpy as np
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


with open('parameter_last.pkl', 'rb') as f:
    pars = pickle.load(f)

with open('parameter_last_with_prompt.pkl', 'rb') as f:
    pars_prompt = pickle.load(f)


# CEM 
best_cem_last_prompt = GenerativeCEM(3072, 3, 400,w_concept_loss = pars_prompt['weights'][0], 
                          w_loss=pars_prompt['weights'][1], 
                          w_concept_loss_int=pars_prompt['weights'][2], device_gpu=True)
best_cem_last_prompt.load_state_dict(torch.load("best_cem_last_prompt.pth"))
best_cem_last_prompt.eval()

best_cem_last = GenerativeCEM(3072, 3, 400,w_concept_loss = pars['weights'][0], 
                          w_loss=pars['weights'][1], 
                          w_concept_loss_int=pars['weights'][2], device_gpu=True)
best_cem_last.load_state_dict(torch.load("best_cem_last.pth"))
best_cem_last.eval()

# Model
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(type(tokenizer))
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:1", torch_dtype=torch.bfloat16)
print(type(model))
print(type(["a"]))
model = model.to("cuda:1")

# Prompts
prompt_list = generate_prompts()

avg_sim_last = []
sd_sim_last = []
avg_sim_last_prompt = []
avg_sim_seed = []

indxs = []
for j in range(6):

    ind = []
    for i in range(8):
        l = random.sample(range(j + 1, j + 10), i)
        l = sorted(l)
        l.insert(0, j)
        ind.append(l)
    indxs.append(ind)



for idx_list in indxs:
    avg_last = []
    avg_last_prompt = []
    avg_seed = []

    for idx in idx_list:
    
        res_llama = llama_gen(prompt_list, tokenizer, model)
        res_last = emb_rep(res_llama, changes_inds= idx, hidden_layer=-1, tokenizer = tokenizer, 
                            llm_model = model, best_cem = best_cem_last, n_gen = 50, llama_seed=True)
        sim = []
        sim_seed = []
        for j in res_last.keys():
            sim.append(res_last[j]["similarity-cem"])
            sim_seed.append(res_last[j]["similarity-seed"])
        avg_last.append(np.mean(sim))
        avg_seed.append(np.mean(sim_seed))

        res_last_prompt = emb_rep(res_llama, changes_inds = idx, hidden_layer=-1, tokenizer = tokenizer, 
                        llm_model = model, best_cem = best_cem_last, n_gen = 50)
        sim = []
        for j in res_last_prompt.keys():
            sim.append(res_last_prompt[j]["similarity-cem"])
        avg_last_prompt.append(np.mean(sim))

        
    avg_sim_last.append(avg_last)
    avg_sim_last_prompt.append(avg_last)
    avg_sim_seed.append(avg_seed)

avg_res = {'base' : avg_sim_last,
            'with prompt' : avg_sim_last_prompt,
            'seed' : avg_sim_seed}

df = pd.DataFrame(avg_res)
df.to_csv('avg_fixed_start_30.csv', index = False)

