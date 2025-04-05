import torch
from functions_cem import emb_rep_sct
from models import GenerativeCEM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random

# Caricamento Dataset e preparazione Train, Valid, Test

with open('prompt_emb.pkl', 'rb') as f:
    loaded_prompt_emb = pickle.load(f)

print("prompt_emb caricato con successo!")

# CEM

# CEM 
best_cem = GenerativeCEM(3072, 3, 128)
best_cem.load_state_dict(torch.load("best_cem_model.pth"))
best_cem.eval()

# Modello

model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(type(tokenizer))
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:1", torch_dtype=torch.bfloat16)
print(type(model))
print(type(["a"]))
model = model.to("cuda:1")

# prompts

# Categories of prompts
topics = [
    "Write a story about", "Explain the concept of", "Generate Python code for", 
    "Create a dialogue between", "Describe the future of", "Give advice on", 
    "List the pros and cons of", "Provide a summary of", "Answer the question:",
    "Imagine a world where"
]

# Random subjects to combine
subjects = [
    "artificial intelligence", "a talking cat", "a futuristic city", "an innovative startup", 
    "general relativity", "a cooking robot", "an intergalactic detective", "an ancient mystery", 
    "a technological revolution", "a society without money", "a habitable planet"
]

def generate_prompts(n=500):
    prompts = [f"{random.choice(topics)} {random.choice(subjects)}." for _ in range(n)]
    return prompts

# Generate prompts
prompt_list = generate_prompts()


avg_sim = []
sd_sim = []

for c in range(1, 10):
    res = emb_rep_sct(prompt_list, c, tokenizer, model, best_cem)
    sim = []
    for i in res.keys():
        sim.append(res[i]["similarity"])

    avg_sim.append(np.mean(sim))
    sd_sim.append(np.std(sim, ddof = 1))



# Creazione del grafico
plt.figure(figsize=(6, 4))

# Plotta entrambe le serie
plt.plot(list(range(1, 10)), avg_sim, marker='o', linestyle='-', label="Avg", color='b')
plt.plot(list(range(1, 10)), sd_sim, marker='s', linestyle='--', label="Std", color='r')

# Etichette degli assi
plt.xlabel("n of changes")
plt.ylabel("Similarity")

# Titolo
plt.title("Similarity cem-llama and llama")

# Legenda
plt.legend()

# Mostra il grafico
plt.savefig('impact_of_changes.png')

