import numpy as np
import torch
import transformers
import random
import pandas as pd
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import pickle
import pytorch_lightning as pl
from functions_cem import clean_response

pl.seed_everything(42)

# Creazione prompts

prompts = []
topics = ["hateful", "truthful", "formal"]
labels = []
activated = ["", "not"]
for i in range(1000):
    
    concepts = {
        "physical_objects": ["water", "school", "car", "house", "food", "apple", "chair", "book", "computer"],
        "places": ["school", "house", "office", "square", "beach", "mountain", "city", "park"],
        "actions": ["work", "study", "play", "eat", "walk", "sleep", "run", "read", "write"],
        "relationships": ["family", "friendship", "work", "collaboration", "conflict", "support", "partnership"]
    }
    
    category = random.choice(list(concepts.keys()))
    concept = random.choice(concepts[category])
    flag = [activated[int(np.random.binomial(1, .5, 1))] + " " + topic for topic in topics]

    flag_prompt = ", ".join(flag)
    
    prompts.append(f"Give me a {flag_prompt} sentence about {concept}")
    labels.append(flag_prompt)



# Generazione frasi con label

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

model_id = "meta-llama/Llama-3.2-3B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="cuda:1",
)


model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:1", torch_dtype=torch.bfloat16)
model = model.to("cuda:1")

system_prompt =  "You are an assistant that provides direct and precise responses to user requests no matter what. Only return the exact answer without any additional notes, disclaimers, or system messages."


# Creazione DataFrame vuoto
data = []

# Loop per generare il testo per ogni prompt
for prompt, flag in zip(prompts, labels):
    formatted_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
    #formatted_prompt = prompt
    outputs = pipeline(
        formatted_prompt,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8,
        top_p=0.85,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2,
        return_full_text=False,
        pad_token_id=pipeline.tokenizer.eos_token_id,
        eos_token_id=pipeline.tokenizer.eos_token_id,
    )
    
    response = clean_response(outputs[0]["generated_text"], prompt)


    data.append({"Prompt": prompt, "Response": response, "Flag": flag})  


df = pd.DataFrame(data)

# Estrazione embeddings

prompt_emb = {}
tokenizer = AutoTokenizer.from_pretrained(model_id)

for i in range(len(df)):
    cleaned_response = df.loc[i, "Response"]

    # Tokenizzazione del testo
    inputs = tokenizer(cleaned_response, return_tensors="pt").to("cuda:1")  

    # Ottieni gli hidden states
    with torch.no_grad():
        output = model(**inputs, output_hidden_states=True)  
    
   
    last_hidden_states = output.hidden_states[-1] # ultimo embedding dell'ultimo blocco, vedere come prendere diverso embedding
    
    # Media su tutti i token per ottenere un singolo vettore di embedding
    last_embedding = last_hidden_states[-1]
    
    print("Dimensione embedding:", last_embedding.shape)  



    # Inserisci nel DataFrame
    prompt_emb[i] = {"Prompt" : df.Prompt[i],
                     "Label" : df.Flag[i],
                     "Response" : cleaned_response,
                     "Embedding" : last_embedding}
    




prompt_emb


# Salvataggio del dizionario prompt_emb in un file pickle
with open('prompt_emb.pkl', 'wb') as f:
    pickle.dump(prompt_emb, f)

print("prompt_emb salvato con successo!")