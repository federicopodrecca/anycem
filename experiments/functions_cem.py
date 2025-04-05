import re
import torch
import warnings
import pytorch_lightning as pl
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Cleaning Generated Response

def clean_response(response, prompt):

    response = response.replace(prompt, "").strip()
    
    response = re.sub(r'(<[^>]*>)|\[.*?\]', '', response).strip()

    response = re.split(r'[.!?]', response)[0]  

    response = re.sub(r'\s+', ' ', response).strip()

    return response.strip()

# Changing llama embeddings with cem ones 

def emb_rep_sct(prompts : list, changes : int, 
                tokenizer, llm_model,  best_cem,
                changes_inds = None, n_gen = 20, c_true = None, intervention_idx = None):
    
    cem_llama = {}
    embs = []
    inv_key = []

    for j in range(len(prompts)):
        cem_llama[j] = {}
        
        # llama
        inputs_ids = tokenizer(prompts[j], return_tensors = "pt").input_ids.to("cuda:1")
        generated_tokens_llama = inputs_ids.clone()

        for step in range(n_gen):
            pl.seed_everything(step)
            with torch.no_grad():
                outputs = llm_model(input_ids = generated_tokens_llama, output_hidden_states = True)
            
            hidden_states = outputs.hidden_states[-1]
            logits = llm_model.lm_head(hidden_states[:, -1, :])
            next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
            generated_tokens_llama = torch.cat((generated_tokens_llama, next_token), dim=-1)
            output_text_llama = tokenizer.decode(generated_tokens_llama[0], skip_special_tokens=True)
            
        
        
        # embeddings extraction
        inputs = tokenizer(clean_response(output_text_llama, prompts[j]), return_tensors = "pt").to("cuda:1")
        
        with torch.no_grad():
            output = llm_model(**inputs, output_hidden_states=True)  
    
   
        last_hidden_states = output.hidden_states[-1] 
    
        last_embedding = last_hidden_states[-1]
        embs.append(last_embedding)

        # cem-llama

        x_input = embs[j].to("cpu").to(next(best_cem.parameters()).dtype)
        embs_cem = best_cem(x_input, c_true = c_true, intervention_idx = intervention_idx)[0]

        bool_vec = [False] * embs_cem.shape[0]
        if changes > embs_cem.shape[0]:
            inv_key.append(j)
            continue

        inds = sorted(random.sample(range(embs_cem.shape[0]), changes))
        if changes_inds:
            for c_i in changes_inds:
                bool_vec[c_i] = True
        else:
            for ind in inds:
                bool_vec[ind] = True
        generated_tokens_cem = inputs_ids.clone()

        for step, mod in enumerate(bool_vec):
            pl.seed_everything(step)
            with torch.no_grad():
                outputs = llm_model(input_ids = generated_tokens_cem, output_hidden_states = True)

            hidden_states = outputs.hidden_states[-1]

            if mod:
                new_tensor = embs_cem[step, :]
                hidden_states[:, -1, :] = new_tensor
                logits = llm_model.lm_head(hidden_states[:, -1, :])
                next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
                generated_tokens_cem = torch.cat((generated_tokens_cem, next_token), dim=-1)
                output_text_cem = tokenizer.decode(generated_tokens_cem[0], skip_special_tokens=True)
                
            
            else:
                hidden_states = outputs.hidden_states[-1]
                logits = llm_model.lm_head(hidden_states[:, -1, :])
                next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
                generated_tokens_cem = torch.cat((generated_tokens_cem, next_token), dim=-1)
                output_text_cem = tokenizer.decode(generated_tokens_cem[0], skip_special_tokens=True)

        # cem_none
        x_input = embs[j].to("cpu").to(next(best_cem.parameters()).dtype)
        embs_cem = best_cem(x_input, c_true = c_true, intervention_idx = None)[0]

        bool_vec = [False] * embs_cem.shape[0]
        if changes > embs_cem.shape[0]:
            inv_key.append(j)
            continue

        if changes_inds:
            for c_i in changes_inds:
                bool_vec[c_i] = True
        else:
            for ind in inds:
                bool_vec[ind] = True
        generated_tokens_cem_none = inputs_ids.clone()

        for step, mod in enumerate(bool_vec):
            pl.seed_everything(step)
            with torch.no_grad():
                outputs = llm_model(input_ids = generated_tokens_cem_none, output_hidden_states = True)

            hidden_states = outputs.hidden_states[-1]

            if mod:
                new_tensor = embs_cem[step, :]
                hidden_states[:, -1, :] = new_tensor
                logits = llm_model.lm_head(hidden_states[:, -1, :])
                next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
                generated_tokens_cem_none = torch.cat((generated_tokens_cem_none, next_token), dim=-1)
                output_text_cem_none = tokenizer.decode(generated_tokens_cem_none[0], skip_special_tokens=True)
                
            
            else:
                hidden_states = outputs.hidden_states[-1]
                logits = llm_model.lm_head(hidden_states[:, -1, :])
                next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
                generated_tokens_cem_none = torch.cat((generated_tokens_cem_none, next_token), dim=-1)
                output_text_cem_none = tokenizer.decode(generated_tokens_cem_none[0], skip_special_tokens=True)
                
        # cem_llama[j]["cem-llama"] = clean_response(output_text, prompts[j])
        cem_llama[j]["llama"] = clean_response(output_text_llama, prompts[j])
        cem_llama[j]["cem-llama"] = clean_response(output_text_cem, prompts[j])
        cem_llama[j]["cem-llama-none"] = clean_response(output_text_cem_none, prompts[j])
        if changes_inds:
            cem_llama[j]["changes"] = changes_inds

        else:
            cem_llama[j]["changes"] = inds

        model_sim = SentenceTransformer('all-MiniLM-L6-v2')
        emb_sim1 = model_sim.encode(cem_llama[j]["cem-llama"]).reshape(1, -1)
        emb_sim2 = model_sim.encode(cem_llama[j]["llama"]).reshape(1, -1)
        similarity = cosine_similarity(emb_sim1, emb_sim2)[0][0]
        cem_llama[j]["similarity"] = float(round(similarity, 2))

    for key in inv_key:
        del cem_llama[key]

    return cem_llama