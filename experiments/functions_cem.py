import re
import torch
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import TensorDataset, DataLoader
from models import GenerativeCEM
import itertools
import pickle


# Cleaning Generated Response

def clean_response(response, prompt):

    response = response.replace(prompt, "").strip()
    
    response = re.sub(r'(<[^>]*>)|\[.*?\]', '', response).strip()

    response = re.split(r'[.!?]', response)[0]  

    response = re.sub(r'\s+', ' ', response).strip()

    return response.strip()

# Changing llama embeddings with cem ones 

def llama_gen(prompts, tokenizer, llm_model, hidden_layer = {10: False, 20 : False, 'last' : True}, n_gen = 50):
    llama = {}
    for j in range(len(prompts)):
            last_embedding = []
            embedding20 = []
            embedding10 = []
            
            # llama
            inputs_ids = tokenizer(prompts[j], return_tensors = "pt").input_ids.to("cuda:1")
            generated_tokens_llama = inputs_ids.clone()

            for step in range(n_gen):
                pl.seed_everything(step)
                with torch.no_grad():
                    outputs = llm_model(input_ids = generated_tokens_llama, output_hidden_states = True)
                
                hidden_states = outputs.hidden_states[-1]
                logits = llm_model.lm_head(hidden_states[:, -1, :])
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens_llama = torch.cat((generated_tokens_llama, next_token), dim=-1)
                output_text_llama = tokenizer.decode(generated_tokens_llama[0], skip_special_tokens=True)
            
            # embeddings extraction
            inputs = tokenizer(clean_response(output_text_llama, prompts[j]), return_tensors = "pt").to("cuda:1")
            
            with torch.no_grad():
                output = llm_model(**inputs, output_hidden_states=True)  
        
            if hidden_layer['last']:
                last_hidden_states = output.hidden_states[-1] 
                last_embedding = last_hidden_states[-1]

            if hidden_layer[20]:
                hidden_states = output.hidden_states[20]
                embedding20 = hidden_states[-1]
            
            if hidden_layer[10]:
                hidden_states = output.hidden_states[10]
                embedding10 = hidden_states[-1]

            llama[j] = (prompts[j], clean_response(output_text_llama, prompts[j]), embedding10, embedding20, last_embedding, inputs_ids)
    return llama

def emb_rep(data, tokenizer, llm_model, best_cem, hidden_layer,
             changes_inds : list = None,
            c_true = None, intervention_idx = None,
            n_gen : int = 50, llama_seed = False, device = 'cuda:1'):
    
    cem_llama = {}
    inv_key = []

    for j in range(len(data.keys())):
        cem_llama[j] = {}
        if hidden_layer == 10:
            embs = data[j][2]
        if hidden_layer == 20:
            embs = data[j][3]
        if hidden_layer == -1:
            embs = data[j][4]
        inputs_ids = data[j][5]
        prompt = data[j][0]

        x_input = embs.to("cpu").to(next(best_cem.parameters()).dtype)
        embs_cem = best_cem(x_input)[0]
        print(f'first emb shape {embs_cem.shape}')

        bool_vec = [False] * embs_cem.shape[0]
        # cem_none
        if changes_inds:
            x_input = embs.to("cpu").to(next(best_cem.parameters()).dtype)
            embs_cem = best_cem(x_input)[0]
            
            bool_vec = [False] * embs_cem.shape[0]

            if any(c_i + 1 > len(bool_vec) for c_i in changes_inds):
                inv_key.append(j)
                continue
            
            for c_i in changes_inds:
                bool_vec[c_i] = True
            print(bool_vec)
            generated_tokens_cem_none = inputs_ids.clone()

            for step, mod in enumerate(bool_vec):
                print(f'embs cem {embs_cem.shape}')
                # if step >= embs_cem.shape[0]:
                #     inv_key.append(j)
                #     break 

                pl.seed_everything(step)
                with torch.no_grad():
                    outputs = llm_model(input_ids = generated_tokens_cem_none, output_hidden_states = True)

                hidden_states = outputs.hidden_states[hidden_layer]

                if mod:
                    if step >= embs_cem.shape[0]:
                        inv_key.append(j)
                        break
                    new_tensor = embs_cem[step, :]
                    hidden_states[:, -1, :] = new_tensor
                    logits = llm_model.lm_head(hidden_states[:, -1, :])
                    #next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated_tokens_cem_none = torch.cat((generated_tokens_cem_none, next_token), dim=-1)
                    output_text_cem_none = tokenizer.decode(generated_tokens_cem_none[0], skip_special_tokens=True)
                    generated_tokens_cem_none_sub = generated_tokens_cem_none.clone()

                    idx_sub = next((i for i in range(step + 1, len(bool_vec)) if bool_vec[i]), False)
                    if idx_sub:
                        for k in range(step + 1, len(bool_vec)): #     idx_sub + 1
                            pl.seed_everything(k)
                            
                            with torch.no_grad():
                                outputs = llm_model(input_ids = generated_tokens_cem_none_sub, output_hidden_states = True)
                            
                            hidden_states = outputs.hidden_states[hidden_layer]
                            logits = llm_model.lm_head(hidden_states[:, -1, :])
                            probs = torch.nn.functional.softmax(logits, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                            generated_tokens_cem_none_sub = torch.cat((generated_tokens_cem_none_sub, next_token), dim=-1)

                            output_text_cem_none_sub = tokenizer.decode(generated_tokens_cem_none_sub[0], skip_special_tokens=True)
                        inputs_sub = tokenizer(clean_response(output_text_cem_none_sub, prompt), return_tensors = "pt").to("cuda:1")
                        
                        with torch.no_grad():
                                output_sub = llm_model(**inputs_sub, output_hidden_states=True)  
                        
                        last_embedding_sub = output_sub.hidden_states[hidden_layer][-1] #togliere
                        
                        x_input_sub = last_embedding_sub.to("cpu").to(next(best_cem.parameters()).dtype)
                        embs_cem = best_cem(x_input_sub)[0]
                                

                
                else:
                    logits = llm_model.lm_head(hidden_states[:, -1, :])
                    #next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated_tokens_cem_none = torch.cat((generated_tokens_cem_none, next_token), dim=-1)
                    output_text_cem_none = tokenizer.decode(generated_tokens_cem_none[0], skip_special_tokens=True)

        # llama-seed
        if llama_seed:
            generated_tokens_llama_seed = inputs_ids.clone()
            if len(bool_vec) < n_gen:
                bool_vec_seed = bool_vec + [False] * (n_gen - len(bool_vec))
            for step in range(n_gen):
                if bool_vec_seed[step]:
                    pl.seed_everything(step + 50)
                else:
                    pl.seed_everything(step)

                with torch.no_grad():
                    outputs = llm_model(input_ids = generated_tokens_llama_seed, output_hidden_states = True)
                
                hidden_states = outputs.hidden_states[-1]
                logits = llm_model.lm_head(hidden_states[:, -1, :])
                #next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens_llama_seed = torch.cat((generated_tokens_llama_seed, next_token), dim=-1)
                output_text_llama_seed = tokenizer.decode(generated_tokens_llama_seed[0], skip_special_tokens=True)

        # cem-llama-int

        if intervention_idx:
            embs_cem_int = best_cem(x_input, c_true = c_true, intervention_idx = intervention_idx)[0]
            bool_vec = [False] * embs_cem_int.shape[0]
            bool_vec[:2] = [True] * 2
            generated_tokens_cem_int = inputs_ids.clone()
            for step, mod in enumerate(bool_vec):
                print(f'step {step}')
                print(f'emb shape {embs_cem_int.shape}')
                

                pl.seed_everything(step)
                with torch.no_grad():
                    outputs = llm_model(input_ids = generated_tokens_cem_int, output_hidden_states = True)

                hidden_states = outputs.hidden_states[hidden_layer]

                if mod:
                    if step >= embs_cem_int.shape[0]:
                        inv_key.append(j)
                        break
                    new_tensor = embs_cem_int[step, :]
                    hidden_states[:, -1, :] = new_tensor
                    logits = llm_model.lm_head(hidden_states[:, -1, :])
                    
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated_tokens_cem_int = torch.cat((generated_tokens_cem_int, next_token), dim=-1)
                    output_text_cem_int = tokenizer.decode(generated_tokens_cem_int[0], skip_special_tokens=True)
                    generated_tokens_cem_int_sub = generated_tokens_cem_int.clone()

                    idx_sub_int = next((i for i in range(step + 1, len(bool_vec)) if bool_vec[i]), False)

                    if idx_sub_int:
                        for k in range(step + 1, len(bool_vec)): 
                            pl.seed_everything(k)
                            
                            with torch.no_grad():
                                outputs = llm_model(input_ids = generated_tokens_cem_int_sub, output_hidden_states = True)
                            
                            hidden_states = outputs.hidden_states[hidden_layer]
                            logits = llm_model.lm_head(hidden_states[:, -1, :])
                            probs = torch.nn.functional.softmax(logits, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                            generated_tokens_cem_int_sub = torch.cat((generated_tokens_cem_int_sub, next_token), dim=-1)

                            output_text_cem_int_sub = tokenizer.decode(generated_tokens_cem_int_sub[0], skip_special_tokens=True)
                        inputs_sub = tokenizer(clean_response(output_text_cem_int_sub, prompt), return_tensors = "pt").to("cuda:1")
                        
                        with torch.no_grad():
                                output_sub = llm_model(**inputs_sub, output_hidden_states=True)  
                        
                        last_embedding_sub = output_sub.hidden_states[hidden_layer][-1]
                        
                        x_input_sub = last_embedding_sub.to("cpu").to(next(best_cem.parameters()).dtype)
                        embs_cem_int = best_cem(x_input_sub)[0]
                        print(f"emb modificato {embs_cem_int.shape}")

                
                else:
                    logits = llm_model.lm_head(hidden_states[:, -1, :])
                    
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated_tokens_cem_int = torch.cat((generated_tokens_cem_int, next_token), dim=-1)
                    output_text_cem_int = tokenizer.decode(generated_tokens_cem_int[0], skip_special_tokens=True)

        # dictionary
        cem_llama[j]["llama"] = data[j][1]
        if changes_inds:
            cem_llama[j]["cem-llama"] = clean_response(output_text_cem_none, prompt)
            cem_llama[j]["changes"] = changes_inds

        if llama_seed:
            cem_llama[j]["cem-llama-seed"] = clean_response(output_text_llama_seed, prompt)

        if intervention_idx:
            cem_llama[j]["cem-llama-int"] = clean_response(output_text_cem_int, prompt)
            active_concepts = ["hateful", "truthful", "formal"]
            inactive_concepts = ["kind", "lie", "informal"]
            concepts = []
            c_true_concept = c_true.squeeze().tolist()
            for int_ind in intervention_idx:
                if c_true_concept[int_ind]:
                    concepts.append(active_concepts[int_ind])
                else:
                    concepts.append(inactive_concepts[int_ind])

            cem_llama[j]['interventions'] = c_true, concepts
        
        if changes_inds or llama_seed:
            model_sim = SentenceTransformer('all-MiniLM-L6-v2', device = device)
            emb_sim_llama = model_sim.encode(cem_llama[j]["llama"], device = device).reshape(1, -1)

            if changes_inds:
                emb_sim_cem = model_sim.encode(cem_llama[j]["cem-llama"], device = device).reshape(1, -1)
                similarity_cem = cosine_similarity(emb_sim_cem, emb_sim_llama)[0][0]
                cem_llama[j]["similarity-cem"] = float(similarity_cem)
            if llama_seed:
                emb_sim_seed = model_sim.encode(cem_llama[j]["cem-llama-seed"]).reshape(1, -1)
                similarity_seed = cosine_similarity(emb_sim_seed, emb_sim_llama)[0][0]
                cem_llama[j]["similarity-seed"] = float(similarity_seed)

    for key in inv_key:
        del cem_llama[key]

    return cem_llama


def labels_to_binary(labels, n_concepts):
    vector = torch.zeros(n_concepts)
    for i in range(n_concepts):
        if "not" not in labels[i]:
            vector[i] = 1
    return vector


def data_loader(dataset):
    c = torch.stack([
        labels_to_binary(dataset[i]["Label"].split(", "), 3)
        for i in range(len(dataset))
    ])

    c_rep = []
    for i in range(len(dataset)):
        c_rep.append(c[i].repeat(dataset[i]["Embedding"].shape[0], 1))

    c = torch.cat(c_rep, dim = 0)
    c = c.float() 
    print(c.shape)
    x = torch.cat([dataset[i]["Embedding"] for i in range(len(dataset))], dim = 0).float()
    x_train = torch.cat([dataset[i]["Embedding"] for i in range(8000)], dim = 0).float()
    x_tr_s = x_train.shape[0]
    c_train = c[: x_tr_s, :]
    x_valid = torch.cat([dataset[i]["Embedding"] for i in range(8000, 8500)], dim = 0).float()
    x_v_s = x_valid.shape[0] + x_tr_s
    c_valid = c[x_tr_s : x_v_s, :]
    x_test = torch.cat([dataset[i]["Embedding"] for i in range(8500, 10000)], dim = 0).float()
    c_test = c[x_v_s : , :]


    train_dataset = TensorDataset(x_train, c_train)
    train_loader = DataLoader(train_dataset, batch_size=2000, shuffle = True)
    valid_dataset = TensorDataset(x_valid, c_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=2000)
    test_dataset = TensorDataset(x_test, c_test)
    test_loader = DataLoader(test_dataset, batch_size=2000)

    return {'train_loader' : train_loader, 'valid_loader' : valid_loader, 'test_loader' : test_loader, 'n_concepts' : c_train.shape[1], 'emb_size' : x_train.shape[1]}





def training_cem(weights, layers : list, epoches, emb_size, n_concepts, device, data, 
                 path, save_path, learning_rate = 0.0001, seed = 42, perm = False, save = True):

    best_loss = float('inf')  
    best_weights = None
    best_layer = None
    best_cem_path = path
    train_loader = data['train_loader']
    valid_loader = data['valid_loader']
    test_loader = data['test_loader']
    weights_per = None
    w_length = 1

    if perm:
        weights_per = list(itertools.product(weights, repeat = 3))
        w_length = len(weights_per)

    for i in range(w_length):
        print(i)

        if weights_per:
            w = weights_per[i]
        else:
            w = weights

        for layer in layers:
            print(layer)
            model_l = GenerativeCEM(
                emb_size, n_concepts, layer,
                w_concept_loss=w[0], w_loss=w[1], w_concept_loss_int=w[2], device_gpu = device[0]
            )
            
            pl.seed_everything(seed)

            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                monitor="val_loss", mode="min", save_weights_only=True
            )
            
            trainer = Trainer(
                max_epochs = epoches, accelerator = device[1], devices = 1, precision=32,
                enable_checkpointing = True, callbacks = [checkpoint_callback]
            )
            
            trainer.fit(model_l, train_loader, valid_loader)
            model_l.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'])
            model_l.eval()
            
            test_results = trainer.test(model_l, test_loader)[0]
            print(test_results)
            test_base_loss = test_results["test_base_loss_epoch"]

            
            if test_base_loss < best_loss:
                best_layer = layer
                best_weights = w
                best_loss = test_base_loss
                test_loss = test_results["test_loss_epoch"]
                test_concept_loss = test_results["test_concept_loss_epoch"]
                test_concept_loss_int = test_results["test_concept_loss_int_epoch"]
                test_mse_loss = test_results["test_mse_loss_epoch"]               
                torch.save(model_l.state_dict(), best_cem_path)


    final_results = {'weights' : best_weights, 'layer' : best_layer} 
    overall_res = {"test loss" : best_loss, "weighted loss" : test_loss,
                   "weights" : best_weights, "layers" : best_layer, 
                   "test concept loss" : test_concept_loss, "test concept loss int" : test_concept_loss_int,
                   "test concept mse loss" : test_mse_loss}
    
    checkpoint = torch.load(checkpoint_callback.best_model_path)
    print(checkpoint['epoch'])


    if save:
        with open(save_path, 'wb') as f:
            pickle.dump(final_results, f)

        print("results salvati con successo!")
    
    return overall_res


def generate_prompts(n=500):
    topics = [
    "Write a story about", "Explain the concept of", "Generate Python code for", 
    "Create a dialogue between", "Describe the future of", "Give advice on", 
    "List the pros and cons of", "Provide a summary of", "Answer the question:",
    "Imagine a world where"
    ]

    subjects = [
    "artificial intelligence", "a talking cat", "a futuristic city", "an innovative startup", 
    "general relativity", "a cooking robot", "an intergalactic detective", "an ancient mystery", 
    "a technological revolution", "a society without money", "a habitable planet"
    ]
    prompts = [f"{random.choice(topics)} {random.choice(subjects)}." for _ in range(n)]
    return prompts
