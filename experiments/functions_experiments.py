import os
import sys

anycbm_path = "/home/fpodrecca/AnyCBM/anycbm"
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + anycbm_path
sys.path.append(anycbm_path)

from huggingface_hub import login
import pickle
import torch
from models import GenerativeCEM
from transformers import pipeline 
import re
from functions_cem import emb_rep
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

def increase_changes(parameter_path, model_path, llm_model, indxs, res_llama, tokenizer, hidden_layer, cem_layer = 400, seed = False):
   
    with open(parameter_path, 'rb') as f:
        pars = pickle.load(f)

    cem_model = GenerativeCEM(3072, 3, cem_layer, 
                              w_concept_loss = pars['weights'][0], w_loss = pars['weights'][1], 
                              w_concept_loss_int = pars['weights'][2], device_gpu = True)
    cem_model.load_state_dict(torch.load(model_path))
    cem_model.eval()

    sim = {}
    sim_seed = {}

    for idx in indxs:
    
        res = emb_rep(res_llama, changes_inds= idx, hidden_layer = hidden_layer, tokenizer = tokenizer, 
                            llm_model = llm_model, best_cem = cem_model, n_gen = 100, llama_seed = seed)
        sim[int(len(idx))] = [res[j]["similarity-cem"] for j in res.keys()]
        if seed:
            sim_seed[int(len(idx))] = [res[j]["similarity-seed"] for j in res.keys()]

            

    return sim, sim_seed


def classification(parameter_path, model_path, tokenizer, llm_model, c_true, intervention_idxs, res_llama, hidden_layer, device='cuda:1'):


    with open(parameter_path, 'rb') as f:
        pars = pickle.load(f)

    cem_model = GenerativeCEM(3072, 3, 400,
                              w_concept_loss=pars['weights'][0],
                              w_loss=pars['weights'][1],
                              w_concept_loss_int=pars['weights'][2],
                              device_gpu=True)
    cem_model.load_state_dict(torch.load(model_path))
    cem_model.eval()

    classifier = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)

    y_true = []
    y_pred = []


    res = emb_rep(
        res_llama,
        c_true=torch.tensor([c_true], dtype=torch.float32),
        intervention_idx=intervention_idxs,
        tokenizer=tokenizer,
        llm_model=llm_model,
        best_cem=cem_model,
        hidden_layer= hidden_layer,
        device=device
    )

    for k in res.keys():
        if "cem-llama-int" not in res[k]:
            print(f"⚠️ Skipping sample {k}, missing 'cem-llama-int'")
            continue
        sentence = res[k]["cem-llama-int"]
        gold_label = res[k]['interventions'][1]  # già singola label
        print(f'gold label {gold_label}')
        prompt = (
            "Classify the following sentence according to these three dimensions:\n"
            "- hateful or kind\n"
            "- truthful or lie\n"
            "- formal or informal\n"
            "Respond in the format:\n"
            "Hatefulness: <hateful/kind>\n"
            "Truthfulness: <truthful/lie>\n"
            "Formality: <formal/informal>\n"
            f"\nSentence: {sentence}\n"
            "Answer:"
        )

        output = classifier(prompt, max_new_tokens=50, do_sample=False)[0]['generated_text']
        answer_part = output.split("Answer:")[-1].strip()

        labels = re.findall(r'Hatefulness:\s*(\w+)\s*|Truthfulness:\s*(\w+)\s*|Formality:\s*(\w+)', answer_part)
        parsed_labels = [label for group in labels for label in group if label]

        if len(parsed_labels) == 3:
            y_true.append(gold_label)
            label_to_dim = {
            "hateful": 0, "kind": 0,
            "truthful": 1, "lie": 1,
            "formal": 2, "informal": 2
                }
            dim = label_to_dim[gold_label[0]]
            y_pred.append(parsed_labels[dim])
        else:
            print(f"⚠️ Parsing failed for:\n{sentence}\nOutput:\n{output}\n")


    assert len(y_true) == len(y_pred), f"Mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"

    return y_true, y_pred

def classify_external_sentence(sentence, classifier, dimension):
    """
    Classifica una frase esterna secondo una dimensione.

    Parameters:
    - sentence (str): La frase da classificare.
    - gold_label (str): La label corretta (y_true).
    - classifier: pipeline Hugging Face (text-generation).
    - dimension (str): hateful/truthful/formal (scegli uno tra 'hatefulness', 'truthfulness', 'formality').

    Returns:
    - tuple: (gold_label, predicted_label)
    """

    prompt = (
        "Classify the following sentence according to these three dimensions:\n"
        "- hateful or kind\n"
        "- truthful or lie\n"
        "- formal or informal\n"
        "Respond in the format:\n"
        "Hatefulness: <hateful/kind>\n"
        "Truthfulness: <truthful/lie>\n"
        "Formality: <formal/informal>\n"
        f"\nSentence: {sentence}\n"
        "Answer:"
    )

    output = classifier(prompt, max_new_tokens=50, do_sample=False)[0]['generated_text']
    answer_part = output.split("Answer:")[-1].strip()

    # Regex parsing
    labels = re.findall(r'Hatefulness:\s*(\w+)\s*|Truthfulness:\s*(\w+)\s*|Formality:\s*(\w+)', answer_part)
    parsed_labels = [label for group in labels for label in group if label]

    if len(parsed_labels) != 3:
        print(f"⚠️ Parsing failed:\n{sentence}\nOutput:\n{output}\n")
        return None

    dim_index = {
        "hatefulness": 0,
        "truthfulness": 1,
        "formality": 2
    }
    if dimension == 'all':
        return parsed_labels
    idx = dim_index.get(dimension.lower())
    if idx is None:
        raise ValueError(f"Dimension '{dimension}' non valida. Usa: hatefulness, truthfulness, formality")

    return parsed_labels[idx]

def get_dimension_from_label(label):
    label = label.lower()
    if label in ['hateful', 'kind']:
        return 'hatefulness'
    elif label in ['truthful', 'lie']:
        return 'truthfulness'
    elif label in ['formal', 'informal']:
        return 'formality'
    else:
        raise ValueError(f"Etichetta '{label}' non riconosciuta.")

def anycem_classifier(parameter_path, model_path, res_llama, labels, hidden_layer, active = 1, inactive = 0):
    with open(parameter_path, 'rb') as f:
        pars = pickle.load(f)

    cem_model = GenerativeCEM(3072, 3, 400,
                    w_concept_loss=pars['weights'][0],
                    w_loss=pars['weights'][1],
                    w_concept_loss_int=pars['weights'][2],
                    device_gpu=True)
    cem_model.load_state_dict(torch.load(model_path))
    cem_model.eval()
    labels_tensor = torch.ones([len(res_llama.keys()), 3]) * -1
    for key, label in labels.items():
        if label:
            if label[0] == 'hateful':
                label[0] = active
                labels_tensor[key, 0] = active
            else:
                label[0] = inactive
                labels_tensor[key, 0] = inactive
            if label[1] == 'truthful':
                label[1] = active
                labels_tensor[key, 1] = active
            else:
                label[1] = inactive
                labels_tensor[key, 1] = inactive

            if label [2] == 'formal':
                label[2] = active
                labels_tensor[key, 2] = active
            else:
                label[2] = inactive
                labels_tensor[key, 2] = inactive

    c_preds_tot = torch.ones([len(res_llama.keys()), 50, 3]) * -1
    null_keys = [k for k, v in labels.items() if v is None]
    cem_labels = {k:{0:0, 1:0, 2:0} for k in range(50)}
    acc_labels = {k:0 for k in range(50)}
    for j in range(len(res_llama.keys())):
        if labels[j]:
            
            if hidden_layer == 10:
                embs = res_llama[j][2]
            if hidden_layer == 20:
                embs = res_llama[j][3]
            if hidden_layer == -1:
                embs = res_llama[j][4]
            

            x_input = embs.to("cpu").to(next(cem_model.parameters()).dtype)
            with torch.no_grad():
                reconstructed_input, c_preds = cem_model(x_input)
            
            c_preds_tot[j, :c_preds.shape[0]] = c_preds

            for token in range(c_preds.shape[0]):
                acc_labels[token] += 1
            

    for i in range(50):
        for j in range(3):
            index_to_keep_token = c_preds_tot[:, i, j] != -1
            index_to_keep_sent = labels_tensor[:, j] != -1
            print(f'index token {index_to_keep_token.shape}')
            print(f'index sent {index_to_keep_sent.shape}')
            index_to_keep = (index_to_keep_token.float() * index_to_keep_sent.float()).bool()
            c_preds_tot_tmp = c_preds_tot[:, i, j][index_to_keep]
            labels_tmp = labels_tensor[:, j][index_to_keep]
            if labels_tmp.shape[0] == 0:
                cem_labels[i][j] = None
            else:
                cem_labels[i][j] = roc_auc_score(labels_tmp, c_preds_tot_tmp)
    

    return cem_labels


