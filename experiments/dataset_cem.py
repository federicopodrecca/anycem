import pickle
import torch

def load_prompt_emb(file_path='prompt_emb.pkl'):
    # Carica il dizionario dal file Pickle
    with open(file_path, 'rb') as f:
        prompt_emb = pickle.load(f)
    
    x = []
    labels = []
    
    for i in prompt_emb:
        # Estrai gli embedding e le etichette
        x.append(prompt_emb[i]["Embedding"].squeeze().to(torch.float32).cpu().numpy())
  # Squeeze per rimuovere dimensioni superflue
        labels.append(prompt_emb[i]["Label"])

    x = torch.tensor(x, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    return x, labels