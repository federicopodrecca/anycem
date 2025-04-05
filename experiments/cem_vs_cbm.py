import torch
import pickle
from torch.utils.data import TensorDataset, DataLoader
from models import GenerativeCEM, CBM
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import numpy as np

# Caricamento Dataset e preparazione Train, Valid, Test

with open('prompt_emb.pkl', 'rb') as f:
    loaded_prompt_emb = pickle.load(f)

print("prompt_emb caricato con successo!")


def labels_to_binary(labels, n_concepts):
    vector = torch.zeros(n_concepts)
    for i in range(n_concepts):
        if "not" not in labels[i]:
            vector[i] = 1
    return vector


c = torch.stack([
    labels_to_binary(loaded_prompt_emb[i]["Label"].split(", "), 3)
    for i in range(len(loaded_prompt_emb))
])

c_rep = []
for i in range(len(loaded_prompt_emb)):
    c_rep.append(c[i].repeat(loaded_prompt_emb[i]["Embedding"].shape[0], 1))

c = torch.cat(c_rep, dim = 0)
c = c.float() 
print(c.shape)
x = torch.cat([loaded_prompt_emb[i]["Embedding"] for i in range(len(loaded_prompt_emb))], dim = 0).float()

x_train = torch.cat([loaded_prompt_emb[i]["Embedding"] for i in range(700)], dim = 0).float()
x_tr_s = x_train.shape[0]
c_train = c[: x_tr_s, :]
x_valid = torch.cat([loaded_prompt_emb[i]["Embedding"] for i in range(700, 850)], dim = 0).float()
x_v_s = x_valid.shape[0] + x_tr_s
c_valid = c[x_tr_s : x_v_s, :]
x_test = torch.cat([loaded_prompt_emb[i]["Embedding"] for i in range(850, 1000)], dim = 0).float()
c_test = c[x_v_s : , :]


train_dataset = TensorDataset(x_train, c_train)
train_loader = DataLoader(train_dataset, batch_size=2000, shuffle = True)
valid_dataset = TensorDataset(x_valid, c_valid)
valid_loader = DataLoader(valid_dataset, batch_size=2000)
test_dataset = TensorDataset(x_test, c_test)
test_loader = DataLoader(test_dataset, batch_size=2000)

# Modelli
results = {}
learning_rate = 0.001
n_concepts = c_train.shape[1]
emb_size = x_train.shape[1]
latent_layer_size = 128 
epoches = 200 
seed = [0, 2, 6, 8, 42]

for s in seed:
    models = [GenerativeCEM(emb_size, n_concepts, latent_layer_size),
              CBM(emb_size, n_concepts, latent_layer_size)]
    results[s] = {}
    for model in models:
        results[s][model.__class__.__name__] = {}
        pl.seed_everything(s)

        if model.__class__.__name__ == 'GenerativeCEM':

            checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_weights_only=True)
            trainer = Trainer(max_epochs= epoches, accelerator='cpu', precision=32, enable_checkpointing=True, callbacks=checkpoint_callback)
            trainer.fit(model, train_loader, valid_loader)
            model.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'])
            model.eval()
            test_results = trainer.test(model, test_loader)[0]

            test_acc = test_results["test_concept_acc_epoch"]
            best_acc = 0
            best_cem_path = "best_cem_model.pth"
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), best_cem_path)
                print(f"Miglior modello CEM salvato con accuracy {best_acc}")
                
            results[s][model.__class__.__name__] = {"test_concept_accuracy" : test_results["test_concept_acc_epoch"], 
                                                    "test_loss" : test_results["test_loss_epoch"]}
        
        else:

            checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_weights_only=True)
            trainer = Trainer(max_epochs=3, accelerator='cpu', precision=32, enable_checkpointing=True, callbacks=checkpoint_callback)
            trainer.fit(model, train_loader, valid_loader)
            model.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'])
            model.eval()
            test_results = trainer.test(model, test_loader)[0]

            results[s][model.__class__.__name__] = {"test_concept_accuracy" : test_results["concept_acc_epoch"], 
                                                    "test_loss" : test_results["test_loss_epoch"]}


cem_acc = []
cem_loss = []
cbm_acc = []
cbm_loss = []

for j in seed:
    cem_acc.append(results[j]["GenerativeCEM"]["test_concept_accuracy"])
    cem_loss.append(results[j]["GenerativeCEM"]["test_loss"])
    cbm_acc.append(results[j]["CBM"]["test_concept_accuracy"])
    cbm_loss.append(results[j]["CBM"]["test_loss"])

overall_cem_res = {"concept_accuracy" : np.mean(cem_acc), "test_loss" : np.mean(cem_loss)}
overall_cbm_res = {"concept_accuracy" : np.mean(cbm_acc), "test_loss" : np.mean(cbm_loss)}
overall_results = {"CEM" : overall_cem_res, "CBM" : overall_cbm_res}


print(overall_results)


with open('cem_vs_cbm_res.pkl', 'wb') as f:
    pickle.dump(overall_results, f)

print("results salvati con successo!")