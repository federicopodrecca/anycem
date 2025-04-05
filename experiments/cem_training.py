import torch
import pickle
from torch.utils.data import TensorDataset, DataLoader
from models import GenerativeCEM
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import random

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

learning_rate = 0.0001
n_concepts = c_train.shape[1]
emb_size = x_train.shape[1]
latent_layer_sizes = [128, 200, 300, 400] 
epoches = 500 
seed = 42
results = {}

best_loss = float('inf') 
best_weights = None
best_layer = None
best_cem_path = "best_cem_model.pth"

for w in range(100):
    weights = [round(random.uniform(0, 1), 4) for _ in range(3)]
    results_l = {}

    for layer in latent_layer_sizes:
        model_l = GenerativeCEM(
            emb_size, n_concepts, layer,
            w_concept_loss=weights[0], w_loss=weights[1], w_concept_loss_int=weights[2]
        )
        
        results_l[layer] = {}
        pl.seed_everything(seed)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss", mode="min", save_weights_only=True
        )
        
        trainer = Trainer(
            max_epochs=epoches, accelerator='gpu', devices = 1, precision=32,
            enable_checkpointing=True, callbacks=[checkpoint_callback]
        )
        
        trainer.fit(model_l, train_loader, valid_loader)
        model_l.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'])
        model_l.eval()
        
        test_results = trainer.test(model_l, test_loader)[0]
        print(test_results)
        test_base_loss = test_results["test_base_loss_epoch"]

        
        if test_base_loss < best_loss:
            best_layer = layer
            best_weights = weights
            best_loss = test_base_loss
            test_loss = test_results["test_loss_epoch"]
            torch.save(model_l.state_dict(), best_cem_path)


print(f"Features Best Model:")
print(f" - Test Loss: {best_loss}")
print(f" - Weighted Loss: {test_loss}")
print(f" - Weights: {best_weights}")
print(f" - Latent Layer Size: {best_layer}")


with open('cem_trained.pkl', 'wb') as f:
    pickle.dump(results, f)

print("results salvati con successo!")