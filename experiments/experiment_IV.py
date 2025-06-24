from functions_experiments import anycem_classifier
import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('prompt_emb_last.pkl', 'rb') as f:
    data = pickle.load(f)

sent = {}
labels = {}

for i in range(500):
    sent[i] = (((),(), (), (), data[9499 + i]['Embedding']))
    labels[i] = [item.strip() for item in data[9499 + i]['Label'].split(',')]
res_i = anycem_classifier('parameter_last.pkl', 'best_cem_last.pth', sent, labels, -1, active = 1, inactive = 0)
res_ii = anycem_classifier('parameter_30_ii.pkl', 'best_cem_30_ii.pth', sent, labels, -1, active = 1, inactive = 0)
res_iii = anycem_classifier('parameter_30_iii.pkl', 'best_cem_30_iii.pth', sent, labels, -1, active = 1, inactive = 0)


res = {}

for outer_key in range(41):
    res[outer_key] = {}
    for inner_key in res_i[outer_key]:

        v1 = res_i[outer_key].get(inner_key, np.nan)
        v2 = res_ii[outer_key].get(inner_key, np.nan)
        v3 = res_iii[outer_key].get(inner_key, np.nan)

      
        mean_token = np.nanmean([v1, v2, v3])

        res[outer_key][inner_key] = mean_token
