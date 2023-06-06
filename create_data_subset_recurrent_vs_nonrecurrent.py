import pickle
import pandas as pd
import numpy as np
import os

os.chdir('/Users/alexanderjambor/Desktop/UCSD/SP23/BENG203/GroupProject/BRCAClassifier')

brca = pd.read_csv('data/initial/GSE131512_cancerTPM.txt',  delimiter='\t')

query_genes = pd.read_csv('features/recurrent_vs_nonrecurrent/all_genes.csv', header=None)
query_genes = set(query_genes.iloc[:, 0])

brca_filtered = brca[brca.index.isin(query_genes)]

idx_to_ensg = {idx: ensg for idx, ensg in enumerate(brca_filtered.index)}  # normal, brca same index
idx_to_sample = {idx: sample for idx, sample in enumerate(brca_filtered.columns)}

pd.to_pickle(brca_filtered, 'data/processed/recurrent_vs_nonrecurrent/cancerTPM_filtered.pkl')

with open('data/processed/recurrent_vs_nonrecurrent/idx_to_ensg.pkl', 'wb') as f:
    pickle.dump(idx_to_ensg, f)

with open('data/processed/recurrent_vs_nonrecurrent/idx_to_sample.pkl', 'wb') as f:
    pickle.dump(idx_to_sample, f)

brca_filtered = brca_filtered.transpose()

X_filtered = np.matrix(brca_filtered)
y_filtered = np.matrix([1 if int(label[1:]) <= 28 else 0 for label in brca_filtered.index]).astype('uint8')

np.savetxt('data/processed/recurrent_vs_nonrecurrent/X_filtered.csv', X_filtered, fmt='%f', delimiter=',')
np.savetxt('data/processed/recurrent_vs_nonrecurrent/y_filtered.csv', y_filtered, fmt='%i', delimiter=',')
