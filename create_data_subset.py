import pickle
import pandas as pd
import numpy as np
import os

os.chdir('/Users/alexanderjambor/Desktop/UCSD/SP23/BENG203/GroupProject/BRCAClassifier')

normal = pd.read_csv('data/initial/GSE131512_normalTPM.txt', delimiter='\t')
brca = pd.read_csv('data/initial/GSE131512_cancerTPM.txt',  delimiter='\t')

query_genes = pd.read_csv('features/all_genes.csv', header=None)
query_genes = set(query_genes.iloc[:, 0])

normal_filtered = normal[normal.index.isin(query_genes)]
brca_filtered = brca[brca.index.isin(query_genes)]

all_filtered = pd.merge(brca_filtered, normal_filtered, left_index=True, right_index=True)
idx_to_ensg = {idx: ensg for idx, ensg in enumerate(all_filtered.index)}  # normal, brca same index
idx_to_sample = {idx: sample for idx, sample in enumerate(all_filtered.columns)}

pd.to_pickle(normal_filtered, 'data/processed/normalTPM_filtered.pkl')
pd.to_pickle(brca_filtered, 'data/processed/cancerTPM_filtered.pkl')
pd.to_pickle(all_filtered, 'data/processed/allTPM_filtered.pkl')

with open('data/processed/idx_to_ensg.pkl', 'wb') as f:
    pickle.dump(idx_to_ensg, f)

with open('data/processed/idx_to_sample.pkl', 'wb') as f:
    pickle.dump(idx_to_sample, f)

all_filtered = all_filtered.transpose()

X_filtered = np.matrix(all_filtered)
y_filtered = np.matrix([1 if label[0] == 'C' else 0 for label in all_filtered.index]).astype('uint8')

np.savetxt('data/processed/X_filtered.csv', X_filtered, fmt='%f', delimiter=',')
np.savetxt('data/processed/y_filtered.csv', y_filtered, fmt='%i', delimiter=',')
