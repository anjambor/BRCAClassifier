import pickle
import pandas as pd
import numpy as np
import os

os.chdir('/Users/alexanderjambor/Desktop/UCSD/SP23/BENG203/GroupProject/BRCAClassifier')
target_dir = f'data/processed/'
tag = 'recurrent_vs_nonrecurrent'

brca = pd.read_csv('data/initial/GSE131512_cancerTPM.txt',  delimiter='\t')

query_genes = pd.read_csv(f'features/{tag}/all_genes.csv', header=None)
query_genes = set(query_genes.iloc[:, 0])

brca_filtered = brca[brca.index.isin(query_genes)]

idx_to_ensg = {idx: ensg for idx, ensg in enumerate(brca_filtered.index)}  # normal, brca same index
idx_to_sample = {idx: sample for idx, sample in enumerate(brca_filtered.columns)}

pd.to_pickle(brca_filtered, f'{target_dir}/1/{tag}/cancerTPM_filtered.pkl')

with open(f'{target_dir}/1/{tag}/idx_to_ensg.pkl', 'wb') as f:
    pickle.dump(idx_to_ensg, f)

with open(f'{target_dir}/1/{tag}/idx_to_sample.pkl', 'wb') as f:
    pickle.dump(idx_to_sample, f)


############################################################

data2 = pd.read_csv('../validation_exon_tpm', delimiter='\t')
data2_filtered = data2[data2.index.isin(query_genes)]
data2_filtered = data2_filtered.reindex(brca_filtered.index)

brca_metadata = pd.read_csv('../validation_bc_meta.csv')

recurrent_ids = set(brca_metadata[brca_metadata.iloc[:, 3] == 'Recurrent']['Mapping ID'])
nonrecurrent_ids = set(brca_metadata[brca_metadata.iloc[:, 3] == 'Nonrecurrent']['Mapping ID'])

idx_to_sample2 = {idx: sample for idx, sample in enumerate(data2_filtered.columns)}

with open(f'{target_dir}/2/{tag}/idx_to_sample.pkl', 'wb') as f:
    pickle.dump(idx_to_sample, f)

#############################################################

brca_filtered = brca_filtered.transpose()
X_filtered = np.matrix(brca_filtered)
y_filtered = np.matrix([1 if int(label[1:]) <= 28 else 0 for label in brca_filtered.index]).astype('uint8')

np.savetxt(f'{target_dir}/1/{tag}/X_filtered.csv', X_filtered, fmt='%f', delimiter=',')
np.savetxt(f'{target_dir}/1/{tag}/y_filtered.csv', y_filtered, fmt='%i', delimiter=',')

######

data2_filtered = data2_filtered.transpose()
X2_filtered = np.matrix(data2_filtered)
y2_filtered = np.matrix([1 if label in recurrent_ids else 0 for label in data2_filtered.index]).astype('uint8')

np.savetxt(f'{target_dir}/2/{tag}/X_filtered.csv', X2_filtered, fmt='%f', delimiter=',')
np.savetxt(f'{target_dir}/2/{tag}/y_filtered.csv', y2_filtered, fmt='%i', delimiter=',')





