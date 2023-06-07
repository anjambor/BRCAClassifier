import pickle
import pandas as pd
import numpy as np
import os

os.chdir('/Users/alexanderjambor/Desktop/UCSD/SP23/BENG203/GroupProject/BRCAClassifier')
target_dir = f'data/processed/'
tag = 'normal_vs_cancer'

#############################

normal_1 = pd.read_csv('data/initial/GSE131512_normalTPM.txt', delimiter='\t')
brca_1 = pd.read_csv('data/initial/GSE131512_cancerTPM.txt',  delimiter='\t')

query_genes = pd.read_csv('features/normal_vs_cancer/all_genes.csv', header=None)
query_genes = set(query_genes.iloc[:, 0])

normal_filtered = normal_1[normal_1.index.isin(query_genes)]
brca_filtered = brca_1[brca_1.index.isin(query_genes)]

all_filtered = pd.merge(brca_filtered, normal_filtered, left_index=True, right_index=True)
idx_to_ensg = {idx: ensg for idx, ensg in enumerate(all_filtered.index)}  # normal, brca same index
idx_to_sample = {idx: sample for idx, sample in enumerate(all_filtered.columns)}

pd.to_pickle(normal_filtered, f'{target_dir}/1/{tag}/normalTPM_filtered.pkl')
pd.to_pickle(brca_filtered, f'{target_dir}/1/{tag}/cancerTPM_filtered.pkl')
pd.to_pickle(all_filtered, f'{target_dir}/1/{tag}/allTPM_filtered.pkl')

with open(f'{target_dir}/1/{tag}/idx_to_ensg.pkl', 'wb') as f:
    pickle.dump(idx_to_ensg, f)

with open(f'{target_dir}/1/{tag}/idx_to_sample.pkl', 'wb') as f:
    pickle.dump(idx_to_sample, f)

############################

data2 = pd.read_csv('../validation_exon_tpm', delimiter='\t')
data2_filtered = data2[data2.index.isin(query_genes)]
data2_filtered = data2_filtered.reindex(all_filtered.index)

normal_metadata = pd.read_csv('../validation_normal_meta.csv')
brca_metadata = pd.read_csv('../validation_bc_meta.csv')

normal_ids = set(normal_metadata['Mapping ID'])
brca_ids = set(brca_metadata['Mapping ID'])

idx_to_sample2 = {idx: sample for idx, sample in enumerate(data2_filtered.columns)}

with open(f'{target_dir}/2/{tag}/idx_to_sample.pkl', 'wb') as f:
    pickle.dump(idx_to_sample, f)

###################################################################################

all_filtered = all_filtered.transpose()
X_filtered = np.matrix(all_filtered)
y_filtered = np.matrix([1 if label[0] == 'C' else 0 for label in all_filtered.index]).astype('uint8')

np.savetxt(f'{target_dir}/1/{tag}/X_filtered.csv', X_filtered, fmt='%f', delimiter=',')
np.savetxt(f'{target_dir}/1/{tag}/y_filtered.csv', y_filtered, fmt='%i', delimiter=',')

#####

data2_filtered = data2_filtered.transpose()
X2_filtered = np.matrix(data2_filtered)
y2_filtered = np.matrix([1 if label in brca_ids else 0 for label in data2_filtered.index]).astype('uint8')

np.savetxt(f'{target_dir}/2/{tag}/X_filtered.csv', X2_filtered, fmt='%f', delimiter=',')
np.savetxt(f'{target_dir}/2/{tag}/y_filtered.csv', y2_filtered, fmt='%i', delimiter=',')

