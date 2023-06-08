import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cancer = pd.read_csv('data/initial/GSE131512_cancerTPM.txt', sep='\t')
normal = pd.read_csv('data/initial/GSE131512_normalTPM.txt', sep='\t')

recur_samples = cancer.columns[:28]
recur = cancer[recur_samples]
cancer = cancer[np.setdiff1d(cancer.columns, recur_samples)]

count_matrix = pd.concat([normal, recur, cancer], axis=1)

# I stole this from ChatGPT

# Define your file paths.
gtf_file = 'data/initial/human_ensemble_genes.gtf'

# Load your count matrix.
# count_matrix = pd.read_csv(count_matrix_file, index_col=0)

# Load GTF as a dataframe, with column names following GTF format.
gtf_columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attributes']
gtf_data = pd.read_csv(gtf_file, sep='\t', comment='#', names=gtf_columns)

# Filter for gene features and extract gene_id from attributes.
gtf_data = gtf_data[gtf_data['feature'] == 'gene']
gtf_data['gene_id'] = gtf_data['attributes'].str.extract('gene_id "([^"]+)"')

# Create a dictionary to map gene ids to chromosomes.
gene_to_chromosome = pd.Series(gtf_data.seqname.values,index=gtf_data.gene_id).to_dict()

# Add a chromosome column to the count matrix.
count_matrix['chromosome'] = pd.Series(gene_to_chromosome)

# Calculate TPM for each gene in each sample.
# Here, I'm assuming that the counts are in all columns except the new chromosome column.
for col in count_matrix.columns:
    if col != 'chromosome':
        count_matrix[col] = (count_matrix[col] / count_matrix[col].sum()) * 1e6

# Group by chromosome and sum to get TPM per chromosome.
chromosome_tpm = count_matrix.groupby('chromosome').sum()

# Write the result to a CSV file.


use_chroms = chromosome_tpm.sum(axis=1) > 50000
# remove low occurance chroms 

chromosome_tpm = np.log2(chromosome_tpm[use_chroms]+1)
# chromosome_tpm = (chromosome_tpm *1e6)/chromosome_tpm.sum(axis=0)

from scipy.stats import differential_entropy
entropy = chromosome_tpm.apply(differential_entropy, axis=0)
chromosome_tpm = pd.concat([chromosome_tpm, entropy.T], axis=0)

chromosome_tpm.to_csv('features/chromosome_tpm.csv')

# chroms_use = ['7', ]
#               # '5', '13', '18', '9', '8', '12']
# chromosome_tpm.loc[chroms_use].to_csv('features/min_chrom.csv')

# chroms_for_recur = ['1', '10', '17', '21']

# chromosome_tpm.loc[chroms_for_recur].to_csv('features/recur_chrom.csv')