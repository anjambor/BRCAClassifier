setwd('/Users/alexanderjambor/Desktop/UCSD/SP23/BENG203/GroupProject/BRCAClassifier/')

library(Seurat)
library(dplyr)
library(edgeR)
library(Matrix)
library(biomaRt)

fusion_genes = read.table('features/fusion_genes.csv', header = TRUE, sep = ',')
fusion_genes = union(fusion_genes$ENSGa, fusion_genes$ENSGb) # 26 ENSG IDs

diff_exp_genes = read.table('features/normal_vs_cancer//diff_exp_genes.csv')[[1]] # 3052 ENSG IDs

lines = readLines('features/literature_genes.csv')
data = lapply(lines, function(line) {unlist(strsplit(line, ","))})

literature_genes = c()
for (entry in data) {
  literature_genes = c(literature_genes, unique(entry[-1]))
}

literature_genes = setdiff(unique(literature_genes), '""') # HGNC symbols

ensembl = useMart('ensembl')
dataset = useDataset('hsapiens_gene_ensembl', mart = ensembl)
attributes = c('hgnc_symbol', 'ensembl_gene_id')

literature_genes = getBM(attributes = attributes,
                         filters = 'hgnc_symbol',
                         values = literature_genes,
                         mart = dataset)

literature_genes = literature_genes$ensembl_gene_id # 689 ENSG IDs

mirna_genes = read.table('features/cancer_mirna_genes.csv', header = TRUE, sep = ',')
mirna_genes = c(mirna_genes['X'])[[1]]

metastasis_genes = read.table('features/metastasis_genes.csv', header = TRUE, sep = ',')
metastasis_genes = c(metastasis_genes['ENSG_id'])[[1]]


all_genes = union(union(literature_genes, fusion_genes), diff_exp_genes) # 772 ENSG IDs
all_genes = union(all_genes, metastasis_genes)
all_genes = union(all_genes, mirna_genes)

all_genes = data.frame(all_genes)

# 711 for recurrent vs. non-recurrent

colnames(all_genes) = NULL
write.csv(all_genes, 'features/normal_vs_cancer//all_genes.csv', row.names = FALSE)
