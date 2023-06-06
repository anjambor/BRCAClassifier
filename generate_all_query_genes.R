setwd('/Users/alexanderjambor/Desktop/UCSD/SP23/BENG203/GroupProject/BRCAClassifier/')

library(Seurat)
library(dplyr)
library(edgeR)
library(Matrix)
library(biomaRt)

fusion_genes = read.table('features/fusion_genes.csv', header = TRUE, sep = ',')
fusion_genes = union(fusion_genes$ENSGa, fusion_genes$ENSGb) # 26 ENSG IDs

# diff_exp_genes = read.table('features/recurrent_vs_nonrecurrent/diff_exp_genes.csv')[[1]] # 3052 ENSG IDs

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

#all_genes = data.frame(union(union(literature_genes, fusion_genes), diff_exp_genes)) # 772 ENSG IDs

all_genes = data.frame(union(literature_genes, fusion_genes)) # 772 ENSG IDs
# 711 for recurrent vs. non-recurrent

colnames(all_genes) = NULL
write.csv(all_genes, 'features/recurrent_vs_nonrecurrent/all_genes.csv', row.names = FALSE)
