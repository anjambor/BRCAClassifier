setwd('/Users/alexanderjambor/Desktop/UCSD/SP23/BENG203/GroupProject/BRCAClassifier/')

library(Seurat)
library(dplyr)
library(edgeR)
library(Matrix)

normal = read.table('data/GSE131512_normalTPM.txt')
normal_seurat = CreateSeuratObject(counts = normal)
normal_seurat$group = "Normal"
rm(normal)

brca = read.table('data/GSE131512_cancerTPM.txt')
brca_seurat = CreateSeuratObject(counts = brca)
brca_seurat$group = "BRCA"
rm(brca)

all_seurat = merge(normal_seurat, brca_seurat)
rm(normal_seurat)
rm(brca_seurat)
 
dge = DGEList(counts = all_seurat@assays$RNA@counts, group = all_seurat$group)

# dge = calcNormFactors(dge)
# dge = dge[filterByExpr(dge), keep.lib.sizes = FALSE]

des = model.matrix(~ all_seurat$group)
dge = estimateDisp(dge, des)
fit = glmQLFit(dge, des)
glm_res = glmQLFTest(fit)

adj_pvals = p.adjust(glm_res$table$PValue, method = "BH")
fdr_threshold = 0.05

degs = glm_res$table[(adj_pvals <= fdr_threshold) & (abs(glm_res$table$logFC) >= 1), ]
ensg_ids = data.frame(rownames(degs))
colnames(ensg_ids) = NULL
write.csv(ensg_ids, 'features/diff_exp_genes.csv', row.names = FALSE)
