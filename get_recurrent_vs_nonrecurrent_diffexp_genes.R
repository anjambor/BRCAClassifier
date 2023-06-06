setwd('/Users/alexanderjambor/Desktop/UCSD/SP23/BENG203/GroupProject/BRCAClassifier/')
# setwd('/home/ethan/homework/cse283/BRCAClassifier/')

library(Seurat)
library(dplyr)
library(edgeR)
library(Matrix)



brca = read.table('data/initial/GSE131512_cancerTPM.txt')
brca_seurat = CreateSeuratObject(counts = brca)
brca_seurat$group = c(rep("recurrent", 28), rep("non-recurrent", 68))
rm(brca)

counts = brca_seurat@assays$RNA@counts
dge = DGEList(counts = counts, group = brca_seurat$group)

keep = filterByExpr(dge)
dge = dge[keep, ,keep.lib.sizes=FALSE]
print(dim(dge))

des = model.matrix(~ brca_seurat$group)
dge = estimateDisp(dge, des, trend.method='locfit')

fit = glmQLFit(dge, des)
glm_res = glmQLFTest(fit)
adj_pvals = p.adjust(glm_res$table$PValue, method = "BY")

fdr_threshold = 0.10
lfc_threshold = 1
degs = glm_res$table[((adj_pvals <= fdr_threshold) & (abs(glm_res$table$logFC) >= lfc_threshold)), ]

ensg_ids = data.frame(rownames(degs))
colnames(ensg_ids) = NULL
print(dim(ensg_ids))
write.csv(ensg_ids, 'features/recurrent_vs_nonrecurrent/diff_exp_genes.csv', row.names = FALSE)
