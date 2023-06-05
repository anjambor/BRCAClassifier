setwd('/Users/alexanderjambor/Desktop/UCSD/SP23/BENG203/GroupProject/BRCAClassifier/')
# setwd('/home/ethan/homework/cse283/BRCAClassifier/')

library(Seurat)
library(dplyr)
library(edgeR)
library(Matrix)

# there's not really a reason to read this using seurat, but it's fine
normal = read.table('data/initial/GSE131512_normalTPM.txt')
normal_seurat = CreateSeuratObject(counts = normal)
normal_seurat$group = "Normal"
rm(normal)

brca = read.table('data/initial/GSE131512_cancerTPM.txt')
brca_seurat = CreateSeuratObject(counts = brca)
brca_seurat$group = "BRCA"
rm(brca)

all_seurat = merge(normal_seurat, brca_seurat)
rm(normal_seurat)
rm(brca_seurat)

counts = all_seurat@assays$RNA@counts
dge = DGEList(counts = counts, group = all_seurat$group)

keep = filterByExpr(dge)
dge = dge[keep, ,keep.lib.sizes=FALSE]
print(dim(dge))

des = model.matrix(~ all_seurat$group)
dge = estimateDisp(dge, des, trend.method='locfit')

fit = glmQLFit(dge, des)
glm_res = glmQLFTest(fit)
adj_pvals = p.adjust(glm_res$table$PValue, method = "BY")

fdr_threshold = 0.05
lfc_threshold = 1
degs = glm_res$table[((adj_pvals <= fdr_threshold) & (abs(glm_res$table$logFC) >= lfc_threshold)), ]

ensg_ids = data.frame(rownames(degs))
colnames(ensg_ids) = NULL
print(dim(ensg_ids))
write.csv(ensg_ids, 'features/diff_exp_genes.csv', row.names = FALSE)
