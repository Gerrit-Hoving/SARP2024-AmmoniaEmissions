# Following this tutorial: https://www.statology.org/principal-components-analysis-in-r/


library(tidyverse)
library(readxl)
library(dbplyr)

manure_pond <- read_excel("C:\\Users\\sarp\\Documents\\sarp_research\\data\\3CData.xlsx", sheet = "ManurePond")

#manure_pond = t(manure_pond)

colnames(manure_pond) <- as.character(manure_pond[1, ])
manure_pond <- manure_pond[,-1]
manure_pond = t(manure_pond)

#head(manure_pond[,2])
#head(manure_pond)

results <- prcomp(manure_pond, scale = TRUE)

results$rotation <- -1*results$rotation

results$rotation

results$sdev^2 / sum(results$sdev^2)





manure_pond_mat <- as.matrix(manure_pond[ ,-1])

# Add row names to your matrix from original data ste
rownames(manure_pond_mat) <- manure_pond[ ,1]

# Keep complete cases only

manure_pond_mat <- manure_pond_mat[complete.cases(manure_pond_mat),]


results <- prcomp(manure_pond_mat, scale = TRUE)




