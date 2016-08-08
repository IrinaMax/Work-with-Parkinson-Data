# Work with Parkinson Data
It is just simply my trials to analize data set with people have parkinson on the erly stage using PCA and K-clustering analysis.

## my work with Parkinson data  "http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/?C=D;O=A"
## description "http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.names" 

    park <- read.csv("http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data", header = T, stringsAsFactors = T)
    head(park, 20)
In this data I try to find the response variable; and the remaining columns are the predictor variables.


    park$subject.
    mean(park$age)
 comma delimited data and no header for each variable

    dim()
    responseYpark <- park[, 3]
    predictorXpark <- park[,1:(dim(park)[2])]
    responseYpark
    predictorXpark

    pca <- princomp(predictorXpark, cor=T, scores = TRUE)# principal components analysis using correlation matrix

pca
pc.comp <- pca$scores
pc.comp
pc.comp1 <- -1*pc.comp[, 1] # principal component 1 scores (negated for convenience)
pc.comp2 <- -1*pc.comp[, 2] # principal component 2 scores (negated for convenience)
plot(pca$scores)
pca$scores
#In R, kmeans performs the K-means clustering analysis, ()$cluster provides the clustering results and ()$centers provides the centroid vector (i.e., the mean) for each cluster.
#For the convenience of visualization, we take the first two principle components as the new feature variables and conduct k-means only on these two dimensional data.

X <- cbind(pc.comp1, pc.comp2)
cl <- kmeans(X,13)
cl$cluster
plot(pc.comp1, col=cl$cluster)
points(cl$centers, pch=15)

library(cluster)
## Plot of the dendrogram with average method
agn <- agnes(x=predictorXpark[1:50,], diss = FALSE, stand = TRUE, 
             method = "average")
DendAgn <-as.dendrogram(agn)
plot(DendAgn)

agn <- agnes(x=predictorXpark[1:25,], diss = FALSE, stand = TRUE,
             method = "single")
DendAgn <-as.dendrogram(agn)
plot(DendAgn)

agn <- agnes(x=predictorXpark[1:25,], diss = FALSE, stand = TRUE,
             method = "complete")
DendAgn <-as.dendrogram(agn)
plot(DendAgn)
