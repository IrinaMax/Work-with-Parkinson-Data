# Analysing Parkinsons Telemonitoring Data Set.
My work with Paprkinson Telemonitoring Data set was  inspired  by many people who have this tragic disease from nowhere :( 
I always was interested to work with medical data and I really would like to do something to help effected people. Parkinson Telemonitoring Datase was created by Athanasios Tsanas (tsanasthanasis '@' gmail.com) 
and Max Little (littlem '@' physics.ox.ac.uk) of the University of Oxford, in 
collaboration with 10 medical centers in the US and Intel Corporation who 
developed the telemonitoring device to record the speech signals.

I perform PCA and K-clustering analysis with the dataset. 

This datase is composed of a range of biomedical voice measurements from 42 people with early-stage Parkinson's disease recruited to a six-month trial of a telemonitoring device for remote symptom progression monitoring. The recordings were automatically captured in the patient's homes.

 my work with Parkinson data  "http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/?C=D;O=A"
 description "http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.names" 

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
In R, kmeans performs the K-means clustering analysis, ()$cluster provides the clustering results and ()$centers provides the centroid vector (i.e., the mean) for each cluster.
For the convenience of visualization, I take the first two principle components as the new feature variables and conduct k-means only on these two dimensional data.

    X <- cbind(pc.comp1, pc.comp2)
    cl <- kmeans(X,13)
    cl$cluster
    plot(pc.comp1, col=cl$cluster)
    points(cl$centers, pch=15)

    library(cluster)
        
![plot_clust_1pca_copm_park](https://user-images.githubusercontent.com/16123495/38159815-963b5398-3465-11e8-85db-c66cb9c4654b.png)

Plot of the dendrogram with average method

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
    
![rplot_chaid_park_sex factor](https://user-images.githubusercontent.com/16123495/38159836-2afde568-3466-11e8-8d22-7bd98d25f261.png)
