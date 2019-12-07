#2019
#1. Import and Preprocess
#1.1. Import Packages
library(keras)      # package for deep learning
library(readr)      # package for reading file
library(dplyr)      # package for preprocessing
library(purrr)      # package for preprocessing
library(knitr)      # package for report generation
library(kableExtra) # package for report generation
library(caret)      # package for machine learning technique
library(yardstick)  # package for measuring model performance
library(stringr)    # package for splitting string
library(imbalance)  # package for oversampling
library(ROSE)       # package for oversampling
library(pROC)       # package for ROC
library(AppliedPredictiveModeling)
library(GGally)
library(ggplot2)
library(rgl)
library(RColorBrewer)
1.2. Read Data
dat <- read_csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"))
## Parsed with column specification:
## cols(
##   .default = col_double(),
##   name = col_character()
## )
## See spec(...) for full column specifications.
dim(dat)
## [1] 195  24
head(dat, 5)
name
<chr>
  MDVP:Fo(Hz)
<dbl>
  MDVP:Fhi(Hz)
<dbl>
  MDVP:Flo(Hz)
<dbl>
  MDVP:Jitter(%)
<dbl>
  MDVP:Jitter(Abs)
<dbl>
  MDVP:RAP
<dbl>
  phon_R01_S01_1	119.992	157.302	74.997	0.00784	0.00007	0.00370
phon_R01_S01_2	122.400	148.650	113.819	0.00968	0.00008	0.00465
phon_R01_S01_3	116.682	131.111	111.555	0.01050	0.00009	0.00544
phon_R01_S01_4	116.676	137.871	111.366	0.00997	0.00009	0.00502
phon_R01_S01_5	116.014	141.781	110.655	0.01284	0.00011	0.00655
5 rows | 1-7 of 24 columns
## number of unique values for each variable
data.frame(t(sapply(dat, function(x) length(unique(x)))))
name
<int>
  MDVP.Fo.Hz.
<int>
  MDVP.Fhi.Hz.
<int>
  MDVP.Flo.Hz.
<int>
  MDVP.Jitter...
<int>
  MDVP.Jitter.Abs.
<int>
  MDVP.RAP
<int>
  MDVP.PPQ
<int>
  195	195	195	195	173	19	155	165
1 row | 1-8 of 24 columns
## number of NAs for each variable
nasum <- sapply(dat, function(x) sum(is.na(x)))
nasum
##             name      MDVP:Fo(Hz)     MDVP:Fhi(Hz)     MDVP:Flo(Hz)
##                0                0                0                0
##   MDVP:Jitter(%) MDVP:Jitter(Abs)         MDVP:RAP         MDVP:PPQ
##                0                0                0                0
##       Jitter:DDP     MDVP:Shimmer MDVP:Shimmer(dB)     Shimmer:APQ3
##                0                0                0                0
##     Shimmer:APQ5         MDVP:APQ      Shimmer:DDA              NHR
##                0                0                0                0
##              HNR           status             RPDE              DFA
##                0                0                0                0
##          spread1          spread2               D2              PPE
##                0                0                0                0
1.3. Preprocessing
X_dat <- dat %>% select(-c(name, status)) %>% as.data.frame()
y_dat <- dat$status

dim(X_dat)
## [1] 195  22
head(X_dat, 5)


MDVP:Fo(Hz)
<dbl>
  MDVP:Fhi(Hz)
<dbl>
  MDVP:Flo(Hz)
<dbl>
  MDVP:Jitter(%)
<dbl>
  MDVP:Jitter(Abs)
<dbl>
  MDVP:RAP
<dbl>
  MDVP:PPQ
<dbl>
  1	119.992	157.302	74.997	0.00784	0.00007	0.00370	0.00554
2	122.400	148.650	113.819	0.00968	0.00008	0.00465	0.00696
3	116.682	131.111	111.555	0.01050	0.00009	0.00544	0.00781
4	116.676	137.871	111.366	0.00997	0.00009	0.00502	0.00698
5	116.014	141.781	110.655	0.01284	0.00011	0.00655	0.00908
5 rows | 1-8 of 23 columns
colnames(X_dat)[4] <- "MDVP:Jitter(perc)"

colnames(X_dat) <- gsub(" ", ".", colnames(X_dat), fixed = TRUE)
colnames(X_dat) <- gsub(",", ".", colnames(X_dat), fixed = TRUE)
colnames(X_dat) <- gsub("/", ".", colnames(X_dat), fixed = TRUE)
colnames(X_dat) <- gsub("(", ".", colnames(X_dat), fixed = TRUE)
colnames(X_dat) <- gsub(")", ".", colnames(X_dat), fixed = TRUE)
colnames(X_dat) <- gsub("&", ".", colnames(X_dat), fixed = TRUE)
colnames(X_dat) <- gsub(":", ".", colnames(X_dat), fixed = TRUE)

colnames(X_dat)
##  [1] "MDVP.Fo.Hz."       "MDVP.Fhi.Hz."      "MDVP.Flo.Hz."
##  [4] "MDVP.Jitter.perc." "MDVP.Jitter.Abs."  "MDVP.RAP"
##  [7] "MDVP.PPQ"          "Jitter.DDP"        "MDVP.Shimmer"
## [10] "MDVP.Shimmer.dB."  "Shimmer.APQ3"      "Shimmer.APQ5"
## [13] "MDVP.APQ"          "Shimmer.DDA"       "NHR"
## [16] "HNR"               "RPDE"              "DFA"
## [19] "spread1"           "spread2"           "D2"
## [22] "PPE"
## Split the data into train, valid, and test.

set.seed(2019)
trainvalid_index <- createDataPartition(y_dat, p = 0.8, list = FALSE)

X_trainvalid <- X_dat[trainvalid_index,] %>% as.matrix()
y_trainvalid <- y_dat[trainvalid_index]

X_test <- X_dat[-trainvalid_index,] %>% as.matrix()
y_test <- y_dat[-trainvalid_index]

train_index <- createDataPartition(y_trainvalid, p = 0.75, list = FALSE)

X_train <- X_trainvalid[train_index,]
y_train <- y_trainvalid[train_index]

X_valid <- X_trainvalid[-train_index,]
y_valid <- y_trainvalid[-train_index]

## Center and scale continuous variables.

prepro <- preProcess(X_train, c("center", "scale", "corr"))
X_train2 <- predict(prepro, X_train)
X_valid2 <- predict(prepro, X_valid)
X_test2 <- predict(prepro, X_test)

data.frame(head(X_train2, 5))


MDVP.Fo.Hz.
<dbl>
  MDVP.Fhi.Hz.
<dbl>
  MDVP.Flo.Hz.
<dbl>
  MDVP.Jitter.Abs.
<dbl>
  MDVP.APQ
<dbl>
  NHR
<dbl>
  HNR
<dbl>
  1	-0.8421962	-0.4640831	-0.894241909	0.6225357	0.3174545	-0.0920495	-0.2006315
2	-0.7841888	-0.5511289	-0.018077197	0.8784622	1.0985063	-0.1522386	-0.6490686
5	-0.9380241	-0.6202364	-0.089484779	1.6462416	1.1527382	-0.1868153	-0.5192337
6	-0.8287061	-0.7270719	-0.018799398	0.8784622	0.4695275	-0.3031382	-0.1212112
7	-0.8355716	-0.6658822	0.004514139	-0.4011702	-0.5882749	-0.4344016	0.6863440
5 rows | 1-8 of 13 columns
table(y_train)
## y_train
##  0  1
## 29 88
table(y_valid)
## y_valid
##  0  1
##  7 32
table(y_test)
## y_test
##  0  1
## 12 27
1.4. Distributions and Scatters of Variables
## Plot the distributions and scatters of variables.

## Matrix of plots
p1 <- ggpairs(data.frame(X_train2),
              lower = list(continuous = wrap("points", alpha = 0.5, color = c("royalblue2", "firebrick2")[y_train+1])),
              diag = list(continuous = wrap("barDiag", bins = 20, fill = "#00AFBB", color = "black")),
              upper = list(continuous = wrap("cor", color = "black", size = 4)))
# Correlation matrix plot
p2 <- ggcorr(data.frame(X_train2), label = TRUE, label_round = 2)
g2 <- ggplotGrob(p2)
colors <- g2$grobs[[6]]$children[[3]]$gp$fill
# Change background color to tiles in the upper triangular matrix of plots
idx <- 1
p <- ncol(X_train2)
for (k1 in 1:(p-1)) {
  for (k2 in (k1+1):p) {
    plt <- getPlot(p1,k1,k2) +
      theme(panel.background = element_rect(fill = colors[idx], color="white"),
            panel.grid.major = element_line(color = colors[idx]))
    p1 <- putPlot(p1,plt,k1,k2)
    idx <- idx+1
  }
}
p1


transparentTheme(trans = .9)
featurePlot(x = data.frame(X_train2),
            y = factor(y_train, levels = 0:1, labels = c("No CVD", "CVD")),
            plot = "density",
            scales = list(x = list(relation="free"),
                          y = list(relation="free")),
            adjust = 1.5,
            pch = "|",
            layout = c(4, 3),
            auto.key = list(columns = 2))


2. Support Vector Machine with Radial Kernel
Support vector machine (SVM) on training set with tuning on validation set.
# Create model with default paramters
set.seed(2019)
control <- trainControl(method="cv", index = list(Fold1=seq(length(y_train))), classProbs = TRUE, summaryFunction = twoClassSummary, verboseIter = FALSE)
metric <- "ROC"

m_svm <- train(Class~., data=data.frame(rbind(X_train2, X_valid2), Class = factor(c(y_train, y_valid), levels = c(1,0), labels = c("X1", "X0"))), method="svmRadial", metric=metric, trControl=control)
print(m_svm$bestTune)
##       sigma C
## 3 0.1223938 1
predictions_train <- predict(m_svm, X_train2, type = "prob")
predictions_valid <- predict(m_svm, X_valid2, type = "prob")
predictions <- predict(m_svm, X_test2, type = "prob")

roc_train <- roc(y_train, predictions_train[,1])
roc_valid <- roc(y_valid, predictions_valid[,1])
roc_test <- roc(y_test, predictions[,1])
plot(roc_train, max.auc.polygon=TRUE, col = "firebrick1")
plot(roc_valid, add = TRUE, col = "royalblue1")
plot(roc_test, add = TRUE, col = "forestgreen")
legend("bottomright", col = c("firebrick1", "royalblue1", "forestgreen"),
       legend = c(paste0("train auc = ", formatC(auc(roc_train), digits = 2)),
                  paste0("valid auc = ", formatC(auc(roc_valid), digits = 2)),
                  paste0("test auc = ", formatC(auc(roc_test), digits = 2))),
       lwd = 2, bty = "n")


## Find optimal threshold.

roc_test <- roc(y_test, predictions[,1])
auc_test <- auc(roc_test)

possible_k <- seq(0, 1, length.out = 101)
specificity_valid <- sapply(possible_k, function(k) {
  predicted_class <- as.numeric(predictions_valid[,1] > k)
  sum(predicted_class == 0 & y_valid == 0)/(length(y_valid)-sum(y_valid))
})

sensitivity_valid <- sapply(possible_k, function(k) {
  predicted_class <- as.numeric(predictions_valid[,1] > k)
  sum(predicted_class == 1 & y_valid == 1)/sum(y_valid)
})

BACC_valid <- sapply(possible_k, function(k) {
  predicted_class <- as.numeric(predictions_valid[,1] > k)
  specificity <- sum(predicted_class == 0 & y_valid == 0)/(length(y_valid)-sum(y_valid))
  sensitivity <- sum(predicted_class == 1 & y_valid == 1)/sum(y_valid)
  1/2*(specificity + sensitivity)
})

threshold <- max(possible_k[BACC_valid == max(BACC_valid)])
threshold_ind <- which(possible_k == threshold)

specificity_test <- sapply(threshold, function(k) {
  predicted_class <- as.numeric(predictions[,1] > k)
  sum(predicted_class == 0 & y_test == 0)/(length(y_test)-sum(y_test))
})

sensitivity_test <- sapply(threshold, function(k) {
  predicted_class <- as.numeric(predictions[,1] > k)
  sum(predicted_class == 1 & y_test == 1)/sum(y_test)
})

BACC_test <- sapply(threshold, function(k) {
  predicted_class <- as.numeric(predictions[,1] > k)
  specificity <- sum(predicted_class == 0 & y_test == 0)/(length(y_test)-sum(y_test))
  sensitivity <- sum(predicted_class == 1 & y_test == 1)/sum(y_test)
  1/2*(specificity + sensitivity)
})

results <- data.frame(network = "Original",
                      test_auc = auc_test,
                      threshold = threshold,
                      test_sens = sensitivity_test,
                      test_spec = specificity_test,
                      test_bacc = BACC_test)

results
network
<fctr>
  test_auc
<dbl>
  threshold
<dbl>
  test_sens
<dbl>
  test_spec
<dbl>
  test_bacc
<dbl>
  Original	0.8919753	0.87	0.8148148	0.8333333	0.8240741
1 row
3. AUTOENCODER
An autoencoder neural network (AE) is an unsupervised learning algorithm that applies backpropagation, setting the target values to be equal to the input.

The AE tries to learn a function hW,b(x)≈x. In other words, it is trying to learn an approximation to the identity function, so as to output xˆ that is similar to x.

By placing constraints on the network, such as by limiting the number of hidden units, we can discover interesting structure about the data.

(https://towardsdatascience.com/deep-autoencoders-using-tensorflow-c68f075fd1a3).

len_layer1 <- 3
len_layer2 <- 6

library(keras)
use_session_with_seed(2019)
## Set session seed to 2019 (disabled GPU, CPU parallelism)
input_nume <- layer_input(c(dim(X_train2)[2]))

pred_nume <- input_nume %>%
  layer_dense(len_layer2, activation = "relu") %>%
  layer_dense(len_layer1, activation = "relu") %>%
  layer_dense(len_layer2, activation = "relu") %>%
  layer_dense(dim(X_train2)[2])

model_nume <- keras_model(input_nume, pred_nume)

# summary(model_nume)

model_nume %>% compile(
  optimizer = optimizer_adam(0.01),
  loss = "mean_squared_error"
  # loss = "mean_absolute_error"
)

history_nume <- model_nume %>% fit(X_train2, X_train2,
                                   batch_size = 4096,
                                   validation_data = list(X_valid2, X_valid2),
                                   epochs = 5000,
                                   verbose = 0,
                                   callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 100, restore_best_weights = TRUE),
                                                    callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1, patience = 10)))


print(history_nume)
## Trained on 117 samples, validated on 39 samples (batch_size=4,096, epochs=338)
## Final epoch (plot to see history):
## val_loss: 0.2807
##     loss: 0.2956
##       lr: 0.00000000000000001
# plot(history_nume)

ae_train <- predict(model_nume, X_train2)
colnames(ae_train) <- colnames(X_train2)

dim(ae_train)
## [1] 117  12
dim(unique(ae_train))
## [1] 113  12
ae_valid <- predict(model_nume, X_valid2)
colnames(ae_valid) <- colnames(X_train2)

ae_test <- predict(model_nume, X_test2)
colnames(ae_test) <- colnames(X_train2)
Plot the distributions and scatters of autoencoded continuous variables.
p1 <- ggpairs(data.frame(ae_train),
              lower = list(continuous = wrap("points", alpha = 0.5, color = c("royalblue2", "firebrick2")[y_train+1])),
              diag = list(continuous = wrap("barDiag", bins = 20, fill = "#00AFBB", color = "black")),
              upper = list(continuous = wrap("cor", color = "black", size = 4)))
# Correlation matrix plot
p2 <- ggcorr(data.frame(ae_train), label = TRUE, label_round = 2)
g2 <- ggplotGrob(p2)
colors <- g2$grobs[[6]]$children[[3]]$gp$fill
# Change background color to tiles in the upper triangular matrix of plots
idx <- 1
p <- ncol(ae_train)
for (k1 in 1:(p-1)) {
  for (k2 in (k1+1):p) {
    plt <- getPlot(p1,k1,k2) +
      theme(panel.background = element_rect(fill = colors[idx], color="white"),
            panel.grid.major = element_line(color = colors[idx]))
    p1 <- putPlot(p1,plt,k1,k2)
    idx <- idx+1
  }
}
p1


transparentTheme(trans = .9)
featurePlot(x = data.frame(ae_train),
            y = factor(y_train, levels = 0:1, labels = c("No CVD", "CVD")),
            plot = "density",
            scales = list(x = list(relation="free"),
                          y = list(relation="free")),
            adjust = 1.5,
            pch = "|",
            layout = c(4, 3),
            auto.key = list(columns = 2))


Support vector machine (SVM) on autoencoded training set with tuning on validation set.
# Create model with default paramters
set.seed(2019)
control <- trainControl(method="cv", index = list(Fold1=seq(length(y_train))), classProbs = TRUE, summaryFunction = twoClassSummary, verboseIter = FALSE)
metric <- "ROC"

m_svm <- train(Class~., data=data.frame(rbind(ae_train, ae_valid), Class = factor(c(y_train, y_valid), levels = c(1,0), labels = c("X1", "X0"))), method="svmRadial", metric=metric, trControl=control)
print(m_svm$bestTune)
##       sigma   C
## 2 0.4291626 0.5
predictions_train <- predict(m_svm, ae_train, type = "prob")
predictions_valid <- predict(m_svm, ae_valid, type = "prob")
predictions <- predict(m_svm, ae_test, type = "prob")

roc_train <- roc(y_train, predictions_train[,1])
roc_valid <- roc(y_valid, predictions_valid[,1])
roc_test <- roc(y_test, predictions[,1])
plot(roc_train, max.auc.polygon=TRUE, col = "firebrick1")
plot(roc_valid, add = TRUE, col = "royalblue1")
plot(roc_test, add = TRUE, col = "forestgreen")
legend("bottomright", col = c("firebrick1", "royalblue1", "forestgreen"),
       legend = c(paste0("train auc = ", formatC(auc(roc_train), digits = 2)),
                  paste0("valid auc = ", formatC(auc(roc_valid), digits = 2)),
                  paste0("test auc = ", formatC(auc(roc_test), digits = 2))),
       lwd = 2, bty = "n")


## Find optimal threshold.

roc_test <- roc(y_test, predictions[,1])
auc_test <- auc(roc_test)

possible_k <- seq(0, 1, length.out = 101)
specificity_valid <- sapply(possible_k, function(k) {
  predicted_class <- as.numeric(predictions_valid[,1] > k)
  sum(predicted_class == 0 & y_valid == 0)/(length(y_valid)-sum(y_valid))
})

sensitivity_valid <- sapply(possible_k, function(k) {
  predicted_class <- as.numeric(predictions_valid[,1] > k)
  sum(predicted_class == 1 & y_valid == 1)/sum(y_valid)
})

BACC_valid <- sapply(possible_k, function(k) {
  predicted_class <- as.numeric(predictions_valid[,1] > k)
  specificity <- sum(predicted_class == 0 & y_valid == 0)/(length(y_valid)-sum(y_valid))
  sensitivity <- sum(predicted_class == 1 & y_valid == 1)/sum(y_valid)
  1/2*(specificity + sensitivity)
})

threshold <- max(possible_k[BACC_valid == max(BACC_valid)])
threshold_ind <- which(possible_k == threshold)

specificity_test <- sapply(threshold, function(k) {
  predicted_class <- as.numeric(predictions[,1] > k)
  sum(predicted_class == 0 & y_test == 0)/(length(y_test)-sum(y_test))
})

sensitivity_test <- sapply(threshold, function(k) {
  predicted_class <- as.numeric(predictions[,1] > k)
  sum(predicted_class == 1 & y_test == 1)/sum(y_test)
})

BACC_test <- sapply(threshold, function(k) {
  predicted_class <- as.numeric(predictions[,1] > k)
  specificity <- sum(predicted_class == 0 & y_test == 0)/(length(y_test)-sum(y_test))
  sensitivity <- sum(predicted_class == 1 & y_test == 1)/sum(y_test)
  1/2*(specificity + sensitivity)
})

## Confusion matrix for Test set
print(confusionMatrix(factor(ifelse(predictions[,1]>threshold, 1, 0), levels = c(1,0), labels = c("yes","no")),
                      factor(y_test, levels = c(1,0), labels = c("yes","no")), mode = "everything"))
## Confusion Matrix and Statistics
##
##           Reference
## Prediction yes no
##        yes  25  2
##        no    2 10
##
##                Accuracy : 0.8974
##                  95% CI : (0.7578, 0.9713)
##     No Information Rate : 0.6923
##     P-Value [Acc > NIR] : 0.002469
##
##                   Kappa : 0.7593
##  Mcnemar's Test P-Value : 1.000000
##
##             Sensitivity : 0.9259
##             Specificity : 0.8333
##          Pos Pred Value : 0.9259
##          Neg Pred Value : 0.8333
##               Precision : 0.9259
##                  Recall : 0.9259
##                      F1 : 0.9259
##              Prevalence : 0.6923
##          Detection Rate : 0.6410
##    Detection Prevalence : 0.6923
##       Balanced Accuracy : 0.8796
##
##        'Positive' Class : yes
##
results <- data.frame(network = paste(dim(X_train2)[2], len_layer2, len_layer1, len_layer2, dim(X_train2)[2], sep = "-"),
                      test_auc = auc_test,
                      threshold = threshold,
                      test_sens = sensitivity_test,
                      test_spec = specificity_test,
                      test_bacc = BACC_test)

results
network
<fctr>
  test_auc
<dbl>
  threshold
<dbl>
  test_sens
<dbl>
  test_spec
<dbl>
  test_bacc
<dbl>
  12-6-3-6-12	0.9228395	0.77	0.9259259	0.8333333	0.8796296
1 row
4. What Does Encoded Layer Represent?
  layer_weight <- keras::get_weights(model_nume)

library(keras)
use_session_with_seed(2019)
## Set session seed to 2019 (disabled GPU, CPU parallelism)
input_nume2 <- layer_input(c(dim(X_train2)[2]))

pred_nume2 <- input_nume2 %>%
  layer_dense(len_layer2, activation = "relu") %>%
  layer_dense(len_layer1, activation = "relu")

model_nume2 <- keras_model(input_nume2, pred_nume2)

set_weights(model_nume2, layer_weight)
keras::get_weights(model_nume2)
## [[1]]
##               [,1]       [,2]        [,3]        [,4]        [,5]
##  [1,]  0.286590338 -0.1793639  0.18121468  0.22520919 -0.06973965
##  [2,]  0.139272898  0.1609254  0.28390157  0.67630029 -0.19852532
##  [3,]  0.140913874 -0.1448469 -0.04474318  0.04948473  0.57023549
##  [4,]  0.497060984 -0.5336908 -0.48124576 -0.09977278 -1.16251922
##  [5,]  0.463201404 -0.1421772 -0.19186242 -0.08540363 -0.06539109
##  [6,]  0.657154441 -0.2802740 -0.29280734 -0.49936086  0.11026258
##  [7,] -0.512822330  0.4430973 -0.12332270 -0.30923128 -0.39297375
##  [8,]  0.142603934  0.2329143 -0.43538696  0.13692233 -0.27949435
##  [9,]  0.005646362 -0.2977278 -0.62710965 -0.48403764 -0.49664074
## [10,]  0.381702542 -0.1906457 -0.42791185 -0.04092406 -0.85809124
## [11,] -0.005422603 -0.4266899 -0.28970915 -0.02312082  0.14651223
## [12,]  0.145327419 -0.3109082  0.05152485  0.21832523 -0.26457927
##               [,6]
##  [1,]  0.627443373
##  [2,] -0.009237301
##  [3,]  0.459880590
##  [4,] -0.298615485
##  [5,] -0.853187144
##  [6,] -0.681565166
##  [7,]  0.594684362
##  [8,] -0.139212295
##  [9,]  0.253491580
## [10,] -0.565868199
## [11,] -0.694983184
## [12,] -0.435368627
##
## [[2]]
## [1] -0.4442440 -0.1566193  0.5486414  1.3323117  0.6391762  0.6826347
##
## [[3]]
##             [,1]        [,2]        [,3]
## [1,]  0.17102952  0.82263052 -0.06747206
## [2,] -0.16307418 -0.29460073 -0.59077674
## [3,]  0.35880554 -0.22678854  0.17819607
## [4,]  0.83184981  0.03800514 -0.01581647
## [5,] -0.04969420 -0.91424811  0.31136891
## [6,]  0.02804342 -0.35988703  0.54151946
##
## [[4]]
## [1] -0.03926072  0.50863582  0.36285111
comp_train <- predict(model_nume2, X_train2)
colnames(comp_train) <- c("X1", "X2", "X3")

cbind(comp_train, y_train)
##                X1         X2         X3 y_train
##   [1,] 0.00000000 0.50863582 0.36285111       1
##   [2,] 0.11953820 1.27243876 0.30020410       1
##   [3,] 0.16271730 1.48012471 0.28316969       1
##   [4,] 0.00000000 0.50863582 0.36285111       1
##   [5,] 0.00000000 0.00000000 1.16646266       1
##   [6,] 0.00000000 0.00000000 1.04138327       1
##   [7,] 0.04024490 0.51226825 0.36133942       1
##   [8,] 0.01662616 0.51118916 0.36178851       1
##   [9,] 1.62873578 0.00000000 0.95188141       1
##  [10,] 1.65550268 0.00000000 1.59317231       1
##  [11,] 1.71126699 0.00000000 0.64879209       1
##  [12,] 2.76757669 0.00000000 0.22414568       1
##  [13,] 2.64265299 1.67285705 0.22975682       1
##  [14,] 1.72025681 2.18875289 0.20331287       1
##  [15,] 1.97361600 1.69305038 0.28960407       1
##  [16,] 1.50126493 1.69782877 0.24538058       1
##  [17,] 1.42030489 0.60642755 0.39428324       1
##  [18,] 1.77771461 0.84694338 0.31931227       1
##  [19,] 1.83197522 0.00000000 0.66029954       1
##  [20,] 1.34342492 0.00000000 1.09233415       1
##  [21,] 1.26403534 0.00000000 1.80661178       1
##  [22,] 1.88017547 0.00000000 1.82381904       1
##  [23,] 1.04723418 0.00000000 1.96857738       1
##  [24,] 1.37651873 0.00000000 4.54159451       0
##  [25,] 0.95734507 0.00000000 5.01077557       0
##  [26,] 1.64061284 0.00000000 4.75961018       0
##  [27,] 1.22598326 0.00000000 3.01110744       1
##  [28,] 1.10365283 0.00000000 3.08450413       1
##  [29,] 1.27467465 0.00000000 3.09984350       1
##  [30,] 3.55656886 0.00000000 4.72866440       0
##  [31,] 3.60416985 0.00000000 5.05160141       0
##  [32,] 3.62445426 0.00000000 4.96966743       0
##  [33,] 3.77175450 0.00000000 4.88946915       0
##  [34,] 3.21175289 0.00000000 4.03523159       0
##  [35,] 0.12466680 0.00000000 1.61490273       0
##  [36,] 0.00000000 0.00000000 1.74960291       0
##  [37,] 0.20477815 0.00000000 2.35825181       0
##  [38,] 0.00000000 0.50863582 0.36285111       1
##  [39,] 0.00000000 0.35900041 0.58800644       1
##  [40,] 0.00000000 0.31489980 0.65436429       1
##  [41,] 0.35891613 0.37217739 0.58738410       1
##  [42,] 2.83261442 0.00000000 3.56147337       0
##  [43,] 2.72590947 0.00000000 3.37607098       0
##  [44,] 3.19229102 0.00000000 5.61731720       0
##  [45,] 2.98019433 0.00000000 4.37336731       0
##  [46,] 0.89484537 0.46218440 0.41290140       1
##  [47,] 0.64137393 0.53973234 0.34990978       1
##  [48,] 0.76555043 0.00000000 0.84446716       1
##  [49,] 0.97333550 1.10836363 0.29997641       1
##  [50,] 0.10991740 0.00000000 1.26579750       1
##  [51,] 0.00000000 0.03122595 1.05322683       1
##  [52,] 0.00000000 0.50863582 0.36285111       1
##  [53,] 0.00000000 0.68908381 0.34805080       1
##  [54,] 0.20064309 0.63270098 0.34937528       1
##  [55,] 0.00000000 0.50863582 0.36285111       1
##  [56,] 0.05515336 0.34528169 0.61269689       1
##  [57,] 0.00000000 0.00000000 0.95845056       1
##  [58,] 1.88363016 1.72585392 0.28348234       1
##  [59,] 2.08099627 0.00000000 0.88747013       1
##  [60,] 2.04578662 3.16686654 0.12120508       1
##  [61,] 1.16943169 0.00000000 1.18304312       1
##  [62,] 1.26943195 0.00000000 1.26647067       1
##  [63,] 1.02589214 2.59967256 0.18162809       1
##  [64,] 0.78351641 2.60317087 0.18508804       1
##  [65,] 0.97633970 4.97569466 0.00000000       1
##  [66,] 1.61455727 8.46329212 0.00000000       1
##  [67,] 1.31142402 3.57282472 0.10052782       1
##  [68,] 3.41372967 7.69344807 0.00000000       1
##  [69,] 1.41462886 0.00000000 3.72267199       1
##  [70,] 0.90863335 0.00000000 3.13981724       1
##  [71,] 1.42672455 0.00000000 3.45182180       1
##  [72,] 1.59203231 0.00000000 0.97518206       1
##  [73,] 1.84966731 0.00000000 1.40873182       1
##  [74,] 2.45510006 0.00000000 3.94332719       1
##  [75,] 1.86211038 0.00000000 2.12126899       1
##  [76,] 2.74732828 0.00000000 2.73952341       1
##  [77,] 3.30561328 0.46531337 0.58999968       1
##  [78,] 3.75214601 0.25425994 0.61608779       1
##  [79,] 3.26908946 0.03125992 0.37019175       1
##  [80,] 2.08378887 0.00000000 1.08601606       1
##  [81,] 0.97709954 0.38975421 0.43906677       1
##  [82,] 1.16152847 0.00000000 1.04739559       1
##  [83,] 0.93973178 0.00000000 0.68982166       1
##  [84,] 0.44836280 0.00000000 1.60223019       1
##  [85,] 0.72344881 0.00000000 1.35763764       1
##  [86,] 0.73522937 0.00000000 0.96228302       1
##  [87,] 0.44178468 0.00000000 1.92888832       1
##  [88,] 2.46208525 1.15748417 0.27315924       1
##  [89,] 2.54350519 0.00000000 1.38524032       1
##  [90,] 3.36336899 0.00000000 2.71317005       1
##  [91,] 1.72347414 3.41088080 0.10694066       1
##  [92,] 2.03304911 4.46875763 0.01879218       1
##  [93,] 3.52636600 3.90268087 0.05008668       1
##  [94,] 1.70263934 8.88695621 0.00000000       1
##  [95,] 0.94705540 0.47664177 0.18473153       1
##  [96,] 1.57825398 0.88381600 0.30835077       1
##  [97,] 0.90288985 0.00000000 0.48015094       1
##  [98,] 1.14858508 1.59069777 0.25925994       1
##  [99,] 3.32388663 0.00000000 3.03821707       0
## [100,] 3.26800609 0.00000000 5.20020580       0
## [101,] 3.68375421 0.00000000 5.13670826       0
## [102,] 0.15397212 0.00000000 1.84131849       0
## [103,] 0.03411100 0.00000000 1.87619257       0
## [104,] 0.79919875 0.00000000 2.72825241       1
## [105,] 0.82101166 0.00000000 1.67185974       1
## [106,] 0.90762037 0.00000000 2.69659662       1
## [107,] 0.92178881 0.00000000 1.85969925       1
## [108,] 1.05338454 0.00000000 3.10782957       1
## [109,] 0.80174255 0.00000000 0.67150962       0
## [110,] 1.53116405 0.00000000 0.00000000       0
## [111,] 1.34988117 0.00000000 0.68217409       0
## [112,] 4.01847601 0.00000000 0.46006307       0
## [113,] 4.11749840 0.00000000 0.97028399       0
## [114,] 0.94940650 0.00000000 1.46744370       0
## [115,] 2.56767344 0.00000000 2.05802441       0
## [116,] 3.39762759 0.00000000 1.72753704       0
## [117,] 2.75376153 0.00000000 1.70106411       0
get_colors <- function(groups, group.col = palette()){
  groups <- as.factor(groups)
  ngrps <- length(levels(groups))
  if(ngrps > length(group.col))
    group.col <- rep(group.col, ngrps)
  color <- group.col[as.numeric(groups)]
  names(color) <- as.vector(groups)
  return(color)
}

cols <- get_colors(y_train, c("firebrick1", "royalblue1"))
rgl.bg(color = "white") # Setup the background color
rgl.spheres(comp_train[,1], comp_train[,2], comp_train[,3], r = 0.1, color = cols)
rgl.bbox(color=c("black","grey"), emission="black",
         specular="grey", shininess=5, alpha=0.8)
rgl.viewpoint(-30, 15, 60, 1)
title3d(xlab = "X1", ylab = "X2", zlab = "X3")
rglwidget()
