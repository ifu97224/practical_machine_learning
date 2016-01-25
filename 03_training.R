# Practical Machine Learning Assignment
# 03_training

# This code creates various models on the training data and predicts on the validation data to get an out of sample
# accuracy

library(dplyr)
library(Hmisc)
library(ClustOfVar)
library(ggplot2)
library(gridExtra)
library(GGally)
library(caret)
library(pROC)
library(scales)

# set the working directory
setwd("/users/Richard/Documents/Coursera Data Science Track/Practical Machine Learning/Assignment")
getwd()

# load the data
load("training.Rda")
load("validation.Rda")
load("testing.Rda")
load("var_selection_A.Rda")
load("var_selection_B.Rda")
load("var_selection_C.Rda")
load("var_selection_D.Rda")
load("var_selection_E.Rda")

str(training)
summary(training)

training$classe <- as.character(training$classe)
training <- subset(training,classe %in% c("A","B","C","D","E"))
training$classe <- as.factor(training$classe)
table(training$classe)

validation$classe <- as.character(validation$classe)
validation <- subset(validation,classe %in% c("A","B","C","D","E"))
validation$classe <- as.factor(validation$classe)
table(validation$classe)

#########################################################################################################
#######                                 RANDOM FOREST                                     ###############
#########################################################################################################

# set the seed
set.seed(123)

# 10 fold cross validation
ctrl <- trainControl(method = "cv",
                     number = 10,
                     classProbs = TRUE)

mtry_grid <- expand.grid(mtry = seq(1,10, by = 1))

rf_mtry_tune <-  train(classe ~ .,
                       data = training,
                       method = "rf",
                       metric = "ROC",
                       trControl = ctrl,
                       tuneGrid = mtry_grid,
                       ntree = 100,
                       preProc = c("center", "scale"),
                       importance = TRUE,
                       allowParallel = TRUE)

rf_mtry_tune$bestTune$mtry
plot(rf_mtry_tune)
plot(varImp(rf_mtry_tune))

results <- rf_mtry_tune$results


# now tune the number of trees

num_tree <- seq(1,101, by = 25)
mtry_grid2 <- as.data.frame(rf_mtry_tune$bestTune$mtry)
names(mtry_grid2)[names(mtry_grid2)=="rf_mtry_tune$bestTune$mtry"] <- "mtry"

cv_accuracy <- data.frame(num_trees = as.numeric(),
                          best_acc = as.numeric())

for(n in 1:length(num_tree)){
  
  tune_tree <- num_tree[n]
  
  rf_trees_tune <-   train(classe ~ .,
                           data = training,
                           method = "rf",
                           trControl = ctrl,
                           tuneGrid = mtry_grid2,
                           ntree = tune_tree,
                           preProc = c("center", "scale"),
                           verbose=FALSE,
                           importance = TRUE,
                           allowParallel = TRUE)
  
  
  cv_accuracy[n,1] <- num_tree[n]
  cv_accuracy[n,2] <- rf_trees_tune$results$Accuracy
  
}
cv_accuracy$num_trees <- as.numeric(cv_accuracy$num_trees)

g <- ggplot(cv_accuracy,aes(as.factor(num_trees), best_acc,group = 1)) +
     geom_point(color = "red") +
     geom_line(color = "red") +
     theme_bw(base_family = "Arial") + 
     ggtitle("CV Accuracy by Numer of Trees") +
     xlab("Number of Trees") + 
     ylab("Cross Validated Accuracy") +
     theme(axis.title.y=element_text(vjust=1,size=14),
           axis.title.x=element_text(vjust=-0.5,size=14),  
           plot.title=element_text(vjust=1,size=16),
           axis.text.x = element_text(size = 12),
           axis.text.y = element_text(size = 12))
g


# finally get the cross validated results using the best tuning settings

best_rf <-  train(classe ~ .,
                  data = training,
                  method = "rf",
                  metric = "ROC",
                  trControl = ctrl,
                  tuneGrid = mtry_grid2,
                  ntree = 50,
                  preProc = c("center", "scale"),
                  importance = TRUE,
                  allowParallel = TRUE)

best_rf$results
# Note:  In sample cross validated accuracy 0.9909133

# predict on the validation set and get the validation accuracy
validation.pred.classes <- predict(best_rf, newdata = validation)

# validation set accuracy
confusionMatrix(data = validation.pred.classes, validation$classe)
# Note:  RF accuracy on the validation set was 0.9917

# save the model
save(best_rf, file = "best_rf.rda")

rm(list = c("cv_accuracy","cv_auc","mtry_grid","mtry_grid2","results","validation.pred.classes","tune_tree",
            "rf_mtry_tune","rf_trees_tune","num_tree"))


#########################################################################################################
#######                             GRADIENT BOOSTING MODEL                               ###############
#########################################################################################################

# 10 fold cross validation
ctrl <- trainControl(method = "cv",
                     number = 10,
                     classProbs = TRUE)

depth_grid <- 10
shrinkage_grid <- 0.1
tree_grid <- seq(200,400, by = 100)
minobs_grid <- n.minobsinnode <- 100

gbm.train <-  train(classe ~ .,
                    data = training,
                    method = "gbm",
                    metric = "ROC",
                    trControl = ctrl,
                    tuneGrid = expand.grid(interaction.depth = depth_grid,
                                           n.trees = tree_grid,
                                           shrinkage = shrinkage_grid,
                                           n.minobsinnode = minobs_grid),
                          preProc = c("center", "scale"))

gbm.train$bestTune$interaction.depth
gbm.train$bestTune$n.trees
gbm.train$bestTune$shrinkage

plot(gbm.train)
plot(varImp(gbm.train))
gbm.train$results

gbm.train$results
# Note:  In sample cross validated accuracy 0.9952

# predict on the validation set and get the validation accuracy
validation.pred.classes <- predict(gbm.train, newdata = validation)
str(validation.pred.classes)

confusionMatrix(data = validation.pred.classes, validation$classe)
# Note:  GBM accuracy on the validation set was 0.9944

# save the model
save(gbm.train, file = "best_gbm.rda")

rm(list = c("ctrl","depth_grid","minobs_grid","n.minobsinnode","shrinkage_grid","tree_grid","validation.pred.classes"))

#########################################################################################################
#######                                     LDA MODEL                                     ###############
#########################################################################################################

drops <- names(training) %in% c("magnet_belt_z_sqrt","magnet_belt_y_sqrt")
training2 <- training[!drops]

# 10 fold cross validation
ctrl <- trainControl(method = "cv",
                     number = 10,
                     classProbs = TRUE)

lda.train <-  train(classe ~ .,
                    data = training2,
                    method = "lda",
                    metric = "ROC",
                    trControl = ctrl,
                    preProc = c("center", "scale"))

lda.train$results
# Note:  In sample cross validated accuracy 0.9042

validation.pred.classes <- predict(lda.train, newdata = validation)
str(validation.pred.classes)

confusionMatrix(data = validation.pred.classes, validation$classe)
# Note:  LDA accuracy on the validation set was 0.8964

#########################################################################################################
#######                                     STACKED MODEL                                       #########
#########################################################################################################

# stack the Random Forest, GBM and LDA models

# get the predictions from all 3 models on the training data
train_pred_gbm <- predict(gbm.train, newdata = training, type = "prob")
train_pred_rf  <- predict(best_rf, newdata = training, type = "prob")
train_pred_lda <- predict(lda.train, newdata = training, type = "prob")
pred_DF <- data.frame(train_pred_gbm,train_pred_rf,train_pred_lda,training$classe)

names(pred_DF)[names(pred_DF)=="A"] <- "gbm_A"
names(pred_DF)[names(pred_DF)=="B"] <- "gbm_B"
names(pred_DF)[names(pred_DF)=="C"] <- "gbm_C"
names(pred_DF)[names(pred_DF)=="D"] <- "gbm_D"
names(pred_DF)[names(pred_DF)=="E"] <- "gbm_E"
names(pred_DF)[names(pred_DF)=="A.1"] <- "rf_A"
names(pred_DF)[names(pred_DF)=="B.1"] <- "rf_B"
names(pred_DF)[names(pred_DF)=="C.1"] <- "rf_C"
names(pred_DF)[names(pred_DF)=="D.1"] <- "rf_D"
names(pred_DF)[names(pred_DF)=="E.1"] <- "rf_E"
names(pred_DF)[names(pred_DF)=="A.2"] <- "lda_A"
names(pred_DF)[names(pred_DF)=="B.2"] <- "lda_B"
names(pred_DF)[names(pred_DF)=="C.2"] <- "lda_C"
names(pred_DF)[names(pred_DF)=="D.2"] <- "lda_D"
names(pred_DF)[names(pred_DF)=="E.2"] <- "lda_E"
names(pred_DF)[names(pred_DF)=="training.classe"] <- "classe"

# now fit a random forest model

# 10 fold cross validation
ctrl <- trainControl(method = "cv",
                     number = 10,
                     classProbs = TRUE)

rf_stacked <-  train(classe ~ .,
                      data = pred_DF,
                      method = "rf",
                      metric = "ROC",
                      trControl = ctrl,
                      preProc = c("center", "scale"),
                      importance = TRUE,
                      allowParallel = TRUE)

plot(rf_stacked)
plot(varImp(rf_stacked))
rf_stacked$results
# Note:  Cross validated accuracy is 

# get the predicitions for each of the 3 models on the validation set
valid_pred_gbm <- predict(gbm.train, newdata = validation, type = "prob")
valid_pred_rf  <- predict(best_rf, newdata = validation, type = "prob")
valid_pred_lda <- predict(lda.train, newdata = validation, type = "prob")

valid_DF <- data.frame(valid_pred_gbm,valid_pred_rf,valid_pred_lda)

names(valid_DF)[names(valid_DF)=="A"] <- "gbm_A"
names(valid_DF)[names(valid_DF)=="B"] <- "gbm_B"
names(valid_DF)[names(valid_DF)=="C"] <- "gbm_C"
names(valid_DF)[names(valid_DF)=="D"] <- "gbm_D"
names(valid_DF)[names(valid_DF)=="E"] <- "gbm_E"
names(valid_DF)[names(valid_DF)=="A.1"] <- "rf_A"
names(valid_DF)[names(valid_DF)=="B.1"] <- "rf_B"
names(valid_DF)[names(valid_DF)=="C.1"] <- "rf_C"
names(valid_DF)[names(valid_DF)=="D.1"] <- "rf_D"
names(valid_DF)[names(valid_DF)=="E.1"] <- "rf_E"
names(valid_DF)[names(valid_DF)=="A.2"] <- "lda_A"
names(valid_DF)[names(valid_DF)=="B.2"] <- "lda_B"
names(valid_DF)[names(valid_DF)=="C.2"] <- "lda_C"
names(valid_DF)[names(valid_DF)=="D.2"] <- "lda_D"
names(valid_DF)[names(valid_DF)=="E.2"] <- "lda_E"

validation.pred.classes <- predict(rf_stacked, newdata = valid_DF)
str(validation.pred.classes)

# get the accuracy of the stacked model
confusionMatrix(data = validation.pred.classes, validation$classe)
# stacked model accuracy 0.9941

# plot the validation accuracy in a bar plot
model <- c("RF","GBM","LDA","Stacked")
valid_accuracy = c(0.9917, 0.9944, 0.8964, 0.9941) 
valid_acc_plot <- data.frame(model,valid_accuracy)

a <- ggplot(valid_acc_plot, aes(factor(model),valid_accuracy)) +
     geom_bar(stat = "identity", fill = "red") +
     theme_bw(base_family = "Arial") + 
     ggtitle("Validation Set Accuracy by Model Type") +
     xlab("Model") + 
     ylab("Validation Set Accuracy") +
     theme(axis.title.y=element_text(vjust=1,size=14),
           axis.title.x=element_text(vjust=-0.5,size=14),
           plot.title=element_text(vjust=1,size=16),
           axis.text.x = element_text(size = 12),
           axis.text.y = element_text(size = 12)) +
           scale_y_continuous(labels=percent) +
     geom_text(label=sprintf("%1.2f%%", 100*valid_accuracy),vjust = 1.5)
a
