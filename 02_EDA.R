# Practical Machine Learning Assignment
# 02_Read_Data_and_EDA

# This code runs some basic EDA to understand potential top predictors and creates some simple plots

library(dplyr)
library(Hmisc)
library(ClustOfVar)
library(ggplot2)
library(gridExtra)
library(GGally)
library(caret)

# set the working directory
setwd("/users/Richard/Documents/Coursera Data Science Track/Practical Machine Learning/Assignment")
getwd()

# load the data
load("training.Rda")
load("testing.Rda")

# NOTE:  All EDA and variable selection will be carried out only on the training set - split the current training
#        set further into training and validation

set.seed(1000)
inTrain <- createDataPartition(y = training$classe, 
                               p = 0.6, 
                               list = FALSE)
training2 <- training[ inTrain,]
training <- training2
rm(training2)
validation = training[-inTrain,]

save(training,file="training.Rda")
save(validation,file="validation.Rda")

#########################################################################################################
#######                                       STEP 1                                      ###############
#######               Identify the univariate top predictors for each classe              ###############
#########################################################################################################

# first create dummy variables for the classe being evaluated
training_univariate <- training
training_univariate$classe <- as.character(training_univariate$classe)

# classe A
classe_A <- ifelse(training_univariate$classe == "A",1,0)
training_univariate$classe_A <- classe_A
table(training_univariate$classe_A)

# classe B
classe_B <- ifelse(training_univariate$classe == "B",1,0)
training_univariate$classe_B <- classe_B
table(training_univariate$classe_B)

# classe C
classe_C <- ifelse(training_univariate$classe == "C",1,0)
training_univariate$classe_C <- classe_C
table(training_univariate$classe_C)

# classe D
classe_D <- ifelse(training_univariate$classe == "D",1,0)
training_univariate$classe_D <- classe_D
table(training_univariate$classe_D)

# classe E
classe_E <- ifelse(training_univariate$classe == "E",1,0)
training_univariate$classe_E <- classe_E
table(training_univariate$classe_E)

rm(classe_A,classe_B,classe_C,classe_D,classe_E)

##### squared correlcations #####

sq_corr <- function(classe){
  
  all_vars <- training_univariate
  
  all_corr_values <<- data.frame(correlation_value = as.numeric(),
                                 variable=as.character(),
                                 squared_correlation = as.numeric(),
                                 correlation_p_value = as.numeric())

  dropvars <- names(all_vars) %in% c("classe", "classe_A", "classe_B", "classe_C", "classe_D", "classe_E")
  vars_to_run_corr <- all_vars[!dropvars]
  
  
  target_var <- eval(parse(text=paste("all_vars$classe_", classe,sep="")))

  lapply(names(vars_to_run_corr),
         
         function(var) {
           
           all_corr_values <<- all_corr_values
           
           var_name <- var
           var <- eval(parse(text=paste("all_vars$", var,sep="")))
           var <- as.matrix(cbind(var,target_var))
           corr_output <- rcorr(var,type="pearson")
           
           # get the correlation value
           corr_value <- as.data.frame(corr_output[1])
           corr_value <- as.data.frame(corr_value[1,2])
           corr_value$variable <- var_name
           names(corr_value)[names(corr_value)=="corr_value[1, 2]"] <- "correlation_value"
           
           # calculate the squared correlation
           corr_value$sqaured_correlation <- corr_value$correlation_value^2
           
           # get the p-value
           corr_p_value <- as.data.frame(corr_output[3])
           corr_p_value <- as.data.frame(corr_p_value[1,2])
           names(corr_p_value)[names(corr_p_value)=="corr_p_value[1, 2]"] <- "correlation_p_value"
           
           corr_value_and_p <- cbind(corr_value,corr_p_value)
           
           all_corr_values <- rbind(all_corr_values,corr_value_and_p)
           
           assign("all_corr_values", all_corr_values, envir = .GlobalEnv) 
           return(all_corr_values)
           
         }
  )
  
  rm(list = c("all_vars","vars_to_run_corr"))

}

sq_corr("A")
all_corr_values <- all_corr_values[order(-all_corr_values$sqaured_correlation),]         
all_corr_values$corr_rank <- 1:nrow(all_corr_values)
all_corr_values_A <- all_corr_values

sq_corr("B")
all_corr_values <- all_corr_values[order(-all_corr_values$sqaured_correlation),]         
all_corr_values$corr_rank <- 1:nrow(all_corr_values)
all_corr_values_B <- all_corr_values

sq_corr("C")
all_corr_values <- all_corr_values[order(-all_corr_values$sqaured_correlation),]         
all_corr_values$corr_rank <- 1:nrow(all_corr_values)
all_corr_values_C <- all_corr_values

sq_corr("D")
all_corr_values <- all_corr_values[order(-all_corr_values$sqaured_correlation),]         
all_corr_values$corr_rank <- 1:nrow(all_corr_values)
all_corr_values_D <- all_corr_values

sq_corr("E")
all_corr_values <- all_corr_values[order(-all_corr_values$sqaured_correlation),]         
all_corr_values$corr_rank <- 1:nrow(all_corr_values)
all_corr_values_E <- all_corr_values

rm(list = c("all_corr_values","sq_corr"))



##### T-Tests #####

t_tests <- function(classe){

  
  # get all negative and positive targets into separate dataframes
  neg <- training_univariate[training_univariate[[paste("classe_", classe,sep="")]]==0, ]
  pos <- training_univariate[training_univariate[[paste("classe_", classe,sep="")]]==1, ]
  
  # remove the target variable
  
  neg <- subset(neg, select=-c(classe,classe_A,classe_B,classe_C,classe_D,classe_E))
  pos <- subset(pos, select=-c(classe,classe_A,classe_B,classe_C,classe_D,classe_E))

  t_test_p_values_all <<- data.frame(t_test.p.value = as.numeric(),
                                    t_stat_abs = as.numeric(),
                                    variable=as.character())
  
  drops <- names(pos) %in% c("magnet_belt_z_sqrt")
  pos <- pos[!drops]
 
  
  lapply(names(pos),
         
         function(var) {
           
           t_test_p_values_all <<- t_test_p_values_all
           
           pos_var <- eval(parse(text=paste("pos$", var,sep="")))
           neg_var <- eval(parse(text=paste("neg$", var,sep="")))
           var_name <- var
           
           # test for equal variance 
           var_test <- var.test(neg_var,pos_var)
           
           if (var_test$p.value <= 0.05){
             t_test <- t.test(neg_var,pos_var,var.equal=FALSE)
           }
           else {
             t_test <- t.test(neg_var,pos_var,var.equal=TRUE)
           }
           
           t_test_out <- data.frame(t_test$p.value)
           t_test_out$t_stat_abs <- abs(t_test$statistic)
           t_test_out$variable <- var_name
           
           t_test_p_values_all <- rbind(t_test_p_values_all,t_test_out)
           assign("t_test_p_values_all", t_test_p_values_all, envir = .GlobalEnv) 
           return(t_test_p_values_all)
           
         }
  )
  
  rm(list = c("neg","pos"))
  
}

#t_test_p_values_all <- t_test_p_values_all[order(t_test_p_values_all$variable),]

t_tests("A")
t_test_p_values_all <- t_test_p_values_all[order(-t_test_p_values_all$t_stat),] 
t_test_p_values_all$t_test_rank <- 1:nrow(t_test_p_values_all)
t_test_p_values_all_A <- t_test_p_values_all

t_tests("B")
t_test_p_values_all <- t_test_p_values_all[order(-t_test_p_values_all$t_stat),] 
t_test_p_values_all$t_test_rank <- 1:nrow(t_test_p_values_all)
t_test_p_values_all_B <- t_test_p_values_all

t_tests("C")
t_test_p_values_all <- t_test_p_values_all[order(-t_test_p_values_all$t_stat),] 
t_test_p_values_all$t_test_rank <- 1:nrow(t_test_p_values_all)
t_test_p_values_all_C <- t_test_p_values_all

t_tests("D")
t_test_p_values_all <- t_test_p_values_all[order(-t_test_p_values_all$t_stat),] 
t_test_p_values_all$t_test_rank <- 1:nrow(t_test_p_values_all)
t_test_p_values_all_D <- t_test_p_values_all

t_tests("E")
t_test_p_values_all <- t_test_p_values_all[order(-t_test_p_values_all$t_stat),] 
t_test_p_values_all$t_test_rank <- 1:nrow(t_test_p_values_all)
t_test_p_values_all_E <- t_test_p_values_all

rm(t_test_p_values_all)



##### Univariate Logistic Regression #####

logistic_prof <- function(classe){

  univariate_logistic_aic_all <<- data.frame(logistic_AIC=as.numeric(),
                                            variable=as.character())
  
  univariate_vars <- subset(training_univariate, select=-c(classe,classe_A,classe_B,classe_C,classe_D,classe_E))
  
  lapply(names(univariate_vars),
         
         function(var) {
           
           univariate_logistic_aic_all <<- univariate_logistic_aic_all
           
           formula    <- as.formula(paste("classe_",classe,"~", var,sep=""))
           
           res.logist <- summary(glm(formula, data = training_univariate, family = binomial))
           univariate_logistic_aic <- data.frame(res.logist$aic)
           names(univariate_logistic_aic)[names(univariate_logistic_aic)=="res.logist.aic"] <- "logistic_AIC"
           univariate_logistic_aic$variable <- var
           
           univariate_logistic_aic_all <- rbind(univariate_logistic_aic_all,univariate_logistic_aic)
           
           assign("univariate_logistic_aic_all", univariate_logistic_aic_all, envir = .GlobalEnv) 
           return(univariate_logistic_aic_all)
           
         })
    
  rm(univariate_vars)

}

logistic_prof("A")
univariate_logistic_aic_all <- univariate_logistic_aic_all[order(univariate_logistic_aic_all$logistic_AIC),] 
univariate_logistic_aic_all$univariate_logistic_rank <- 1:nrow(univariate_logistic_aic_all)
univariate_logistic_aic_A <- univariate_logistic_aic_all

logistic_prof("B")
univariate_logistic_aic_all <- univariate_logistic_aic_all[order(univariate_logistic_aic_all$logistic_AIC),] 
univariate_logistic_aic_all$univariate_logistic_rank <- 1:nrow(univariate_logistic_aic_all)
univariate_logistic_aic_B <- univariate_logistic_aic_all

logistic_prof("C")
univariate_logistic_aic_all <- univariate_logistic_aic_all[order(univariate_logistic_aic_all$logistic_AIC),] 
univariate_logistic_aic_all$univariate_logistic_rank <- 1:nrow(univariate_logistic_aic_all)
univariate_logistic_aic_C <- univariate_logistic_aic_all

logistic_prof("D")
univariate_logistic_aic_all <- univariate_logistic_aic_all[order(univariate_logistic_aic_all$logistic_AIC),] 
univariate_logistic_aic_all$univariate_logistic_rank <- 1:nrow(univariate_logistic_aic_all)
univariate_logistic_aic_D <- univariate_logistic_aic_all

logistic_prof("E")
univariate_logistic_aic_all <- univariate_logistic_aic_all[order(univariate_logistic_aic_all$logistic_AIC),] 
univariate_logistic_aic_all$univariate_logistic_rank <- 1:nrow(univariate_logistic_aic_all)
univariate_logistic_aic_E <- univariate_logistic_aic_all

rm(univariate_logistic_aic_all)
  
  

##### Chi-Square Tests #####

chi_sq <- function(classe){

  chi_sq_p_values_all <<- data.frame(chi_square_p_value = as.numeric(),
                                    chi_stat = as.numeric(),
                                    variable=as.character())
  
  vars_to_test <- subset(training_univariate, select=-c(classe,classe_A,classe_B,classe_C,classe_D,classe_E))
  quants <- training_univariate
  
  drops <- names(vars_to_test) %in% c("magnet_belt_z_sqrt")
  vars_to_test <- vars_to_test[!drops]
  
  
  lapply(names(vars_to_test),
         
         function(var) {
           
           chi_sq_p_values_all <<- chi_sq_p_values_all
           
           test_var <- eval(parse(text=paste("quants$", var,sep="")))
           quants$group <- cut(test_var,10)
           chi_test <- chisq.test(quants[,"group"],quants[,paste("classe_",classe,sep="")])
           
           all_chi_sq_test <- as.data.frame(chi_test$p.value)
           names(all_chi_sq_test)[names(all_chi_sq_test)=="chi_test$p.value"] <- "chi_square_p_value"
           all_chi_sq_test$chi_stat <- chi_test$statistic
           all_chi_sq_test$variable <- var
           
           chi_sq_p_values_all <- rbind(chi_sq_p_values_all,all_chi_sq_test)
           assign("chi_sq_p_values_all", chi_sq_p_values_all, envir = .GlobalEnv) 
           return(chi_sq_p_values_all)
           
         })
  
  rm(list = c("quants","vars_to_test"))

}
chi_sq("A")
chi_sq_p_values_all <- chi_sq_p_values_all[order(-chi_sq_p_values_all$chi_stat),] 
chi_sq_p_values_all$chi_rank <- 1:nrow(chi_sq_p_values_all)
chi_sq_p_values_A <- chi_sq_p_values_all

chi_sq("B")
chi_sq_p_values_all <- chi_sq_p_values_all[order(-chi_sq_p_values_all$chi_stat),] 
chi_sq_p_values_all$chi_rank <- 1:nrow(chi_sq_p_values_all)
chi_sq_p_values_B <- chi_sq_p_values_all

chi_sq("C")
chi_sq_p_values_all <- chi_sq_p_values_all[order(-chi_sq_p_values_all$chi_stat),] 
chi_sq_p_values_all$chi_rank <- 1:nrow(chi_sq_p_values_all)
chi_sq_p_values_C <- chi_sq_p_values_all

chi_sq("D")
chi_sq_p_values_all <- chi_sq_p_values_all[order(-chi_sq_p_values_all$chi_stat),] 
chi_sq_p_values_all$chi_rank <- 1:nrow(chi_sq_p_values_all)
chi_sq_p_values_D <- chi_sq_p_values_all

chi_sq("E")
chi_sq_p_values_all <- chi_sq_p_values_all[order(-chi_sq_p_values_all$chi_stat),] 
chi_sq_p_values_all$chi_rank <- 1:nrow(chi_sq_p_values_all)
chi_sq_p_values_E <- chi_sq_p_values_all

rm(chi_sq_p_values_all)

##### Merge the datasets together #####
 
# Classe A
var_selection_A <- merge(all_corr_values_A,chi_sq_p_values_A,by = "variable",all=T)
var_selection_A <- merge(var_selection_A,t_test_p_values_all_A,by = "variable",all=T)
var_selection_A <- merge(var_selection_A,univariate_logistic_aic_A,by = "variable",all=T)

var_selection_A$mean_rank <- rowMeans(subset(var_selection_A, 
                                             select = c(corr_rank, chi_rank,t_test_rank,univariate_logistic_rank)), 
                                             na.rm = TRUE)

var_selection_A <- var_selection_A[order(var_selection_A$mean_rank),]

rm(list = c("all_corr_values_A","chi_sq_p_values_A","t_test_p_values_all_A","univariate_logistic_aic_A"))


# Classe B
var_selection_B <- merge(all_corr_values_B,chi_sq_p_values_B,by = "variable",all=T)
var_selection_B <- merge(var_selection_B,t_test_p_values_all_B,by = "variable",all=T)
var_selection_B <- merge(var_selection_B,univariate_logistic_aic_B,by = "variable",all=T)

var_selection_B$mean_rank <- rowMeans(subset(var_selection_B, 
                                             select = c(corr_rank, chi_rank,t_test_rank,univariate_logistic_rank)), 
                                      na.rm = TRUE)

var_selection_B <- var_selection_B[order(var_selection_B$mean_rank),]

rm(list = c("all_corr_values_B","chi_sq_p_values_B","t_test_p_values_all_B","univariate_logistic_aic_B"))


# Classe C
var_selection_C <- merge(all_corr_values_C,chi_sq_p_values_C,by = "variable",all=T)
var_selection_C <- merge(var_selection_C,t_test_p_values_all_C,by = "variable",all=T)
var_selection_C <- merge(var_selection_C,univariate_logistic_aic_C,by = "variable",all=T)

var_selection_C$mean_rank <- rowMeans(subset(var_selection_C, 
                                             select = c(corr_rank, chi_rank,t_test_rank,univariate_logistic_rank)), 
                                      na.rm = TRUE)

var_selection_C <- var_selection_C[order(var_selection_C$mean_rank),]

rm(list = c("all_corr_values_C","chi_sq_p_values_C","t_test_p_values_all_C","univariate_logistic_aic_C"))


# Classe D
var_selection_D <- merge(all_corr_values_D,chi_sq_p_values_D,by = "variable",all=T)
var_selection_D <- merge(var_selection_D,t_test_p_values_all_D,by = "variable",all=T)
var_selection_D <- merge(var_selection_D,univariate_logistic_aic_D,by = "variable",all=T)

var_selection_D$mean_rank <- rowMeans(subset(var_selection_D, 
                                             select = c(corr_rank, chi_rank,t_test_rank,univariate_logistic_rank)), 
                                      na.rm = TRUE)

var_selection_D <- var_selection_D[order(var_selection_D$mean_rank),]

rm(list = c("all_corr_values_D","chi_sq_p_values_D","t_test_p_values_all_D","univariate_logistic_aic_D"))


# Classe E
var_selection_E <- merge(all_corr_values_E,chi_sq_p_values_E,by = "variable",all=T)
var_selection_E <- merge(var_selection_E,t_test_p_values_all_E,by = "variable",all=T)
var_selection_E <- merge(var_selection_E,univariate_logistic_aic_E,by = "variable",all=T)

var_selection_E$mean_rank <- rowMeans(subset(var_selection_E, 
                                             select = c(corr_rank, chi_rank,t_test_rank,univariate_logistic_rank)), 
                                      na.rm = TRUE)

var_selection_E <- var_selection_E[order(var_selection_E$mean_rank),]

rm(list = c("all_corr_values_E","chi_sq_p_values_E","t_test_p_values_all_E","univariate_logistic_aic_E"))

rm(list = c("training_univariate","chi_sq","logistic_prof","t_tests"))


#########################################################################################################
#######                                       STEP 2                                      ###############
#######     Run variable clustering to understand the similarity between predictors       ###############
#########################################################################################################

drops <- names(training) %in% c("magnet_belt_z_sqrt")
training <- training[!drops]

# numeric variables only
var_clust_quant <- select(training,-classe)

var_clust_quant <- sapply(var_clust_quant, as.numeric )


tree <- hclustvar(var_clust_quant)
plot(tree)

# cut tree with 35 clusters
P35 <- cutreevar(tree,35,matsim=TRUE)
P35$var
var_clusters <- as.data.frame(P35$cluster)
var_clusters$variable <- row.names(var_clusters)

# merge the cluster number to the variable selection data frames
var_selection_A <- merge(var_selection_A,var_clusters,by = "variable")
var_selection_A <- var_selection_A[order(var_selection_A$mean_rank),]
names(var_selection_A)[names(var_selection_A)=="P35$cluster"] <- "cluster"

var_selection_B <- merge(var_selection_B,var_clusters,by = "variable")
var_selection_B <- var_selection_B[order(var_selection_B$mean_rank),]
names(var_selection_B)[names(var_selection_B)=="P35$cluster"] <- "cluster"

var_selection_C <- merge(var_selection_C,var_clusters,by = "variable")
var_selection_C <- var_selection_C[order(var_selection_C$mean_rank),]
names(var_selection_C)[names(var_selection_C)=="P35$cluster"] <- "cluster"

var_selection_D <- merge(var_selection_D,var_clusters,by = "variable")
var_selection_D <- var_selection_D[order(var_selection_D$mean_rank),]
names(var_selection_D)[names(var_selection_D)=="P35$cluster"] <- "cluster"

var_selection_E <- merge(var_selection_E,var_clusters,by = "variable")
var_selection_E <- var_selection_E[order(var_selection_E$mean_rank),]
names(var_selection_E)[names(var_selection_E)=="P35$cluster"] <- "cluster"

rm(list = c("var_clusters","var_clust_quant","P35","tree"))

# save the variabel selection datasets
save(var_selection_A,file="var_selection_A.Rda")
save(var_selection_B,file="var_selection_B.Rda")
save(var_selection_C,file="var_selection_C.Rda")
save(var_selection_D,file="var_selection_D.Rda")
save(var_selection_E,file="var_selection_E.Rda")

#########################################################################################################
#######                                       STEP 3                                      ###############
#######       Create descriptive plots of the predictors and classe variables             ###############
#########################################################################################################

# count of classe variable
classe_count <- as.data.frame(table(training$classe))
classe_count <- subset(classe_count,Var1 %in% c("A","B","C","D","E"))

barplot(classe_count$Freq,
        names.arg = classe_count$Var1,
        col = "red",
        xlab = "Classe",
        cex.axis = 1,
        ylab = "Classe Count",
        main = "Training Set Classe Count")

rm(classe_count)

####### plot the top 4 variables for classe A (that are not in the same variable cluster)  #######

a <- ggplot(training,aes(factor(classe), pitch_forearm)) +
            geom_boxplot(fill = "red") +
            theme_bw(base_family = "Arial") + 
            ggtitle("Pitch Forearm by Classe") +
            xlab("Classe") + 
            ylab("Pitch Forearm")
            theme(axis.title.y=element_text(vjust=1,size=14),
                  axis.title.x=element_text(vjust=-0.5,size=14),
                  plot.title=element_text(vjust=1,size=16),
                  axis.text.x = element_text(size = 12),
                  axis.text.y = element_text(size = 12))
a

b <- ggplot(training,aes(factor(classe), roll_forearm_squared)) +
            geom_boxplot(fill = "red") +
            theme_bw(base_family = "Arial") + 
            ggtitle("Roll Forearm ^2 by Classe") +
            xlab("Classe") + 
            ylab("Roll Forearm ^2")
            theme(axis.title.y=element_text(vjust=1,size=14),
                  axis.title.x=element_text(vjust=-0.5,size=14),
                  plot.title=element_text(vjust=1,size=16),
                  axis.text.x = element_text(size = 12),
                  axis.text.y = element_text(size = 12))
b

c <- ggplot(training,aes(factor(classe), accel_arm_x_squared)) +
            geom_boxplot(fill = "red") +
            theme_bw(base_family = "Arial") + 
            ggtitle("Accel Arm X ^2 by Classe") +
            xlab("Classe") + 
            ylab("Accel Arm X ^2")
            theme(axis.title.y=element_text(vjust=1,size=14),
                  axis.title.x=element_text(vjust=-0.5,size=14),
                  plot.title=element_text(vjust=1,size=16),
                  axis.text.x = element_text(size = 12),
                  axis.text.y = element_text(size = 12))
c

d <- ggplot(training,aes(factor(classe), magnet_dumbbell_x_squared)) +
            geom_boxplot(fill = "red") +
            theme_bw(base_family = "Arial") + 
            ggtitle("Magnet Dumbbell X ^2 by Classe") +
            xlab("Classe") + 
            ylab("Magnet Dumbbell X ^2")
            theme(axis.title.y=element_text(vjust=1,size=14),
                  axis.title.x=element_text(vjust=-0.5,size=14),
                  plot.title=element_text(vjust=1,size=16),
                  axis.text.x = element_text(size = 12),
                  axis.text.y = element_text(size = 12))
d

grid.arrange(a, b, c, d, ncol=2, main = "Classe A Top (Non-Correlated) Variables")

####### plot the top 4 variables for classe B (that are not in the same variable cluster)  #######

a <- ggplot(training,aes(factor(classe), gyros_dumbbell_y_squared)) +
            geom_boxplot(fill = "red") +
            theme_bw(base_family = "Arial") + 
            ggtitle("Gyros DumbbellY ^2 by Classe") +
            xlab("Classe") + 
            ylab("Gyros DumbbellY ^2")
            theme(axis.title.y=element_text(vjust=1,size=14),
                  axis.title.x=element_text(vjust=-0.5,size=14),
                  plot.title=element_text(vjust=1,size=16),
                  axis.text.x = element_text(size = 12),
                  axis.text.y = element_text(size = 12))
a

b <- ggplot(training,aes(factor(classe), magnet_dumbbell_y_squared)) +
            geom_boxplot(fill = "red") +
            theme_bw(base_family = "Arial") + 
            ggtitle("Magnet Dumbbell Y ^2 by Classe") +
            xlab("Classe") + 
            ylab("Magnet Dumbbell Y ^2")
            theme(axis.title.y=element_text(vjust=1,size=14),
                  axis.title.x=element_text(vjust=-0.5,size=14),
                  plot.title=element_text(vjust=1,size=16),
                  axis.text.x = element_text(size = 12),
                  axis.text.y = element_text(size = 12))
b

c <- ggplot(training,aes(factor(classe), accel_dumbbell_x_sqrt)) +
            geom_boxplot(fill = "red") +
            theme_bw(base_family = "Arial") + 
            ggtitle("Sq Rt. Accel Dumbbell X by Classe") +
            xlab("Classe") + 
            ylab("Sq Rt. Accel Dumbbell X")
            theme(axis.title.y=element_text(vjust=1,size=14),
                  axis.title.x=element_text(vjust=-0.5,size=14),
                  plot.title=element_text(vjust=1,size=16),
                  axis.text.x = element_text(size = 12),
                  axis.text.y = element_text(size = 12))
c

d <- ggplot(training,aes(factor(classe), gyros_arm_y_squared)) +
  geom_boxplot(fill = "red") +
  theme_bw(base_family = "Arial") + 
  ggtitle("Gyros Arm Y ^2 by Classe") +
  xlab("Classe") + 
  ylab("Gyros Arm Y ^2")
  theme(axis.title.y=element_text(vjust=1,size=14),
        axis.title.x=element_text(vjust=-0.5,size=14),
        plot.title=element_text(vjust=1,size=16),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))
d

grid.arrange(a, b, c, d, ncol=2, main = "Classe B Top (Non-Correlated) Variables")


####### plot the top 4 variables for classe C (that are not in the same variable cluster)  #######

a <- ggplot(training,aes(factor(classe), roll_dumbbell_cubed)) +
            geom_boxplot(fill = "red") +
            theme_bw(base_family = "Arial") + 
            ggtitle("Roll Dumbbell ^3 by Classe") +
            xlab("Classe") + 
            ylab("Roll Dumbbell ^3")
            theme(axis.title.y=element_text(vjust=1,size=14),
                 axis.title.x=element_text(vjust=-0.5,size=14),
                 plot.title=element_text(vjust=1,size=16),
                 axis.text.x = element_text(size = 12),
                 axis.text.y = element_text(size = 12))
a

b <- ggplot(training,aes(factor(classe), magnet_dumbbell_x_squared)) +
            geom_boxplot(fill = "red") +
            theme_bw(base_family = "Arial") + 
            ggtitle("Magnet Dumbbell X ^2 by Classe") +
            xlab("Classe") + 
            ylab("Magnet Dumbbell X ^2")
            theme(axis.title.y=element_text(vjust=1,size=14),
                  axis.title.x=element_text(vjust=-0.5,size=14),
                  plot.title=element_text(vjust=1,size=16),
                  axis.text.x = element_text(size = 12),
                  axis.text.y = element_text(size = 12))
b

c <- ggplot(training,aes(factor(classe), magnet_dumbbell_y_squared)) +
            geom_boxplot(fill = "red") +
            theme_bw(base_family = "Arial") + 
            ggtitle("Magnet Dumbbell Y ^2 by Classe") +
            xlab("Classe") + 
            ylab("Magnet Dumbbell Y ^2")
            theme(axis.title.y=element_text(vjust=1,size=14),
                 axis.title.x=element_text(vjust=-0.5,size=14),
                 plot.title=element_text(vjust=1,size=16),
                 axis.text.x = element_text(size = 12),
                 axis.text.y = element_text(size = 12))
c

d <- ggplot(training,aes(factor(classe), pitch_dumbbell)) +
            geom_boxplot(fill = "red") +
            theme_bw(base_family = "Arial") + 
            ggtitle("Ptich Dumbbell by Classe") +
            xlab("Classe") + 
            ylab("Pitch Dumbbell")
            theme(axis.title.y=element_text(vjust=1,size=14),
            axis.title.x=element_text(vjust=-0.5,size=14),
            plot.title=element_text(vjust=1,size=16),
            axis.text.x = element_text(size = 12),
            axis.text.y = element_text(size = 12))
d

grid.arrange(a, b, c, d, ncol=2, main = "Classe C Top (Non-Correlated) Variables")

####### plot the top 4 variables for classe D (that are not in the same variable cluster)  #######

a <- ggplot(training,aes(factor(classe), pitch_forearm)) +
            geom_boxplot(fill = "red") +
            theme_bw(base_family = "Arial") + 
            ggtitle("Pitch Forearm by Classe") +
            xlab("Classe") + 
            ylab("Pitch Forearm")
            theme(axis.title.y=element_text(vjust=1,size=14),
                  axis.title.x=element_text(vjust=-0.5,size=14),
                  plot.title=element_text(vjust=1,size=16),
                  axis.text.x = element_text(size = 12),
                  axis.text.y = element_text(size = 12))
a

b <- ggplot(training,aes(factor(classe), accel_forearm_x)) +
            geom_boxplot(fill = "red") +
            theme_bw(base_family = "Arial") + 
            ggtitle("Accel Forearm X by Classe") +
            xlab("Classe") + 
            ylab("Accel Forearm X")
            theme(axis.title.y=element_text(vjust=1,size=14),
                  axis.title.x=element_text(vjust=-0.5,size=14),
                  plot.title = element_text(vjust=1,size=16),
                  axis.text.x = element_text(size = 12),
                  axis.text.y = element_text(size = 12))
b

c <- ggplot(training,aes(factor(classe), magnet_arm_y_squared)) +
            geom_boxplot(fill = "red") +
            theme_bw(base_family = "Arial") + 
            ggtitle("Magnet Arm Y ^2 by Classe") +
            xlab("Classe") + 
            ylab("Magnet Arm Y ^2")
            theme(axis.title.y=element_text(vjust=1,size=14),
                  axis.title.x=element_text(vjust=-0.5,size=14),
                  plot.title=element_text(vjust=1,size=16),
                  axis.text.x = element_text(size = 12),
                  axis.text.y = element_text(size = 12))
c

d <- ggplot(training,aes(factor(classe), roll_dumbbell)) +
            geom_boxplot(fill = "red") +
            theme_bw(base_family = "Arial") + 
            ggtitle("Roll Dumbbell by Classe") +
            xlab("Classe") + 
            ylab("Roll Dumbbell")
            theme(axis.title.y=element_text(vjust=1,size=14),
                  axis.title.x=element_text(vjust=-0.5,size=14),
                  plot.title=element_text(vjust=1,size=16),
                  axis.text.x = element_text(size = 12),
                  axis.text.y = element_text(size = 12))
d

grid.arrange(a, b, c, d, ncol=2, main = "Classe D Top (Non-Correlated) Variables")

####### plot the top 4 variables for classe E (that are not in the same variable cluster)  #######

a <- ggplot(training,aes(factor(classe), magnet_belt_y_sqrt)) +
            geom_boxplot(fill = "red") +
            theme_bw(base_family = "Arial") + 
            ggtitle("Magnet Belt Y ^0.5 by Classe") +
            xlab("Classe") + 
            ylab("Magnet Belt Y ^0.5")
            theme(axis.title.y=element_text(vjust=1,size=14),
                  axis.title.x=element_text(vjust=-0.5,size=14),
                  plot.title=element_text(vjust=1,size=16),
                  axis.text.x = element_text(size = 12),
                  axis.text.y = element_text(size = 12))
a

b <- ggplot(training,aes(factor(classe), gyros_belt_z_sqrt)) +
            geom_boxplot(fill = "red") +
            theme_bw(base_family = "Arial") + 
            ggtitle("Gyros Belt Z ^0.5 by Classe") +
            xlab("Classe") + 
            ylab("Gyros Belt Z ^0.5")
            theme(axis.title.y=element_text(vjust=1,size=14),
                  axis.title.x=element_text(vjust=-0.5,size=14),
                  plot.title = element_text(vjust=1,size=16),
                  axis.text.x = element_text(size = 12),
                  axis.text.y = element_text(size = 12))
b

c <- ggplot(training,aes(factor(classe), yaw_belt_log)) +
            geom_boxplot(fill = "red") +
            theme_bw(base_family = "Arial") + 
            ggtitle("Yaw Belt (log) by Classe") +
            xlab("Classe") + 
            ylab("Yaw Belt (log)")
            theme(axis.title.y=element_text(vjust=1,size=14),
                  axis.title.x=element_text(vjust=-0.5,size=14),
                  plot.title=element_text(vjust=1,size=16),
                  axis.text.x = element_text(size = 12),
                  axis.text.y = element_text(size = 12))
c

d <- ggplot(training,aes(factor(classe), roll_belt_cubed)) +
            geom_boxplot(fill = "red") +
            theme_bw(base_family = "Arial") + 
            ggtitle("Roll Belt ^3 by Classe") +
            xlab("Classe") + 
            ylab("Roll Belt ^3")
            theme(axis.title.y=element_text(vjust=1,size=14),
                  axis.title.x=element_text(vjust=-0.5,size=14),
                  plot.title=element_text(vjust=1,size=16),
                  axis.text.x = element_text(size = 12),
                  axis.text.y = element_text(size = 12))
d

grid.arrange(a, b, c, d, ncol=2, main = "Classe E Top (Non-Correlated) Variables")

#### Create correlation plots of the top predictors within each cluster for each classe ####

## Classe A

# get the top variable within each cluster 
classe_A_var_sort <- var_selection_A[order(var_selection_A$cluster, var_selection_A$mean_rank), ]
top_A_vars <- by(classe_A_var_sort, classe_A_var_sort$cluster, head, n=1)
top_A_vars <- do.call("rbind", as.list(top_A_vars))
top_A_vars <- top_A_vars[order(top_A_vars$mean_rank), ]

# select the top 10 variables
top_10_vars <- top_A_vars[1:10,1]
top_10_vars <- training[,top_10_vars]
classe <- training$classe
top_10_vars <- cbind(top_10_vars,classe)
top_10_vars$classe <- ifelse(classe == "A","A","Other")
top_10_vars$classe <- as.factor(top_10_vars$classe)
table(top_10_vars$classe)

# take a random sample of the training data
top_10_vars.samp <- top_10_vars[sample(1:dim(top_10_vars)[1],2000),]

a <- ggpairs(top_10_vars.samp,
             title = "Top 10 Univariate Predictors - Classe A",
             colour = "classe",
             alpha = 0.3,
             axisLabels = "internal")

a

rm(list = c("classe_A_var_sort","top_10_vars","top_10_vars.samp","top_A_vars"))

## Classe B

# get the top variable within each cluster 
classe_B_var_sort <- var_selection_B[order(var_selection_B$cluster, var_selection_B$mean_rank), ]
top_B_vars <- by(classe_B_var_sort, classe_B_var_sort$cluster, head, n=1)
top_B_vars <- do.call("rbind", as.list(top_B_vars))
top_B_vars <- top_B_vars[order(top_B_vars$mean_rank), ]

top_10_vars <- top_B_vars[1:10,1]
top_10_vars <- training[,top_10_vars]
classe <- training$classe
top_10_vars <- cbind(top_10_vars,classe)
top_10_vars$classe <- ifelse(classe == "B","B","Other")
top_10_vars$classe <- as.factor(top_10_vars$classe)
table(top_10_vars$classe)

# take a random sample of the training data
top_10_vars.samp <- top_10_vars[sample(1:dim(top_10_vars)[1],2000),]

b <- ggpairs(top_10_vars.samp,
             title = "Top 10 Univariate Predictors - Classe B",
             colour = "classe",
             alpha = 0.3,
             axisLabels = "internal")

b

rm(list = c("classe_B_var_sort","top_10_vars","top_10_vars.samp","top_B_vars"))

## Classe C

# get the top variable within each cluster 
classe_C_var_sort <- var_selection_C[order(var_selection_C$cluster, var_selection_C$mean_rank), ]
top_C_vars <- by(classe_C_var_sort, classe_C_var_sort$cluster, head, n=1)
top_C_vars <- do.call("rbind", as.list(top_C_vars))
top_C_vars <- top_C_vars[order(top_C_vars$mean_rank), ]

top_10_vars <- top_C_vars[1:10,1]
top_10_vars <- training[,top_10_vars]
classe <- training$classe
top_10_vars <- cbind(top_10_vars,classe)
top_10_vars$classe <- ifelse(classe == "C","C","Other")
top_10_vars$classe <- as.factor(top_10_vars$classe)
table(top_10_vars$classe)

# take a random sample of the training data
top_10_vars.samp <- top_10_vars[sample(1:dim(top_10_vars)[1],2000),]

c <- ggpairs(top_10_vars.samp,
             title = "Top 10 Univariate Predictors - Classe C",
             colour = "classe",
             alpha = 0.3,
             axisLabels = "internal")

c

rm(list = c("classe_C_var_sort","top_10_vars","top_10_vars.samp","top_C_vars"))

## Classe D

# get the top variable within each cluster 
classe_D_var_sort <- var_selection_D[order(var_selection_D$cluster, var_selection_D$mean_rank), ]
top_D_vars <- by(classe_D_var_sort, classe_D_var_sort$cluster, head, n=1)
top_D_vars <- do.call("rbind", as.list(top_D_vars))
top_D_vars <- top_D_vars[order(top_D_vars$mean_rank), ]

top_10_vars <- top_D_vars[1:10,1]
top_10_vars <- training[,top_10_vars]
classe <- training$classe
top_10_vars <- cbind(top_10_vars,classe)
top_10_vars$classe <- ifelse(classe == "D","D","Other")
top_10_vars$classe <- as.factor(top_10_vars$classe)
table(top_10_vars$classe)

# take a random sample of the training data
top_10_vars.samp <- top_10_vars[sample(1:dim(top_10_vars)[1],2000),]

d <- ggpairs(top_10_vars.samp,
             title = "Top 10 Univariate Predictors - Classe D",
             colour = "classe",
             alpha = 0.3,
             axisLabels = "internal")

d

rm(list = c("classe_D_var_sort","top_10_vars","top_10_vars.samp","top_D_vars"))

## Classe E

# get the top variable within each cluster 
classe_E_var_sort <- var_selection_E[order(var_selection_E$cluster, var_selection_E$mean_rank), ]
top_E_vars <- by(classe_E_var_sort, classe_E_var_sort$cluster, head, n=1)
top_E_vars <- do.call("rbind", as.list(top_E_vars))
top_E_vars <- top_E_vars[order(top_E_vars$mean_rank), ]

top_10_vars <- top_E_vars[1:10,1]
top_10_vars <- training[,top_10_vars]
classe <- training$classe
top_10_vars <- cbind(top_10_vars,classe)
top_10_vars$classe <- ifelse(classe == "E","E","Other")
top_10_vars$classe <- as.factor(top_10_vars$classe)
table(top_10_vars$classe)

# take a random sample of the training data
top_10_vars.samp <- top_10_vars[sample(1:dim(top_10_vars)[1],2000),]

e <- ggpairs(top_10_vars.samp,
             title = "Top 10 Univariate Predictors - Classe E",
             colour = "classe",
             alpha = 0.3,
             axisLabels = "internal")

e

rm(list = c("classe_E_var_sort","top_10_vars","top_10_vars.samp","top_E_vars"))

