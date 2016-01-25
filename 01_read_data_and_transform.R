# Practical Machine Learning Assignment
# 01_read_data_and_transform

# This code reads in the data, removes outliers, treats missing and creates various transformations

library(dplyr)
library(Hmisc)
library(ClustOfVar)
library(ggplot2)
library(gridExtra)
library(GGally)

# set the working directory
setwd("/users/Richard/Documents/Coursera Data Science Track/Practical Machine Learning/Assignment")
getwd()


#########################################################################################################
#######                                       STEP 1                                      ###############
####### Read in the training and test sets and drop columns not required for the modeling ###############
#########################################################################################################

# read in the training and test sets
training <- read.csv("./pml-training.csv",
                     header = T)

testing <- read.csv("./pml-testing.csv",
                    header = T)

summary(training)
str(training)

# what is the frequency of the variable that is to be predicted?
table(training$classe)

# remove the summary metric rows from the dataset

# remove the variables with kurtosis, skewness, max, min, amplitude, var, avg, std, new_windo, X, user, raw, cvtd,num_w
summary_columns <- training[, grep("^kurtosis|^skewness|^max|^min|^amplitude|^var|^avg|^std|^new_w|^X|^user|^raw|^cvtd|^num_w", 
                                   colnames(training))]
drops <- as.vector(names(summary_columns))
training <- training[,!colnames(training)%in%drops] 
testing <- testing[,!colnames(testing)%in%drops]

rm(drops)
rm(summary_columns)

# check for any missing values
sum(is.na(training))
sum(is.na(testing))


#########################################################################################################
#######                                       STEP 2                                      ###############
#######                       Treat outlier values with Winsorization                     ###############
#########################################################################################################

# replace any values that are less than the 5th or greater than the 95th percentile with the 5th or 95th percentile
# this will create a new data frame called training 2 where all values have been winsorized

drops <- c("classe")
vars_to_winsorize <- training[,!names(training) %in% drops]
rm(drops)

training2 <- training

lapply(names(vars_to_winsorize),

  winsorize <- function (x, fraction=.05){
         
  training2 <- training2[,!names(training2) == x]
  var <- eval(parse(text=paste("training$", x,sep="")))
         
  if(length(fraction) != 1 || fraction < 0 || fraction > 0.5) {
           stop("bad value for 'fraction'")
         }
         
   lim <- quantile(var, probs=c(fraction, 1-fraction))
   winsor_var <- ifelse(var < lim[1],lim[1],
                  ifelse(var > lim[2],lim[2],var))
         
   training2 <- cbind(training2,winsor_var)
   names(training2)[names(training2)=="winsor_var"] <- x
   assign("training2", training2, envir = .GlobalEnv) 
   return(training2)
})

# check a variable
quantile(training$pitch_belt, probs = c(0,0.05,0.1,0.25,0.5,0.75,0.95,1))
quantile(training2$pitch_belt, probs = c(0,0.05,0.1,0.25,0.5,0.75,0.95,1))

rm(list = c("vars_to_winsorize","winsorize"))

#########################################################################################################
#######                                       STEP 3                                      ###############
#######                          Create some variable transformations                     ###############
#########################################################################################################

###### log ######

drops <- c("classe")
log_transform <- training2[,!names(training2) %in% drops]
rm(drops)


lapply(names(log_transform),
       
       transform <- function (x){
         
         var <- eval(parse(text=paste("training2$", x,sep="")))
         log_transform <- log_transform[,!names(log_transform) == x]
         
         var <- ifelse(var<=0,0.000001,var)
         
         log_var <- log(var)
         log_transform <- cbind(log_transform,log_var)
         names(log_transform)[names(log_transform)=="log_var"] <- paste(x,"_log",sep="")
         assign("log_transform", log_transform, envir = .GlobalEnv) 
         return(log_transform)
})

# replace the missing values with 0.00001
log_transform[is.na(log_transform)] <- 0.000001

rm(transform)

# these variables were all <= 0 so need to be removed
drops <- names(log_transform) %in% c("magnet_belt_z_log")
log_transform <- log_transform[!drops]
rm(drops)

###### Squared ######

drops <- c("classe")
squared_transform <- training2[,!names(training2) %in% drops]
rm(drops)

lapply(names(squared_transform),
       
       transform <- function (x){
         
         var <- eval(parse(text=paste("training2$", x,sep="")))
         squared_transform <- squared_transform[,!names(squared_transform) == x]
         
         squared_var <- var^2
         squared_transform <- cbind(squared_transform,squared_var)
         names(squared_transform)[names(squared_transform)=="squared_var"] <- paste(x,"_squared",sep="")
         assign("squared_transform", squared_transform, envir = .GlobalEnv) 
         return(squared_transform)
})

# replace the missing values with 0.00001
squared_transform[is.na(squared_transform)] <- 0.000001

rm(transform)


###### Cubed ######

drops <- c("classe")
cubed_transform <- training2[,!names(training2) %in% drops]
rm(drops)

lapply(names(cubed_transform),
       
       transform <- function (x){
         
         var <- eval(parse(text=paste("training2$", x,sep="")))
         cubed_transform <- cubed_transform[,!names(cubed_transform) == x]
         
         cubed_var <- var^3
         cubed_transform <- cbind(cubed_transform,cubed_var)
         names(cubed_transform)[names(cubed_transform)=="cubed_var"] <- paste(x,"_cubed",sep="")
         assign("cubed_transform", cubed_transform, envir = .GlobalEnv) 
         return(cubed_transform)
       })

# replace the missing values with 0.00001
cubed_transform[is.na(cubed_transform)] <- 0.000001

rm(transform)

###### Square Root ######

drops <- c("classe")
sqrt_transform <- training2[,!names(training2) %in% drops]
rm(drops)

lapply(names(sqrt_transform),
       
       transform <- function (x){
         
         var <- eval(parse(text=paste("training2$", x,sep="")))
         sqrt_transform <- sqrt_transform[,!names(sqrt_transform) == x]
         
         sqrt_var <- sqrt(var)
         sqrt_transform <- cbind(sqrt_transform,sqrt_var)
         names(sqrt_transform)[names(sqrt_transform)=="sqrt_var"] <- paste(x,"_sqrt",sep="")
         assign("sqrt_transform", sqrt_transform, envir = .GlobalEnv) 
         return(sqrt_transform)
       })

sqrt_transform[is.na(sqrt_transform)] <- 0.000001

rm(transform)

###### Cubed Root ######

drops <- c("classe")
cubrt_transform <- training2[,!names(training2) %in% drops]
rm(drops)

lapply(names(cubrt_transform),
       
       transform <- function (x){
         
         var <- eval(parse(text=paste("training2$", x,sep="")))
         cubrt_transform <- cubrt_transform[,!names(cubrt_transform) == x]
         
         cubrt_var <- (var)^1/3
         cubrt_transform <- cbind(cubrt_transform,cubrt_var)
         names(cubrt_transform)[names(cubrt_transform)=="cubrt_var"] <- paste(x,"_cubrt",sep="")
         assign("cubrt_transform", cubrt_transform, envir = .GlobalEnv) 
         return(cubrt_transform)
       })

cubrt_transform[is.na(cubrt_transform)] <- 0.000001

rm(transform)

###### Exponent ######

drops <- c("classe")
exp_transform <- training2[,!names(training2) %in% drops]
rm(drops)

lapply(names(exp_transform),
       
       transform <- function (x){
         
         var <- eval(parse(text=paste("training2$", x,sep="")))
         exp_transform <- exp_transform[,!names(exp_transform) == x]
         
         exp_var <- (var)^1/3
         exp_transform <- cbind(exp_transform,exp_var)
         names(exp_transform)[names(exp_transform)=="exp_var"] <- paste(x,"_exp",sep="")
         assign("exp_transform", exp_transform, envir = .GlobalEnv) 
         return(exp_transform)
       })

exp_transform[is.na(exp_transform)] <- 0.000001

rm(transform)

training <- cbind(training2,exp_transform,log_transform,sqrt_transform,squared_transform,cubed_transform,cubrt_transform)

rm(list = c("training2","exp_transform","log_transform","sqrt_transform","squared_transform","cubed_transform",
            "cubrt_transform"))

# save the test and training data frames for the next code
save(training,file="training.Rda")
save(testing,file="testing.Rda")

