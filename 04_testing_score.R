# Practical Machine Learning Assignment
# 04_testing_score

# This code creates the predicted classes on the 20 testing samples

setwd("/users/Richard/Documents/Coursera Data Science Track/Practical Machine Learning/Assignment")
getwd()

# load the testing data
load("testing.Rda")

#########################################################################################################
#######                                       STEP 1                                      ###############
#######                          Create the variable transformations                      ###############
#########################################################################################################

###### log ######

drops <- c("classe")
log_transform <- testing[,!names(testing) %in% drops]
rm(drops)

lapply(names(log_transform),
       
       transform <- function (x){
         
         var <- eval(parse(text=paste("testing$", x,sep="")))
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
squared_transform <- testing[,!names(testing) %in% drops]
rm(drops)

lapply(names(squared_transform),
       
       transform <- function (x){
         
         var <- eval(parse(text=paste("testing$", x,sep="")))
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
cubed_transform <- testing[,!names(testing) %in% drops]
rm(drops)

lapply(names(cubed_transform),
       
       transform <- function (x){
         
         var <- eval(parse(text=paste("testing$", x,sep="")))
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
sqrt_transform <- testing[,!names(testing) %in% drops]
rm(drops)

lapply(names(sqrt_transform),
       
       transform <- function (x){
         
         var <- eval(parse(text=paste("testing$", x,sep="")))
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
cubrt_transform <- testing[,!names(testing) %in% drops]
rm(drops)

lapply(names(cubrt_transform),
       
       transform <- function (x){
         
         var <- eval(parse(text=paste("testing$", x,sep="")))
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
exp_transform <- testing[,!names(testing) %in% drops]
rm(drops)

lapply(names(exp_transform),
       
       transform <- function (x){
         
         var <- eval(parse(text=paste("testing$", x,sep="")))
         exp_transform <- exp_transform[,!names(exp_transform) == x]
         
         exp_var <- (var)^1/3
         exp_transform <- cbind(exp_transform,exp_var)
         names(exp_transform)[names(exp_transform)=="exp_var"] <- paste(x,"_exp",sep="")
         assign("exp_transform", exp_transform, envir = .GlobalEnv) 
         return(exp_transform)
       })

exp_transform[is.na(exp_transform)] <- 0.000001

rm(transform)

testing <- cbind(testing,exp_transform,log_transform,sqrt_transform,squared_transform,cubed_transform,cubrt_transform)

rm(list = c("exp_transform","log_transform","sqrt_transform","squared_transform","cubed_transform",
            "cubrt_transform"))

#########################################################################################################
#######                                       STEP 2                                      ###############
#######                                  Predict the Classe                               ###############
#########################################################################################################

# Random Forest
rf.predictions <- predict(best_rf,newdata=testing)

