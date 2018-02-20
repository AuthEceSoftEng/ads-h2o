# This script generates training metafeatures.

##  --- set up --- 
rm(list=ls())
setwd("C:/Users/christina/Desktop/AutoML/ads-master v2")

pkgs <- c("methods","statmod","stats","graphics","RCurl","jsonlite","tools","utils")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}

#-- set up h2o package----
library(h2o)
## Create an H2O cloud 
h2o.init(
  nthreads=-1,            ## -1: use all available threads
  max_mem_size = "2G")    ## specify the memory size for the H2O cloud
h2o.removeAll() # Clean slate - just in case the cluster was already running

#---- import datasets from directory ----

repo       <-"/Users/christina/Dropbox/AutoML/Datasets/Classification/dataset-name/processed"
files_list <- list.files(path = repo,  pattern="*.csv", recursive = TRUE)


for(i in 1: length(files_list)) {
  dataset_path <- files_list[[i]]
  dataset_name <- substr(dataset_path,1,nchar(dataset_path)-4)
  print("Importing data.......")
  print(dataset_name)
  df <- h2o.importFile(path = normalizePath( paste(repo, dataset_path, sep = "/")))
  response <- "Class"
  df[[response]] <- as.factor(df[[response]])       
  predictors <- setdiff(names(df), response)
  splits <- h2o.splitFrame(
    df,           ##  splitting the H2O frame we read above
    c(0.6,0.2))    ##  setting a seed will ensure reproducible results (not R's seed)
  
  train <- h2o.assign(splits[[1]], "train.hex")   
  valid <- h2o.assign(splits[[2]], "valid.hex")   ## R valid, H2O valid.hex
  test <- h2o.assign(splits[[3]], "test.hex")     ## R test, H2O test.hex
  
  # RFd <- h2o.randomForest(predictors, response, train, model_id="RF_defaults", nfolds=10,
  #                         fold_assignment = "Modulo", keep_cross_validation_predictions = T,max_runtime_secs = 90, ntrees = 1000,max_depth = 40)
  # model.h2o = RFd
 #  best_dl_perf <- h2o.performance(model = RFd, 
 #                                  newdata = test)
 # # accurasy1 = h2o.auc(best_dl_perf)
 #  accurasy = 1 - best_dl_perf@metrics$RMSE
 #  
 #  model = RFd@algorithm
 #  #summary(RFd)
 #  test = RFd@model
 #  parameters =  test$model_summary
  # 
  # g <- h2o.grid("randomForest",
  #               hyper_params = list(
  #                 ntrees = c(50, 100, 120,500,700,1000),
  #                 max_depth = c(40, 60),
  #                 min_rows = c(1, 2)
  #               ),
  #               x = predictors, y = response, training_frame = train, nfolds = 10
  # )
  
  g1 <- h2o.grid("randomForest",
                 search_criteria = list(
                   strategy = "RandomDiscrete",
                   max_models = 600 #Of the 108 possible
                   
                 ),
                 hyper_params = list(
                   ntrees = c(50, 100, 120,500,700,1000),
                   max_depth = c(40, 60)
                 ),
                 x = predictors, y = response, training_frame = train,
                 validation_frame = test,
                 stopping_metric = "RMSE",
                 stopping_rounds = 3,
                 score_tree_interval = 3
                 
  )
  
  
  g_r2 <- h2o.getGrid(g2@grid_id, sort_by = "accuracy", decreasing = TRUE)
  
  g2 <- h2o.grid("randomForest",
                 search_criteria = list(
                   strategy = "RandomDiscrete",
                   max_models = 600 #Of the 108 possible
                   
                 ),
                 hyper_params = list(
                   ntrees = c(50, 100, 120,500,700,1000),
                   max_depth = c(40, 60)
                 ),
                 x = predictors, y = response, training_frame = train,
                 validation_frame = test,
                 stopping_metric = "AUC",
                 stopping_rounds = 3,
                 score_tree_interval = 3
                 
  )
  
  
  g_r2 <- h2o.getGrid(g2@grid_id, sort_by = "auc", decreasing = TRUE)
  model.h2o <- h2o.getModel(g_r2@model_ids[[1]])
  best_dl_perf <- h2o.performance(model = model.h2o, 
                                  newdata = test)
  # accurasy1 = h2o.auc(best_dl_perf)
  accurasy = best_dl_perf@metrics$AUC
  
  model = "drf"
  ntrees = model.h2o@allparameters$ntrees
  max_depth = model.h2o@allparameters$max_depth
  min_rows = model.h2o@allparameters$min_rows
  parameters =  cbind(ntrees,max_depth,min_rows)
  

  info = cbind(dataset_name,model,accurasy)
  hpp <- cbind(info,parameters)
  if(i == 1 ) {
    total_hpp <- hpp
  } else {
    total_hpp <- rbind(total_hpp,hpp)
  } 
}

write.csv(total_hpp, "total_hpp-final_90sec.csv")
total_hpp = read.csv("total_hpp-final.csv")
k=0
#total_hpp = as.data.frame(total_hpp)

for(i in 1: nrow(total_hpp)) {
  dataset_path <- files_list[[i]]
  dataset_name <- substr(dataset_path,1,nchar(dataset_path)-4)
  print("Importing data.......")
  print(dataset_name)
  acc = total_hpp$accurasy[i]
  if(acc < 0.7){
  k=k+1
  df <- h2o.importFile(path = normalizePath( paste(repo, dataset_path, sep = "/")))
  response <- "Class"
  df[[response]] <- as.factor(df[[response]])       
  predictors <- setdiff(names(df), response)
  splits <- h2o.splitFrame(
    df,           ##  splitting the H2O frame we read above
    c(0.6,0.2))    ##  setting a seed will ensure reproducible results (not R's seed)
  
  train <- h2o.assign(splits[[1]], "train.hex")   
  valid <- h2o.assign(splits[[2]], "valid.hex")   ## R valid, H2O valid.hex
  test <- h2o.assign(splits[[3]], "test.hex")     ## R test, H2O test.hex
  x = predictors
  y = response
  RFd <- h2o.randomForest(x, y, train, model_id="RF_defaults", nfolds=10,ntrees = 200,
                          fold_assignment = "Modulo", keep_cross_validation_predictions = T,max_runtime_secs = 60)
  best_dl_perf <
    - h2o.performance(model = RFd, 
                                  newdata = test)
  # accurasy1 = h2o.auc(best_dl_perf)
  accurasy = 1 - best_dl_perf@metrics$RMSE
  
  model = RFd@algorithm
  #summary(RFd)
  test = RFd@model
  parameters =  test$model_summary
  info = cbind(dataset_name,model,accurasy)
  hpp1 <- cbind(info,parameters)
  if(k == 1 ) {
    total_hpp1 <- hpp1
  } else {
    total_hpp1 <- rbind(total_hpp1,hpp1)
  }
}
}
write.csv(total_hpp1, "total_hpp_tuned.csv")


