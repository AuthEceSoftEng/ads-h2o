# This script generates training metafeatures.

##  --- set up --- 
rm(list=ls())
setwd("C:/Users/christina/Desktop/AutoML/ads-master v2")
pkgs <- c("methods","statmod","stats","graphics","RCurl","jsonlite","tools","utils")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}
library(h2o)
## Create an H2O cloud 
h2o.init(
  nthreads=-1,            ## -1: use all available threads
  max_mem_size = "2G")    ## specify the memory size for the H2O cloud
h2o.removeAll() # Clean slate - just in case the cluster was already running
repo       <-"/Users/christina/Dropbox/AutoML/Datasets/Classification/dataset-name/not run/check"
metafeatures = read.csv("training_metafeatures_v1.csv")
#View(summary(metafeatures[,-1]))
files_list <- list.files(path = repo,  pattern="*.csv", recursive = TRUE)
i=1

#for(i in seq(1, length(files_list))) {

for(i in 1:length(files_list)) {
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
  # Construct a large Cartesian hyper-parameter space
  ntrees_opts = c(10000)       # early stopping will stop earlier
  max_depth_opts = seq(1,20)
  
  
  hyper_params = list( ntrees = ntrees_opts, 
                       max_depth = max_depth_opts
                       # min_rows = min_rows_opts, 
                       #learn_rate = learn_rate_opts
                       # sample_rate = sample_rate_opts,
                       # col_sample_rate = col_sample_rate_opts,
                       # col_sample_rate_per_tree = col_sample_rate_per_tree_opts
                       #,nbins_cats = nbins_cats_opts
  )
  search_criteria = list(strategy = "RandomDiscrete", 
                         max_runtime_secs = 60, 
                         max_models = 20, 
                         stopping_metric = "AUTO", 
                         stopping_tolerance = 0.00001, 
                         stopping_rounds = 5, 
                         seed = 12)
  
  print("tuning hyperparameters.......")
  gbm_grid <- h2o.grid("randomForest", 
                       grid_id = "mygrid",
                       x = predictors, 
                       y = response, 
                       
                       # faster to use a 80/20 split
                       training_frame = train,
                       validation_frame = valid,
                       nfolds = 0,
                       distribution="gaussian",
                       stopping_rounds = 2,
                       stopping_tolerance = 1e-3,
                       stopping_metric = "MSE",
                       score_tree_interval = 100, 
                       seed = 123456,
                       hyper_params = hyper_params,
                       search_criteria = search_criteria)
  
  gbm_sorted_grid <- h2o.getGrid(grid_id = "mygrid", sort_by = "rmse")
  #gbm_sorted_grid <- h2o.getGrid(grid_id = "mygrid", sort_by = "auc",decreasing = TRUE)
  
  best_model <- h2o.getModel(gbm_sorted_grid@model_ids[[1]])
  parameters = best_model@model$model_summary
  #accurasy = h2o.auc(best_model,valid=T) # on test 
  best_dl_perf <- h2o.performance(model = best_model, 
                                  newdata = test)
  accurasy = h2o.auc(best_dl_perf)
  model = best_model@algorithm
  
  info = cbind(dataset_name,model,accurasy)
  hpp <- cbind(info,parameters)
  if(i == 1 ) {
    total_hpp <- hpp
  } else {
    total_hpp <- rbind(total_hpp,hpp)
  } 
}
write.csv(total_hpp, "./Documents/Github/ads-h2o/h2o/total_hpp.csv")
