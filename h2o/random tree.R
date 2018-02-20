# decision trees h2o autoML
# Parameteres : ntrees & max_depth


rm(list=ls())
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

## Load a file from disk
setwd("/Users/christina")

# setwd("~/h2o-tutorials/tutorials/deeplearning")
df <- h2o.importFile(path = normalizePath("./Dropbox/AutoML/Datasets/Classification/dataset-name/processed/bank-additional.csv"))
response <- "Class"
df[[response]] <- as.factor(df[[response]])       
predictors <- setdiff(names(df), response)
predictors
splits <- h2o.splitFrame(
  df,           ##  splitting the H2O frame we read above
  c(0.6,0.2),   ##  create splits of 60% and 20%; 
  ##  H2O will create one more split of 1-(sum of these parameters)
  ##  so we will get 0.6 / 0.2 / 1 - (0.6+0.2) = 0.6/0.2/0.2
  seed=1234)    ##  setting a seed will ensure reproducible results (not R's seed)

train <- h2o.assign(splits[[1]], "train.hex")   
## assign the first result the R variable train
## and the H2O name train.hex
valid <- h2o.assign(splits[[2]], "valid.hex")   ## R valid, H2O valid.hex
test <- h2o.assign(splits[[3]], "test.hex")     ## R test, H2O test.hex

## take a look at the first few rows of the data set
train[1:5,]   ## rows 1-5, all columns

## run our first predictive model
rf1 <- h2o.randomForest(         
  training_frame = train,        
  validation_frame = valid,      
  x=predictors,                       
  y=response,                          
  model_id = "rf_covType_v1"    ## name the model in H2O
)   

summary(rf1)   
###-----grid search----trial 2 ----------

# Construct a large Cartesian hyper-parameter space
ntrees_opts = c(10000)       # early stopping will stop earlier
max_depth_opts = seq(1,20)
# min_rows_opts = c(1,5,10,20,50,100)
# learn_rate_opts = seq(0.001,0.01,0.001)
# sample_rate_opts = seq(0.3,1,0.05)
# col_sample_rate_opts = seq(0.3,1,0.05)
# col_sample_rate_per_tree_opts = seq(0.3,1,0.05)
# #nbins_cats_opts = seq(100,10000,100) # no categorical features
# # in this dataset

hyper_params = list( ntrees = ntrees_opts, 
                     max_depth = max_depth_opts
                     # min_rows = min_rows_opts, 
                     # learn_rate = learn_rate_opts,
                     # sample_rate = sample_rate_opts,
                     # col_sample_rate = col_sample_rate_opts,
                     # col_sample_rate_per_tree = col_sample_rate_per_tree_opts
                     #,nbins_cats = nbins_cats_opts
)


# Search a random subset of these hyper-parmameters. Max runtime 
# and max models are enforced, and the search will stop after we 
# don't improve much over the best 5 random models.
search_criteria = list(strategy = "RandomDiscrete", 
                       max_runtime_secs = 600, 
                       max_models = 100, 
                       stopping_metric = "AUTO", 
                       stopping_tolerance = 0.00001, 
                       stopping_rounds = 5, 
                       seed = 123456)

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
                     
                     # how often to score (affects early stopping):
                     score_tree_interval = 100, 
                     seed = 123456,
                     hyper_params = hyper_params,
                     search_criteria = search_criteria)

gbm_sorted_grid <- h2o.getGrid(grid_id = "mygrid", sort_by = "mse")
print(gbm_sorted_grid)

best_model <- h2o.getModel(gbm_sorted_grid@model_ids[[1]])
summary(best_model)
max_depth = best_model@parameters$max_depth
n_tress = best_model@parameters$ntrees

write.csv(h2o_test, "./thesis_experiments/readiness_metric/h2o_test.csv")
