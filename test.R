# This script generates training and testing metafeatures.

##  --- set up --- 
rm(list=ls())
setwd("C:/Users/christina/Documents/GitHub/ads-h2o")

##  --- generate testing metafeatures --- 
repo       <-"C:/Users/christina/Dropbox/AutoML/Datasets/Classification/dataset-name/processed"
files_list <- list.files(path = repo,  pattern="*.csv", recursive = TRUE)
for(i in 1:length(files_list)) {
  dataset_path <- files_list[[i]]
  dataset      <- read.csv(paste(repo, dataset_path, sep = "/"),
                           header = TRUE, sep=",", stringsAsFactors=FALSE)
  dataset_name <- substr(dataset_path,1,nchar(dataset_path)-4)
  Num_class = length(unique(dataset$Class))
  rows = nrow(dataset)
  columns = ncol(dataset)
  chk_f =0
  chk_n = 0
    f <- sapply(dataset, is.factor)
    for (k in 1:columns){
      if (f[k]==TRUE){
        chk_f = 1
    }
    if (f[k]==FALSE)
      chk_n = 1
    }
  if(chk_f==1){
    artibutes = "Categorical"
  }else if(chk_n==1){
    artibutes = "Numeric"
  }else if ( chk_f ==1 & chk_n ==1){
    artibutes = "Categorical, Numeric"
  }
  info <- cbind(dataset_name,rows,columns,artibutes,Num_class)
  if(i == 1 ) {
    info_total <- info
  } else {
    info_total <- rbind(info_total,info)
  } 
}
rownames(total_metafeatures) <- files_list
write.csv(info_total, "info.csv")
