# thesis_experiments

This is a repository is for extract training metafeatures. 

In order to run the code and etract the training metafeatures replicate the following steps:


* Download or clone the folder on your PC.
* Open the **readiness_metrics** folder;  contains experiments for calculating the readiness of ADS for particular datasets.
* Open the **extractMetafeatures.R** and change the following command pointing in the ads-h2o folder on your PC:
```R
setwd("/Users/christina/Documents/GitHub/ads-h2o")
```
* Run the **extractMetafeatures.R** . 
* The datasets allocate on the dropbox; no need to add the datasets under the workspace folder. 
* In case your data is in a different folder please change the directory path from the **extractMetafeatures.R**  change the following command pointing in the directory with the processed data folder. :

```R
repo <-"/Users/christina/Dropbox/AutoML/Datasets/Classification/dataset-name/processed"
```
You can find the excel file containing the training meatfeatures on [trainning metafeatures](https://github.com/issel-ml-squad/ads-h2o/blob/MFextractor_v1/thesis_experiments/readiness_metric/training_metafeatures.csv)

