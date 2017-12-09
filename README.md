
## ads-h2o

Automated Data Scientist with H2O: AutoML+H2O for classification tasks

This repository is the first version of our work.

Please go to the **thesis_experiments** folder open the **readiness_metrics** subfolder and run the **extractMetafeatures.R** scrip.


# thesis_experiments

This folder is for training metafeatures extraction.

In order to run the code and etract the training metafeatures replicate the following steps:


* Download or clone the folder on your PC.
* Open the **readiness_metrics** folder;  contains experiments for calculating the readiness of ADS for particular datasets.
* Open the **extractMetafeatures.R** and change the following command pointing in the ads-h2o folder on your PC:
```R
setwd("/Users/christina/Desktop/ads-master")
```
* Run the **extractMetafeatures.R** . 
* The datasets allocate on the dropbox; no need to add the datasets under the workspace folder. 
* In case your data is in a different folder please change the directory path from the **extractMetafeatures.R**  change the following command pointing in the directory with the processed data folder. :

```R
repo <-"/Users/christina/Dropbox/AutoML/Datasets/Classification/dataset-name/processed"
```


[Here](https://github.com/issel-ml-squad/ads-h2o/blob/master/thesis_experiments/readiness_metric/training_metafeatures.csv) you can find the training metafeatures of 131 datasets. 