# Ch 3 - Classification

## Chapter Objectives

1. Apply the ML process through a classification problem
2. Learn various performance measures for a classification problem
  a. Cross-Validation
  b. Confusion matrix
  c. Precision and recall
3. Learn about the trade-off between precision and recall
4. Implement a multiclass classifier
5. Learn about multilabel classification
6. Learn about multioutput classification

In the last chapter, we went balls-to-the-wall on a regression task and went through the entire ML process in great detail. In this chapter, we do the same for a classification task but focus more on various performance measures, and less on outlining the ML process. As it turns out, analyzing the performance of a classification task is very different from a regression task, so we will invest much time learning about this in detail. 


## 3.0 - Intro to MNIST Dataset

The data set we will use is the very well-known "MNIST" data set. This data set consists of 70,000 images of handwritten digits (0,1,...,9) - we will create a ML model in an attempt to classify these images. This data set is actually a subset of a much more larger and lesser-known data set called the "NIST Special Database 19". This data set is much more general in that it contains images of forms with both alphabet and numerical handwriting samples. It was used by NIST in an attempt to make a proper OCR algorithm. This project was essentially abandoned by NIST in 1993, but an update was made in 2003. No more updates are expected. See here for more details: https://www.nist.gov/srd/nist-special-database-19

To procure the data, Scikit-Learn has a built-in function that allows one to download popular data sets. There are a few ways to do this:
1. Small ("toy") data sets -> 
2. Large ("real world") data sets ->
3. Generated data sets ->
4. Download from OpenML repository: This is a public repository of many user-made data sets -> https://www.openml.org

https://www.openml.org/d/554

http://yann.lecun.com/exdb/mnist/ -> original website

We will be downloading the MNIST data set from the openml.org repository, and so we can use SKL's 'datasets.fetch_openml' function. Once downloaded, we can get some key information about what this data set contains using various SKL methods. Every data set downloaded through SKL comes loaded as a dictionary with the same keys, so we can look at each key in turn:
* .DESCR -> Contains detailed description about the data set
* .details -> Contains various meta-data properties of the data set
* .data -> Contains the actual data points (NumPy array)
* .target -> Contains associated labels for each data point (NumPy array)
* .feature_names -> Contains names of all feature columns (784 columns for MNIST; each digit image is a 28x28 pixel grid)
* .target_names -> Contains names of all target columns (only one column for MNIST)
* .url -> Gives URL of data set on the OpenML repository

You will find that there are 70,000 images with each image posessing 784 features.  This is because each image is in fact a 28x28 grid of pixel, and so each feature is the grayscale value of a particular pixel (0 = white; 255 = black). We can visualize these images simply by using matplotlib.

The final (and very important!) thing we should do is to **split our data into a training and test set**. As it happens, the MNIST data set is already split into a training and test set for us: The first 60,000 instances make the training set, and the last 10,000 instances make the test set. Moreover, the training set comes pre-shuffled, meaning that all our cross-validation folds will have the same ratio of digits.


## 3.1 - Training a Binary Classifier



## 3.2 - Performance Measures



### 3.2.1 - Measuring Accuracy Using Cross-Validation
### 3.2.2 - Confusion Matrix
### 3.2.3 - Precision and Recall
### 3.2.4 - Precision/Recall Trade-off
### 3.2.5 - The ROC Curve


## 3.3 - Multiclass Classification
## 3.4 - Error Analysis
## 3.5 - Multilabel Classification
## 3.6 - Multioutput Classification


## - Concluding Remarks

In this chapter we have learned the many aspects of a classification task.





[anomaly_detection]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/anomaly_detection.png "illustration of anomaly detection"
