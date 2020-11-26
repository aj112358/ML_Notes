# Ch 8 - Dimensionality Reduction (DR)

## Chapter Objectives

1. Learn about the 'curse of dimensionality'
2. Learn the two main approaches for dimensionality reduction
    1. Projection
    2. Manifold learning
3. Learn about PCA
    1. Finding principle components
    2. How/why to preserve variance
    3. Implementing with SKL
    4. How to select the dimension
    5. Randomized and incremental PCA
4. Learn about the Kernal PCA method
5. Learn about locally linear embeddings
6. Gain familiarity with some other techniques

It is often the case that your ML problem will involve a very VERY large number of features. As you may expect, this will not only cause your training to take an extraordinate amount of time, but it will also be difficult to uncover hidden patterns and relationships in your data. This idea is referred to as the **curse of dimensionality**.

To remedy this curse, it is possible to *reduce the number of features* in your data set. Of course this may cause some information loss (ex: compressing an image will reduce its quality), so your resulting model *may* have lower performance. Moreover, your *data pipeline* will need to be more complicated and hence harder to maintain. Thus, as a general rule of thumb, you should **first try and train your system with the full original data set *BEFORE* considering dimensionality reduction**. Generally, dimensionality reduction will simply speed up training, and only *in rare cases* will it filter out unwanted noise and details.

As you can also imagine, a data set with a very high number of features does not lend itself to easy visualization techniques. Here, dimensionality reduction can help greatly - by reducing the number of features down to two or three, you can create 2D and 3D plots to visualize your data (albeit a condensed view). Perhaps more importantly, as a data scientist you will need to create visualizations to help lay-people understand your results and put them into action (ie. decision makers, stakeholders, etc.).

So, in this chapter, we will look at some ways to implement dimensionality reduction (DR). Let's get started!


## 8.1 - The Curse of Dimensionality






## 8.2 - Main Approaches for Dimensionality Reduction
### 8.2.1 - Projection
### 8.2.2 - Manifold Learning

## 8.3 - Principle Component Analysis (PCA)
### 8.3.1 - Preserving the Variance
### 8.3.2 - Principle Components
### 8.3.3 - Projecting Down to d Dimensions
### 8.3.4 - Using SKL
### 8.3.5 - Explained Variance Ratio
### 8.3.6 - Choosing the Right Number of Dimensions
### 8.3.7 - PCA for Compression
### 8.3.8 - Randomized PCA
### 8.3.9 - Incremental PCA

## 8.4 - Kernel PCA
### 8.4.1 - Selecting a Kernal & Tuning Hyperparameters

## 8.5 - LLE

## 8.6 - Other Dimensionality Reduction Techniques



## - Concluding Remarks


[anomaly_detection]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/anomaly_detection.png "illustration of anomaly detection"
