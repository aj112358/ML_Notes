# Ch 9 - Unsupervised Learning Techniques

## Chapter Objectives

1. Learn about clustering - algorithms and theories
    1. Algorithms and theories
    2. Common applications
2. Learn about the k-means clustering algorithm
    1. How to decide on the centroids
    2. Using a performance metric (*inertia*)
    3. Accelerated and mini-batch k-means
    4. Limits of the k-means algorithm
3. Learn how to use clusterin for:
    1. Image segmentation
    2. Preprocessing
    3. Semi-supervised learning
4. Learn about the DBSCAN algorithm
5. Gain familiarity of a few other clustering algorithms
6. Learn about the Gaussian mixture model
    1. Theory
    2. Implementation in SKL
7. Learn how to apply Gaussian mixtures to anomaly detection
    1. Lots of statistical concepts!
8. Become familiar with other algorithms for anomaly/novelty detection

Most applications in ML today (so far!) are based on *supervised* learning - your input features come with associated labels. On the other hand, most data sets come **unlabelled**, and hence we need to implement strategies of **unsupervised** learning for these scenarios.

As an example, suppose there is an assembly line and the output is a fully made product. Your task is to determine if there are any defective items. To do this, you set up an image detection system and thereby get 1000s of pictures of the product every day. You have procured your data set...BUT: These images are unlabelled! Your image detection system is not able to distinguish between good and defected products. 

One solution to this is to manually create a label for each image but this would take up a lot of human resources. You could just label a subset of your data and then look into semi-supervised learning but this would probably not yield a very good ML model. It would be so much easier if there was a ML algorithm that could just classify the images without labels! This is where **unsupervised learning techniques** can help us. 

In this chapter we will look at two common unsupervised learning techniques: clustering and Gaussian mixture models. The former will be studied through the k-means and DSCAN algorithms. We will then use the latter for clustering (as well), anomaly detection, and density estimation. Here are some common use applications for unsupervised learning (*see p.237-238 for more details*):

* Data analysis
* Customer segmentation
* Recommender systems
* Search engines
* Image segmentation
* Semi-supervised learning
* Dimensionality reduction


## 9.1 - Clustering

In ML, **clustering** is the task of identifying similar instances in a data set. Similar instances are grouped together in **clusters**. This is similar to classification except that here we do not know what the classes are upfront (like we do with a supervised classification task). Clustering algorithms can help us with such a problem, and they can **use all the features** to acheive relatively good results!

There is no fixed way to define a cluster - depending on the problem, you may wish to use a different mathematical definition for how an instance is categorized into a cluster. One common way to define clusters is to categorize an instance based on its distance from the clusters. We will look at two ways to create clusters: K-Means and DBSCAN.


## 9.1.1 - K-Means

The **k-means algorithm** is a ML algorithm that is capable of clustering data sets quick and efficiently (usually taking only a few iterations). It was created at the famous Bell Laboratories in 1957 by Stuart Lloyd, and actually published in 1982. Another individual, Edward W. Forgy, published a paper for the same algorithm in 1975, and so this algorithm is also called the "Lloyd-Forgy".

We first illustrate the use of this method in SKL before talking about the underlying theory.



### 9.1.2 - Limits of K-Means
### 9.1.3 - Using Clustering for Image Segmentation
### 9.1.4 - Using Clustering for Preprocessing
### 9.1.5 - Using Clustering for Semi-Supervised Learning
### 9.1.6 - DBSCAN
### 9.1.7 - Other Clustering Algorithms

## 9.2 - Gaussian Mixtures
### 9.2.1 - Anomaly Detection Using Gaussian Mixtures
### 9.2.2 - Selecting the Number of Clusters
### 9.2.3 - Bayesian Gaussian Mixture Models
### 9.2.4 - Other Algorithms for Anomaly and Novelty Detection

## - Concluding Remarks

[anomaly_detection]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/anomaly_detection.png "illustration of anomaly detection"
