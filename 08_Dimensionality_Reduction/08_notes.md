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

It is difficult to envision any object that exists in a world with greater than three dimensions - our human intuition only works up to three dimensions. It so happens that many intersting things can happen in higher-dimensional worlds:

* In a 10,000-dimensional unit hypercube, the probability that a point will be close to any particular axis is essentially 1 (compared to only 0.001 in a unit square!)
* Two points in a 1,000,000-dimensional unit hypercube will have average distance ~408.25 between them (compared to only 0.66 in a unit cube!)

The second point above begs the question of *how can there be so much space in a UNIT hypercube in 1,000,000 dimensional space?!* It turns out that there is a LOT more space in a unit hypercube as you continue to go up in dimensions.

Coming back to data science, we can now infer that a high-dimensional data set will be **very sparse** - there will be a lot of "distance" between each training instance. This will result in making it harder to make predictions on new data, as the new data has the possibility of being even further away from the training set. Thus, **the more dimensions the training set has, the greater the risk of overfitting it**.

One (obvious?) solution to the curse of dimensionality is to ("simply") increase the number of training instances, so that there are a sufficient amount to take care of all the features. One (obvious!) problem with this is that it may not be possible to get more training data. But even if so, a bigger problem is that the number you would need to reach a given density (for the number of features) grows exponentially with the dimension. (*ex: Even with only 100 features (vastly less than even MNIST) you would need more training instances than observable atoms in the universe for them to be within distance 0.1 from each other).*

So, now that we understand the problems of working in higher dimensions, let's begin our learning of dimensionality reduction.


## 8.2 - Two Main Approaches for Dimensionality Reduction

To stat, we will look at the two main approaches used for dimensionality reduction.


### 8.2.1 - Projection

As you may have inferred throughout this course, for more real-world data problems, the training instances are never spread uniformly across all dimensions. Some features may be constant, while others may have high correlations with one another. Because of this, although the training instances live in a more general high-dimensional space, we can simply consider them to be existing a lower-dimensional **subspace**.

As an example, consider the visual below:

<insert 3D image>

We can see that the blue data points are very close to the gray shaded plane. We can perform an orthogonal projection onto this plane and obtain the new (but related) image below:

<insert subspace image>

By doing this, we have taken our original data (which lives in 3D space) and instead projected it onto the 2D plane (which is the subspace), thereby reducing the dimension of the data! The new features z_1 & z_2 are related to the original features x_1, x_2, x_3.

Unfortunately, this technique does not work for all data sets. As an example, consider the so-called **"Swiss roll"** toy data set below:

<insert swiss roll>

We can see that if we project to the x_1,x_2-plane, the resulting 2D data set will have lots of points with different target values overlapping (ie. it looks like a mess). On the other hand, if we were to somehow "unroll" the original data set, we would obtain a data set that is 2D and preserves the relations between each point! We can see this in the figure below:

<insert 2D swiss roll figure>

Hence, we have to be careful when deciding whether to use projection!


### 8.2.2 - Manifold Learning

The Swiss roll data set above is an example of a 2D *manifold*. A **$d$-dimensional manifold** is a part of an $n$-dimensional space (with d<n) that *locally* resembles a $d$-dimensional hyperplane. More simply, a 2D manifold is a 3D shape that can be bent/twisted within a higher-dimensional space.

**Note:** From Wikipedia *a manifold is a topological space that locally resembles Euclidean space near each point*.

Many of the dimensionality reduction techniques that exist work by **modelling the manifold on which the training instances lie**. This is called **manifold learning**, and it relies on the so-called **manifold assumption** which is that **most real-world high-dimensional data sets lie close to a much lower-dimensional manifold** (which is often what is observed).

The manifold assumption is often accompanied by another assumption: *the ML problem will be simpler if expressed in the lower-dimensional space of the manifold*. As it turns out, this actually *depends on the data set*.

Consider the figure below:

<insert figure>
    
In the top two diagrams, we see that the data set in the original space (left-hand plot) would have a complicated decision boundary, but the same data in the manifold space (right-hand plot) has a very simple decision boundary. In the lower two diagrams, we see the opposite: the data set in the original space has a much more simple decision boundary, but the data in the manifold space has a more complicated decision boundary (consisting of four different lines).

The takeaway is that implementing dimensionality reduction techniques will indeed make your training execute faster, but the resulting ML model **may or may not** perform better. Even more so, reducing the dimension may not even lead to a simpler solution, as the above figure shows! This is why **it is important to train on the original data set first, and THEN think about dimensionality reduction techniques after**.

Now that we have knowledge of the two main techniques of DR, we will look into more specific methods, learn their underlying theory, and how to implement them in SKL.


## 8.3 - Principal Component Analysis (PCA)

**Principal component analysis (PCA)** is the most popular DR method. The general procedure is to define a hyperplane, and then project all the data points onto this (lower dimensional) hyperplane. We look into the detail of this technique now.


### 8.3.1 - Preserving the Variance

The first question to ask is how do we choose an appropriate hyperplane? The guiding principle behind this question is to **choose the hyperplane that will preserve the (maximum) variance of the original data set**. This is a reasonable idea as we will then be *minimizing the amount of information that is lost from the projection*.

Also, it so happens that choosing such a hyperplane will also have the minimal mean-squared distance between the original data and the projected data.


### 8.3.2 - Principle Components

The method of PCA finds the aforementioned axis that will preserve the maximum variance. It will also find a second axis that accounts for the **largest *remaining* variance**, and this second axis would be orthogonal to the first one. Depending on the dimensionality, PCA will actually find an axis for each feature, each one taking into account the remaining variance from the previous ones. Each axis is called a **principal component**, and we order them as 1st, 2nd, etc.

To mathematically define each principal compenent, PCA will fina a **zero-centered unit vector** for that axis. These vectors are highly sensitive to small changes in the data set, but will generally produce the same hyperplane. To find these vectors, we simply compute the **singular value decomposition (SVD)** of the training set matrix, and the principal components are contained within the matrix V as its columns.

As an illustration, we can use NumPy to compute the principal components for us. In this case, we must first manually zero-center the data (ie. each feature column) first, as **PCA expects the data set to be centered about the origin**. Later when we use SKL, this will be done for us automatically. We go to the JN now.


### 8.3.3 - Projecting Down to d Dimensions (ie. Onto a Hyperplane)

Now that you have computed all n pricipal components (where n is the number of features), you now need to decide what d dimensional hyperplane you wish to project onto, with d<n representing the first d principal components (so as to preserve as much variance as possible). 

Once you have chosen what dimension to project onto (see later below), you can then simply project your data set onto that hyperplane by computing the matrix product of (i) the matrix of training instances, and (ii) a matrix with the desired d principal components as its column. We illustrate this quickly in the JN.


### 8.3.4 - Using SKL

All the above can be automated using SKL's "PCA" class. This will compute the SVD automatically (and will center the data itself beforehand), and will output the projected data. It also contains the principal components for our use, among other things. If you need the principal components, we can use the 'components_' method. We illustrate this in the JN now.


### 8.3.5 - Explained Variance Ratio

A useful metric is the so-called **explained variance ratio** of each principal component. This ratio **indicates the proportion of the dataset's variance that *lies along each principal component***. This information is found in the 'explained_variance_ratio_' variable.

As an example, from the JN we see that we have an output of [0.84248607, 0.14631839]. This indicates that ~84.2% of the variance of the original data lies on the first principal component, and ~14.6% of the variance lies on the second. This means that the third (and final) principal component contains <1.2% of the variance, and hence probably does not contain too much information about the original data set.


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
