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

We first illustrate the use of this method in SKL (using the 'KMeans' class) before talking about the underlying theory.

The k-means ML model's decision boundaries form what is called a **Voronoi diagram**; see the image below:

<insert decision boundary diagram>
    
You can see that at any particular boundary there is some ambiguity in the ML model's prediction, so such points *may* have been mis-assigned to the wrong cluster. Since the k-means algorithm only looks at the distance between an instance and a cluster's centroid, **it does not behave well when the clusters have varying diameters**.

To try and overcome this issues, instead of assigning each instance to a single cluster (called **hard clustering**), we can instead determine a score for each cluster (called **soft clustering**). This **score** can simply be the distance between an instance and all cluster centroids, or you can also compute it as a *similarity score* (using Gaussian RBF perhaps!). In SKL, the 'KMeans' class offers the 'transform()' method can be used to measure the distance of each instance to all clusters.

This method is very useful for dimensionality reduction as it will return a k-dimensional data set (recall, k is the number of clusters). **This can be a very efficient *non-linear* way of DR on a data set!!!**


#### 9.1.1a - The K-Means Algorithm

To appreciate the k-means algorithm, consider the following. If you had a supervised learning task, so that the data set instances came equipped with target labels, you could then easily find the cluster centroids (take the mean of the instances with the same labels). On the other hand, if you somehow already knew the centroid locations, you could easily assign each instance to a cluster (assign to the cluster with the minimum instance-centroid distance).

Now, we **do not have either of these above scenarios**! So it is amazing to think that it is still possible to define cluster *blindly*! The k-means algorithm accomplishes this as follows. We start by randomly defining the k centroid locations (by randomly selection k instances to act as the initial centroids). We then create the clusters by assigning labels to all instances. This will yield our initial Voronoi diagram. Then, for each cluster, we find its centroid and then update the clusters (by *perhaps* re-assigning each instance to a new cluster, depending on its updated distance from the new centroids).

We continue this process until the cluster centroids reach an equilibrium position. Since the average distance between the instances and their *closest* clusters is continually decreasing, this algorithm is guaranteed to converge in a finite (and usually small!) number of iterations.

*Unfortunately*, the algorithm is highly sensitive to the choice of initial selection, and it's possible to get non-unique decision boundaries. This could cause the algorithm to converge to a local optimum, which is not what we want. Let's see how to fix this issue!


#### 9.1.1b - Centroid Initialization Methods

One obvious way to prevent the above issue is to already have an idea of the centroid locations. This can be done by simply trying to visualize the data, or you can run another clustering algorithm first. If you happen to know the centroids' approximate locations, you can include that in the 'KMeans' class instantiation via the 'init' hyperparameter.

Another (obvious?) solution is to simply run the k-means algorithm multiple times, usign a different initialization each time, and keep the "best" solution. The number of random initializations can be set using the 'n_init' hyperparameter (it is 10 by default).

This begs the question: How does SKL know what the "best" solution is? The answer: It uses a performance metric called **inertia**. This is defined to be the mean squared distance between each instance and its closest centroid (ie. it averages all these squared distances). SKL will run the k-means algorithm as many times as you specify (via 'n_init'), measure the inertia of each completed model, and then select the final ML model which has the **lowest inertia**. You can access a k-means model's inertia using the 'inertia_' variable.

Finally, it so happens that an improvement to the original k-means algorithm was made back in 2006 by Arthur & Vassilvitskii, called **K-Means++**. They define the initialization step as one that selects the centroids to be as far from each other as possible. As a result, the k-means algorithm is not as likely to converge to a *local* optimum, and although this requires more computation, the trade-off is for a vast reduction in the number of iterations of the algorithm. 

It so happens that the SKL class 'KMeans' automatically uses the K-Means++ initialization by default.


#### 9.1.1a - Accelerated K-Means & Mini-Batch K-Means

As an aside, we quickly discuss these two variants of the original k-means algorithm.

The first is called **accelerated k-means**, created in 2003 by Charles Elkan. It takes advantage of the well-known **triangle inequality** to avoid unnecessary distance computations and therefore speeds up the training of the k-means algorithm. SKL uses this method by default.

The second is called **mini-batch k-means**, created in 2010 by David Sculley. Here, as with any mini-batch technique, we use only a subset of the data at each iteration, and so the centroids will update only slightly per iteration. This is a good implementation for data sets that do not fit in memory, but you can still always use NumPy's 'memmap' function instead (see Ch 8). With mini-batches, we can use the 'partial_fit()' method but of course this will take more time to train the model. The inertial of mini-batch k-means is usually (slightly) worse than that of regular k-means.


#### 9.1.1a - Finding the Optimal Number of Clusters

We now investigate one obvious question: How can we specify an appropriate number of clusters when initializing the 'KMeans' class? If you can visualize the data, then it is easy, but if not we need to develop a way to determine a good choice for k.

As can be shown, we cannot simply pick the number of clusters based on minimizing inertia. Indeed, by the definition of inertia, with a greater number of clusters you will naturally continue to minimize inertia. Instead, we can use the inertia performance metric in a more educated way as follows.

As a first attempt, we can plot the inertia value for various number of clusters. A typical curve will most definitely have an elbow, and so we can select the value of k corresponding to this elbow. This may not be the optimal number of clusters but it is still a decent choice. Any lesser value of k would have higher inertia, and any higher value of k may cause a cluster to be cut in half for no reason.

As a second and more refined attempt, we can compute the so-called **silhouette score** of a trained k-means model. This is defined as the mean of the **silhouette coefficient** over all instances. This coefficient is simply $(b-a) / max(a,b)$, where:
* a -> mean distance to other instances *in the same cluster*
* b -> mean distance to the instances *in the closest cluster* (not its own cluster)

The silhouette *coefficient* ranges from -1 to +1, with the following interpretations based on the value:
* closer to +1 -> instance is inside its own cluster and far from other clusters
* close to 0 -> instance is close to a cluster boundary
* closer to -1 -> instance has likely been assigned to the wrong cluster

Now, we can use SKL's 'silhouette_score' (from the 'metrics' class) to compute the score. So, we can compute the score for various numbers of clusters and plot the values. Let's go to the JN to see how this is done and also to summarize the above discussions through code.

As a final method of making an educated guess for the optimal number of clusters, we can create what is called a **silhouette diagram**. This is a plot of *each instances* silhouette coefficient, and then sorted by cluster, and then sorted by value (in each cluster). Here is an example:

<insert silhouette diagram>

Each plot depicts a silhouette diagram, which consists of the "knife shaped" regions, one representing each cluster. Some important features are:
* height of each region -> the number of instances in each cluster
* width of each region -> sorted silhouette coefficients of the instances in that cluster (**wider is better!**)
* dashed line -> mean silhouette coefficient (ie. silhouette score) for that number of clusters

When the regions in a certain plot fall short of the dashed line, this means that most of the instances in that cluster have a lower coefficient than the mean score. This indicates that the **cluster is bad**, as most instances are too close to other clusters. In the diagram above, the plots for k=3 and k=6 show we will get bad clusters.

On the other hand, when most instances go beyond the dashed line, this indicates **a good choice for the number of clusters**. We can also look at the heights of each region. A particular number of clusters is **more promising if each cluster region has relatively the same heights**.


### 9.1.2 - Limits of K-Means

Although k-means is a fast and scalable algorithm, it also has its flaws:
* necessary to run the algorithm several times (to avoid sub-optimal solutions)
* need to specify the number of clusters beforehand
* does not behave well with:
    * clusters of varying size
    * clusters of varying density
    * non-spherical shapes

So, you must look at your data and decide if using k-means is a good choice; there are still many other clustering algorithms available for use too. In particular, for *elliptical clusters* the Gaussian Mixture models work pretty well (see below...).

**IMPORTANT: It is important that you scale your data before implementing k-means. This will not guarantee that the clusters will become perfectly spherical, but will still give you better results.**

Regardless of all these limits, the k-means algorithm is still a very useful clustering technique, and so we will briefly look at three applications now.


### 9.1.3 - Using Clustering for Image Segmentation

The first application is with **image segmentation**, which is the task of partitioning an image into multiple segments. From the Wikipedia page, we have the sentance: *"The goal of segmentation is to simplify and/or change the representation of an image into something that is more meaningful and easier to analyze"*. Image segmentation can be further divided into two types.

**Semantic segmentation** is a technique where for every pixel associated to *instances* of the same class, these pixels are assigned to the same overall class. For example, in a photo of people and cats, all the people will become colored the same, and all the dogs will become colored the same.

**Instance segmentation** is a technique where every pixel that is associated with a different instance is assigned to a different class. For example, in the same photo scenario, each person is it's own class and each cat is it's own class.

Nowadays, both these types of image segmentation is done through very complex methods (based on CNNs!) so we will not look into these further. Instead, we will consider a much simpler type of image segmentation called **color segmentation**. Here, a "segment" will consist of all pixels that have a similar color, and we will be in control of how many segments we want to use. An example of this is in analyzing satellite images - if you want to investigate the change in the total amount of forestry present in the image, it may be sufficient to simply implement color segmentation.

To illustrate an example of color segmentation, we finish our discussion in the JN now.


### 9.1.4 - Using Clustering for Preprocessing

As mentioned previously, we can take advantage of clustering techniques when trying to do dimensionality reduction. We will work through an example in the JN now.


### 9.1.5 - Using Clustering for Semi-Supervised Learning

We can also consider using clustering for semi-supervised learning (*Recall: Semi-supervised learning is a technique that uses a few lablelled instances when ultimately working on an unsupervised task with lots of unlabelled instances*). We will work through an example of this in the JN now.


### Active Learning

There are further steps we can take to continue attempting to improve our ML models, one of which is implementing what is called **active learning**. This is when the ML model asks for human input when trying to assign labels to unclassified data.

One of the more common ways to implement active learning is called **uncertainty sampling**, that has the following procedure:
1) Train the ML model on some labelled instances, and use it to make predictions on the unlabelled instances
2) For those predictions that the model is uncertain about, we get a person to label them (and it's now a labelled instance)
3) We repeat the above two steps until *the performance improvements are no longer worth the effort of labelling manually*

Some other active learning strategies are to label those instances such that:
* would result in the largest model change
* would result in the largest drop in model's validation error
* different models disagree on


### 9.1.6 - DBSCAN

**Density-based spatial clustering of applications with noise** or more commonly known as **DBSCAN** is a clustering algortihm that is based on *local density estimation*, and is able to identify clusters of arbitrary shapes. It does this by defining a "cluster" as a **continuous region of high density**. Here is the pseudocode for this algorithm:
1) For each instance, count how many other instances are in it's $\epsilon$-neighborhood.
2) If there are at least 'min_samples' (a hyperparameter) number of instances in the neighborhood, then the instance is called a **core instance**
    * So, a core instance is one that is located in a highly-dense region
3) Assign all instances in the neighborhood to the same cluster
    * Note: This neighborhood *may* include other core instances -> a long sequence of neighboring core instances form a single cluster
4) Any instance that is NOT a core instance NOR has a core instance in its neighborhood is considered an **anomaly**

As you may have inferred, this algorithm works better if you data set has clusters that are spread far apart and are very dense. 

SKL offers us the 'DBSCAN' class for implementation. We go to the JN now to see a simple use of this class using the 'moons' data set.

It may come as a surprise that DBSCAN with SKL does not come equipped with a 'predict()' method - that is, we cannot use the fitted model to make predictions on new input instances. It does have a 'fit_predict()' method though, and this is not too difficult to implement in SKL. Let's see the code in the JN.

In conclusion, DBSCAN is a very powerful clustering algorithm that we can **use to identify clusters of odd shapes**. It can be used to find outliers (hence, anomaly detection!). But, as stated above, if the cluster densities vary, then it may not capture all the clusters properly. The time complexity of DBSCAN happens to be about O(m * log(m)) (which is very close to linear!), but for the implementation in SKL this can go up to O(m^2) if the neighborhood radius is large.


### 9.1.7 - Other Clustering Algorithms

SKL offers us the use of many other clustering algorithms; here is a quick description of some.

* **Agglomerative clustering** - This is an iterative algorithm that builds clusters starting from each individual instance in a 'bottom-up' fashion. At each iteration, the nearest pair of clusters are connected, and the result (mathematically speaking) is a binary tree whose leaves are the individual instances, essentially outputting a "cluster tree". This scales very well to large number of instances/clusters, but requires the use of a **connectivity matrix** which indicates which pairs of instances are neighbors.

* **BIRCH** - An acronym for **Balanced Iterative Reducing and Clustering using Hierarchies**. This is designed to be used for very large data sets. During training, it builds a tree data structure containing just enough information so as to quickly assign new instances to a cluster. Instances do not need to be stored in the tree, and hence this algorithm minimizes the use of memory.

* **Mean-Shift** - This algorithm starts by defning a circle centered at each instance point, and then computes the mean of all interior instances. It then relocates to this mean and repeates theis proceure. Eventually, all circles will reach an equilibrium as they slowly move closer to the local density maximum. Then, all the circles that have settled generally in the same place, their instances are assigned to the same cluster. This algorithm is not well suited for use with large data sets, as the complexity is O(m^2).

* **Affinity Propagation** - This algorithm essentially partitions the data sets into equivalence classes (instances vote for those instances to which they are most similar to - these instances are in the same class/cluster). The time complexity is again O(m^2), so it is not suited for large data sets.

* **Spectral Clustering** - This algorithm takes a similarity matrix between the instances and creates a lower-dimensional embedding from it. It then implements any other clustering algorithm to this reduced data set (SKL implements k-means). This method has the ability to **capture complex cluster structures** and can be used to "cut graphs" (ex: identify clusters of friends on a social network). It does not scale well with the size of data, and also does not behave well with varying-sized clusters.


## 9.2 - Gaussian Mixtures

A **Gaussian mixture model (GMM)** is a probabilistic model that assumes the instances were generated from a mixture of several Gaussian distributions whose parameters are unknown. They can be used for density estimation, clustering, and anomaly detection.

All the instances (that were) generated from a the same Gaussian distribution form a cluster, usually in the shape of an ellipse (or ellipsoid in 3D). Each elliptical cluster can have a different size, shape, density, and orientation. 

So, when you observe an instance, you know that it must have come from one of these Gaussian distributions, but you don't know which one. You also don't have prior knowledge of the parameters of any of these distributions.

Among the several different types of GMMs, the simplest one is the one *where you know the number k of Gaussian distributions beforehand*. SKL implements this variant through the 'GaussianMixture' class.

When choosing to work with such a GMM, you are making the following assumptions about your data set X, and how it was generated through the following probabilistic process:
1. For each instance, a cluster is selected randomly from among k clusters
    * probability of selecting j-th cluster is defined by that cluster's weight, \phi^j
    * index of selected cluster for the i-th instance is z^i
2. Suppose the i-th instance has been assigned to the j-th cluster (denoted: z^i = j)
    * **THUS**: The location x^(i) of this instance is sampled from the j-th Gaussian distribution (denoted: x^(i) ~ norm(\mu-j, \Sigma-j)
    
This is a *generative process*, and can be represented in the following diagram below, which represents the conditional dependencies between the random variables/vectors:

<insert diagram>

The interpretation of this diagram takes a bit to understand, so we leave it to carefully read the descriptions on p.261 of the textbook (*I may add the interpretations in my notes later*). The diagram uses the so-called **plate notation**, which is a way to visualize repeated variables in a graphical model (*see Wikipedia page for more info*). Variables that are known are called **observed variables**, and those which are unknown are called **latent variables**.

---

Now, you may be wondering: What can we use a GMM model for?!

First, given the data set X, we would normally start by trying to estimate the weight vector \phi, as well as trying to estimate each Gaussian's parameters (the mean vectors and covariance matrices). All this can be accomplished using SKL's 'GaussianMixture' class. Let's go to the JN to see how it's done.

You will have noticed that for the three Gaussians their means were 2-vectors. This makes sense because the blobs are 2D and so their locations require two components. 

All these computations by the 'GaussianMixture' class were able to be done because of the **Expectation-Maximization (EM) algorithm**. This is a concept from statistics, and is use with point estimation, namely with finding point estimates of a random sample. It is an algorithm that is guaranteed to converge to the MLE.

We won't go too much in depth of this algorithm as it is a more advanced statistical concept, but we will say it is very similar to the k-means algorithm in the following way:
* It initializes the cluster parameters randomly
* Repeat the following two steps:
    1. Assign instances to clusters (the *expectation step*)
    2. Update each cluster (the *maximization step*)

We can think of the EM algorithm as a generalization of the k-means algorithm, in that it not only finds the cluster's centers (ie. means) but also their size/shape/orientation (via the covariance matrices), as well as each clusters relative weights. **On the other hand**, the EM algorithm implements **soft clustering** - it estimates the probability that an instances belongs to each cluster (whereas k-means implements *hard clustering*). During each maximization step, each custer is updated using **all** instances via their weights (the estimated probability it belongs to a cluster). These probabilities are referred to as the **responsibilities** of the clusters. So, during maximization, each cluster's update will mostly be impacted by the instances it is more responsible for.

**NOTE: In SKL's implementation, the EM algorithm falls to the same trap as the k-means algorithm, where it may not converge to the most optimal solution. Hence, we must run the algorithm multiple times, the number set by the 'n_init' hyperparameter. By default, this is set to only 1!!!**.

Finally, now that we have the clusters defined (location, size, etc.), we can go ahead and assign the instances to the clusters. SKL allows us to use the 'predict()' method for hard clustering, or the 'predict_proba()' for soft clustering. Also, a GMM is an example of a **generative model**, meaning we can actual **sample new instances from it**.

Let's go the JN to see how all this is done.

We can also visualize the decision boundaries that the GMM created, and from the JN they look like this:

<insert plot>

It appears that the GMM found a really good fit to our data! This was expected as we had actually generated the blob data from a Gaussian in the first place! Also, we gave the algorithm the correct number of clusters (since we knew how many there were).

Unfortunately, not all your data sets will be Gaussian, and you won't always know a priori how many clusters there are. This is what **limits the EM algorithm**: if there are too many dimensions, or too many clusters, or few instances, the EM algorithm may not converge to the optimal solution.

To overcome this issue, we can limit the number of parameters (weights, means, covariances) the algorithm needs to learn. One way to achieve this is to limit the range of shapes the clusters can have, and this can be done by prescribing specifications on the entries of the covariance matrices. In SKL, this is done via the 'covariance_type' hyperparameter, which can be set to either "spherical" (clusters must be spherical in shape), "diag" (clusters must have axes parallel to coordinate axes), or "tied" (all clusters are exactly the same shape, size, orientation - ie. they have the exact same covariance matrices), among other options. SKL's default is "full" which essentially means that the clusters have no predefined form.


### 9.2.1 - Anomaly Detection Using Gaussian Mixtures

As you probably already know, **anomaly detection** (aka: **outlier detection**) is the task of detecting instances that deviate greatly from the norm. These instances are called **anomalies/outliers**. Some common uses of anomaly detection:
* fraud detection
* detection of defective products
* removing outliers from a data set before training another model with it
    * can achieve significant improvements in performance!
    
It is very easy to use a GMM for the purposes of anomaly detection: We simply classify an instance to be an anomaly if it is located in a low-density region (so, we must define what density threshold is appropriate for our application).

As an example, suppose you know that the defective rate of products is 4%. You could then set this as your density threshold to start. If you are getting **too many false positives** (good products classified as defective) you can try **lowering** the threshold. If you get **too many false negatives** (defective products that weren't caught), you can try **increasing** the threshold.

We now go to the JN to illustrate a quick implementation of this example.

Related to anomaly detection is the task of **novelty detection**. Here, the GMM algorithm is assumed to have been trained on already clean data (no outliers). This is in contrast to anomaly detection, which does not make this assumption. 

**Note**: GMMs try to fit all the data, including any potential outliers. If there are many of them, this may cause increased bias, as the algorithm assumes normality so some outliers may be considered normal. A remedy to this is to fit the model once, identify any outliers, remore them from the data set, and fit a fresh model on this cleaned data set.


### 9.2.2 - Selecting the Number of Clusters

Just like k-means, GMM required us to specify the number of clusters upon instantiation of the class. So we will take a look at how to do this with GMMs.

Recall that with k-means, we can take advantage of the performance metric called "inertia", as well as plot so-called "silhouette diagrams" to help us make an informed decision as to an appropriate number of clusters to specify. Unfortunately, with GMMs these metrics are **not reliable since the clusters may not be spheical or may have different sizes**. 

For GMMs, we instead use so-called **theoretical information criterion** and try to minimize such criterion. Examples are the **Bayesian information criterion (BIC)** or the **Akaike information criterion (AIC)**. These are defined below:

<img src="http://latex.codecogs.com/svg.latex?BIC=\operatorname{log}(m)p-2\operatorname{log}(\hat{L})" title="Bayesian information criterion (BIC)" />

<img src="http://latex.codecogs.com/svg.latex?AIC=2p-2\operatorname{log}(\hat{L})" title="Akaike information criterion (AIC)" />

where the variables are defined as:
* m -> number of instances
* p -> **number** of parameters learned by the model
* \hat{L} -> maximized value of the **likelihood function** for the model

As can be inferred from the equations, both the BIC and AIC penalize models that have more parameters to learn (ex: more clusters), and reward models that fit the data well. They usually just end up selecting the exact same model, but when they don't the BIC tends to be the simpler model BUT also does not fit the data quite as well as the AIC.

In SKL, we can use the methods 'bic()' and 'aic()' to compute these quantities


### 9.2.3 - Bayesian Gaussian Mixture Models

In the section above we saw ways of determining an appropriate number of clusters to use. It turns out that we can actually find this number automatically using what are called **Bayesian GMMs**. In this model, the cluster parameters (weights, means, covariances) are **not treated as fixed**, but rather as random variables themselves (*indeed this is the foundation of Bayesian statistics!*).

In SKL, we have the 'BayesianGaussianMixture' class that can be used. The input hyperparameter 'n_components' is set to a good educated guess, and it will find the optimal number of clusters for your data set by assigning a weight of (essentially) 0 to any unnecessary clusters. It assigns these weights using somethign called the **stick breaking process**. This is a statistical interpretation of the larger **Dirichlet process**, and this statistical interpretation is a way to compute a multivariate Beta distribution from univariate Beta distributions. *You can read about this on p.270, and also see the Wikipedia article too*.

Let's go to the JN to see how this is done.

With regards to Bayesian statistics, the prior distribution p(z) of the labels z can be taken into account using the 'weight_concentration_prior' hyperparameter of the class, where we can incorporate our prior knowledge of having few or many clusters. We can then use **Bayes' Theorem** to compute the posterior distribution p(z|X), given an observation of some data X. 

Unfortunately, one of the main problems of Bayesian statistics is that of finding the distribution of the **evidence** p(X), and this is also the case with GMMs. This is because this requires integration over all possible value of z (where the integrand is written using the marginal distrubtions, which happens to include the likelihood function). There happen to be many ways to try and work around this problem, one of which is known as **Variational inference**. Here, we select a family of distributions q(z; \lambda) (where \lambda is called the **variational parameters**) and try to find optimal values of \lambda so as to make q(z) a good approximation of p(z|X) (the posterior distribution!). This is done by minimizing the so-called **KL divergence** from q(z) to p(z|X). This is a measure of how different once probability distribution is from another (*see Wikipedia article for more information; "Relative Entropy"*). The equation for the KL divergence is shown below:

<img src="http://latex.codecogs.com/svg.latex?D_{KL}(q||p)=E_q\left[\operatorname{log}\left(\frac{q(\mathbf{z})}{p(\mathbf{z|X})}\right)\right]" title="http://latex.codecogs.com/svg.latex?D_{KL}(q||p)=E_q\left[\operatorname{log}\left(\frac{q(\mathbf{z})}{p(\mathbf{z|X})}\right)\right]" />



















### 9.2.4 - Other Algorithms for Anomaly and Novelty Detection

## - Concluding Remarks

[anomaly_detection]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/anomaly_detection.png "illustration of anomaly detection"
