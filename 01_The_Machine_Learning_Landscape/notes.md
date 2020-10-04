# Ch 1 - The Machine Learning Landscape

## Chapter Objectives

1. Define "machine learning"
2. Compare the three different high-level categorizations of ML
3. Learn the basic workflow of a ML project
4. Discuss some challenges of ML
5. Learn how to evaluate and improve a ML model


## 1.1 - What is Machine Learning?

Here are a few ways that one might define "machine learning":

* *Machine learning is the science of programming computers so they can learn from data.*
* *Machine learning is the field of study that gives comptuers the ability to learn without being explicitly programmed* - Arthur Samuel (1959)
* *A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.* - Tom Mitchell (1997)

In essence, "machine learning" is the idea that a computer can become better at performing a task, if given sufficient amounts of data to learn from (what we mean by "learn" will be discussed later). This data may or may not come with an "answer key", and could be given to the computer all at once or continuously throughout some time interval. 

On this note, we can definitively say that simply downloading new data onto your computer is NOT considered ML (your computer has not gotten any better at performing a certain task, right?!). Hence, this concept of "learning" is more nuanced than simply creating/copying new data onto the machine; indeed, we will be seeing various ways a computer may "learn" throughout our study.


## 1.2 - Why Use Machine Learning?

As a thought experiment, let us consider the example of creating an email spam filter. Here are the steps that one would follow if implementing a traditional programming technique:

1. Figure out some common characteristics of spam emails (certain words, phrases, email addresses, etc.)
2. Write a program that, upon detection of these characteristics, classifies an email as spam
3. Do some testing, add other characteristics you observe
4. Launch the program!

As you may have picked up, there are a few problems with this problem solving scheme:

* You may not pick up all possible characteristics of spam emails
* Your program will not work for **new types** of spam emails
* Your program will forever require continuous updates/improvements/maintenance

That sounds pretty annoying to me! In contrast, consider instead the following steps that a ML implementation would take:

1. Gather existing data (your observations of spam vs ham email)
2. Train a ML algorithm to detect spam emails based on this data
3. Evaluate/improve the ML model
4. Launch the model!
5. Feed new data into the ML system (algorithm will improve by itself with new data)

In this problem solving scheme, we do not have to do as much maintanence work since the ML algorithm will continue to become better as new data is fed into it (which is a process that can be automated).

<p align="center">
 <img src="https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/machine_learning_flow_chart.png"
      title = "general machine learning flow chart"/>
</p>

<-- ![alt text][machine_learning_flow_chart] -->

Another field in which implementing ML is useful are those which are too complex (in terms of scalability) or cannot be described via an algorithm. Some examples are: image recognition, speech recognition, online customer support, product recommendations, and many others.

As another use, we can even use ML to learn ourselves! Given a large data set with many characteristics, we may wish to know which few are the most important for predicting something about that data set. A common example that is seen is that of housing prices. We can get data on various characteristics for many different houses on the market, and we wish to know which few characteristics have the most effect on the price of a house. Applying ML techniques to big data can help us discover patterns or trends that are not immediately apparant by the human eye.

Overall, most ML projects are about making predictions (either numerical or categorical). This means that the ML system must be able to make a good prediction on **new** data, once it has been trained on the training data. Having good performance on the training set is good but not sufficient - we need the ML model to perform well on **new** data!

In summary, ML is very useful for:
* Problems for which traditional solutions would require a lot of maintanence and/or a lot of conditional statements
* Complex problems for which the traditional approach would not give a good solution
* Fluctuating environments, where your program would need to continually adapt to new data
* Gaining insight into your data sets, and uncovering unseen patterns/trends


## 1.3 - Example Applications of ML

Here is a list of a few real-world applications of ML, of which we will be learning the techniques used throughout our learning journey:

* Analyzing images of products on a production line to classify them
* Detecting brain tumors
* Flagging offensive/off-topic comments in online forums/chats
* Summarizing long documents
* Forecasting a company's revenue
* Creating voice command features in phones, call centers, etc.
* Detecting credit card fraud
* Representing complex, high-dimensional data in a clear and insightful diagram
* Creating a recommendation system for books, movies, restaurants, etc.
* Creating bots for games, web crawling, etc.


## 1.4 - Types of ML Algorithms

We classify ML algorithms into three broad categories:

* Supervised Learning vs. Unsupervised Learning (Whether or not they are trained with desired results)
* Batch Learning vs. Online Learning (Whether or not they can learn continuously/incrementally)
* Instance-Based Learning vs. Model-Based Learning (Whether they compare new data to old data **or** detect patterns in input data to create a predictive model)

We look at these in turn.


### 1.4.1 - Supervised Learning vs. Unsupervised Learning

We can classify ML algorithms based on the amount and type of human supervision they receive. In general, we have four major subcategories:


#### a. Supervised Learning

Here, the training set you feed your ML algorithm comes equipped with the associated/desired results, which are called "labels". A common example of a supervised learning task is "classification": Given a data set with many features, we wish to classify each sample data point into one or more categories (called a "class"). Another common example is that of "regression": Given a data set, we wish to predict a target numerical value based on the features.

Some important supervised learning algorithms:
* Linear regression
* Logistic regression
* k-Nearest Neighbors
* Decision trees & Random forests
* Support vector machines
* Neural networks


#### b. Unsupervised Learning

Here, the training set that we feed to our ML algorithm does **not** come equipped with the desired output (either classification or numerical value). These types of ML algorithms essentially try to look for patterns in the data set that would allow it to categorize sample points together based on some rules.

As an example, suppose you have some big data about a blog - you may want to classify users into groups based on some characteristics. You'll then be able to target your future blog posts tailored to these individual groups.

Some important unsupervised learning algorithms:
* Clustering
  * k-Means
  * DBSCAN
  * Hierarchical Cluster Analysis (HCA)
* Anomaly detection & novelty detection
  * One-class SVM
  * Isolation forest
* Visualization & dimensionality reduction
  * Principal Component Analysis (PCA)
  * Kernal PCA
  * Locally Linear Embedding (LLE)
  * t-Distributed Stochasting Neighbor Embedding (t-SNE)
* Association rule learning
  * Apriori
  * Eclat
  
  
![alt text][clustering_visualization]
  
![alt text][anomaly_detection]
  

#### c. Semi-Supervised Learning

You may encounter data sets that have a few labelled samples and many unlabelled samples. Having to manually provide labels for a data set is very time-consuming and costly (if the data set is missing labels for some or all data points). In these cases, one can look into using semi-supervised learning algorithms which are combinations of unsupervised algorithms and supervised algorithms.

Some important semi-supervised learning algorithms:
* Deep belief networks (DBN)
  * Based on unsupervised components called "RBM"s
  * RBMs are stacked and trained sequentially in an unsupervised manner
  * The entire stack is fine-tuned using supervised techniques
* Restricted Boltzmann machines (RBM)

![alt text][semi_supervised_learning]

#### d. Reinforcement Learning

This type of learning is very different from the above three. A learning system (called an "agent") is able to observe an environment, perform actions, and in turn get either "rewards" or "penalties". It must then learn on its own to determine what the best strategy is (called a "policy") in order to maximize rewards. A policy defines what action the agent should choose when it is in a given situation.

Some examples of reinforcement learning include:
* Teaching robots how to walk
* DeepMind's AlphaGo


### 1.4.2 - Batch Learning vs. Online Learning

In "batch learning", the ML algorithm is unable to learn in increments, hence all the training data must be used at once. As such, this is slower to execute (can potentially take many hours) and takes a lot of computing resources (CPU, memory, disk space, I/O channels, etc.) and so is performed offline. The algorithm is trained and then immediately launched; this launched ML system does not undergo subsequent learning. This is called "offline learning". If you want this ML system to learn about new data, you will need to re-train an entirely new ML model from scratch (using the old and new data)! Training and creating a new ML model is normally done once every 24 hours or sometimes even once per week. That being said, if the scenario you are trying to model is rapidly changing, you may need a more reactive ML system. Moreover, since training takes a lot of resources, if you happen to be doing it on a cluster then this could end up costing a lot of money (even in opportunity cost). And if your data is truly big data, then it may be impossible to do batch learning!

In "online learning" (aka: "incremental learning"), the ML algorithm is trained in increments i.e. by feeding in the data sequentially, either as individual samples or in small groups (called "mini-batches"). It's easy to see that this type of learning is very fast and cheap, and allows the ML system to learn on-the-fly as new data becomes available. So, even low memory systems can learn from big data! As such, this type of learning is very optimal for systems that receive data in a continuous manner (ex: stock prices, web server logs, time series data, etc.) and hence needs to adapt/improve quickly and independently. Moreover, since the ML system does not need to learn again from old data, the old data may be discarded once used which translates into more available disk space.

An important parameter of online learning systems is their "learning rate" - how frequently they should adapt to new data. A **high** learning rate means more frequent updates but may cause the system to forget old data before it has been processed. A **low** learning rate means less frequent updates but the ML system will also be less sensitive to noise/outliers in new data.

One challenge with online learning: if bad data is fed to the ML system, its performance will decline. To deal with this issue, you would need to monitor the ML system closely and shut it down if you detect a drop in performance. You could also monitor the input data and react to any abnormal data you see (you could use an anomaly detection algorithm if needed!).

![alt text][online_learning_flow_chart]

### 1.4.3 - Instance-Based Learning vs. Model-Based Learning

"Instance-based learning" is probably the most simple way to learn - the system simply memorizes each data point in the training set along with its label. It then deals with new data by using a pre-defined "measure of similarity" to compare the new data to each individual training sample it has memorized.

As an example, consider again the task of classifying spam emails. A very basic measure of similarity would be to compare the total number of spam words between two emails. The reponse would be to flag an email as spam if it has many spam words in common with some email from the training sample.

"Model-based learning" involves using the training set to construct a mathematical model, then use it to make predictions (called "inferences") on new data. If your training set is large and contains mostly good data, then your model will have high prediction accuracy. If it does not make good predictions, then you can try to improve it by using more attributes, creating more informative features, get higher quality data, or simply by trying a different model.


## 1.5 - Main Challenges of ML

### 1.5.1 - Insufficient Quantity of Training Data (Bad Data)

It takes a large amount of data to train a ML model to a high standard. Simple problems may require 1000s of examples; complex problems (image recognition, etc.) may even require millions!


### 1.5.2 - Non-Representative Training Data (Bad Data)

If your data is not a good representation of the real-world situation you are trying to model, then your ML model will not be able to make accurate predictions on new data. It is extremely important that your training data be representative of the new data you are expecting and wish to generalize your model for. This may not be easy in practice: if your training sample is too small you will have "sample noise" (non-representative data as a result of chance). On the other hand, if you have a very large training data set,  this can still be non-representative if your method of data collection is flawed (called "sampling bias").


### 1.5.3 - Poor-Quality Data (Bad Data)

If your training data is full of errors, outliers, noise, etc., then it will difficult for your ML model to detect any underlying patterns and make accurate predictions. Hence, it is very important that you invest time in "cleaning" your data (indeed, the majority of a data scientist's time is spent at this stage!).


### 1.5.4 - Irrelevant Features (Bad Data)

You ML algorithm will only be capable of learning if the training data contains enough **relevant** features and fewer irrelevant features. Hence, a very important indicator of success for a ML model is your ability to come up with a good set of features to train the ML algorithm on. This process is called "feature engineering", and it involves the following general steps:
* Feature Selection: determine the most useful features you should train on
* Feature Extraction: combine information in existing features to create new (hopefully more indicitive!) features
* New Data: create new features by gathering new data


### 1.5.5 - Overfitting the Training Data (Bad Algorithm)

Human nature is prone to overgeneralizing observations of a small sample to the entire overall population. Surprisingly, ML can suffer from this as well and in this context it is called "overfitting". This is when the ML model performs (i.e. makes predictions) very well on the *training data*, but does not perform well or generalize to *new data*. Overfitting occurs when your model is too complex relative to the amount and noisy-ness of the training data.

The effort of trying to reduce overfitting by simplifying the model is called "regularization". It is important to find a balance between fitting the training data perfectly and keeping the model simple enough to ensure it will generalize well. The amount of regularization to apply during learning is controlled by what's known as a "hyperparameter". A hyperparameter is an intrinsic parameter of a ML algorithm, and hence it is not affected by the training of the algorithm - it must be set prior to training AND remains constant during training. The process of deciding on the value of a hyperparameter is called "hyperparameter tuning" which is an important part of the ML process.


### 1.5.6 - Underfitting the Training Data (Bad Algorithm)

"Underfitting" occurs when your ML model is too simple to learn any underlying patterns in the training data.


## 1.6 - Testing and Validating

Finally, once we have trained a ML algorithm to obtain a ML model, we need to evaluate the model's performance and fine-tune it accordingly. The obvious way is to simply deploy the model and see how it performs in your real-world application. The drawback is that your client may need a fully-working model to launch, and is not willing to do any sort of test launch.

The way to overcome this hurdle (i.e. how to test your ML model before/without launching it) is to split your entire data set into two subsets sets, called the "training set" and "test set". You would train a ML algorithm using the training set, and then use the test set and evaluate the model's performance before launch.

A common ratio to split your entire data set into a training set and test set is 80%:20%. In particular, if your data set is very large and contains 10 million sample data points, then even a 1%-sized test set is sufficient as your model will still be tested on 100,000 data points, which is quite substantial.

Upon deployment of your model, it will encounter new data and its error rate on these new data points is called the "generalization error" (aka: "out-of-sample error"). This is the error we are trying to estimate by evaluating the model on the test set - it tells you how well your model will perform on instances of data it has never seen (just like new data).

If the error on the training set is low but the generalization error from the test set is high, it means your model would likely not perform well with new data and is most likely overfitting the data. 


### 1.6.1 - Hyperparameter Tuning & Model Selection

Suppose you have decided what algorithm you want to train. How do we then decide on an appropriate hyperparameter to apply? One way is to create/train multiple different models, each with one possible value of the hyperparameter. For each choice of hyperparameter, you fit your model to the training data and evaluate it on the test set. Unfortunately, you may find that your model produces more error than you expected upon deployment!

This problem occurs because you were using the *exact same test set* to evaluate your model *for each hyperparameter*. In other words, your hyperparameters only work best for *that* test set, and hence the model is not performing well with new data.

A common solution to this problem is called "holdout validation". We take our training set (that we created by splitting our entire data set into a training set and test set) and further split data from *it* - this data that we split out is called the "validation set" (aka: "development set", "dev set"). Hence, we have essentially further reduced our training set by taking out the validation set.

It is on this reduced training set where we would perform the above strategy of creating/training multiple models with various hyperparameters. In this case, we would evaulate each model on the *validation set*. Once we find the best hyperparameter to use, we can then use it to train our final ML model on the *entire* original training set. Finally, as usual, we can then determing the generalization error by using the test set.

This solution works well in practice but has its own flaws. If the validation set is too small, then your model hyperparameter evaluations will be imprecise (you may select a sub-optimal hyperparameter by mistake). If the validation set is too large, then the reduced training set must be very small. In this case, it is not ideal to compare candidate hyperparameter models on such a small training set with the model that will be trained on the (much larger overall) full training set.

One solution to this problem is to repeatedly perform a process called "cross-validation". This is where you split your training set into a reduced training set and multiple validation sets that are each small. Each hyperparameter model would then be evaluated once for each small validation set, and then you take the average of each model's evaluations. This provides a more accurate measure of that particular hyperparameter model's performance. The drawback to cross-validation is that the time complexity scales with the number of validation sets.


### 1.6.2 - Data Mismatch

It is possible that you may have access to big data for training, but this data is not actually representative of that which will be seen when your model is launched. As an illustration, consider the following example.

You wish to make a ML model to classify pictures of flowers taken on users' phones. To train this model, you could easily download a very large set of flower pictures from the internet to be used as the data set. The downside is these pictures may not be representative of what a picture taken on a phone looks like.

Now, suppose that you still choose to download these pictures from the internet, and you were also able to collect pictures taken on phones. In this case, it is very important to **remember that the test set and validation set must contain data that is closely representative of the data that would be seen upon launch**. As such, you decide to use all your phone pictures for these two sets (perhaps split evenly between the two) and decide to use the internet pictures as the training set.

Then, you continue through the ML process and use the internet pictures to train your model. Upon evaluating using the validation set, you may find its performance to be very low. This may be for two reasons: (1) your model may be overfitting to the training set, or (2) the internet pictures are not representative of the phone pictures (there is a **mismatch** between them).

To determine which is the case, you can split your training data (of internet pictures, which you haven't done throughout this example yet) into a reduced training set and a so-called "train-dev set" (coined by Andrew Ng). You then train your model on the reduced training set, and evaluate it's performance on the train-dev set (keep in mind both sets only contain internet pictures). Then there would be two cases based on how the model performs on the train-dev set.

First, if your model performs well on the train-dev set, this indicates that overfitting is NOT happening, which solves issue (1) above. If that same model performs poorly on the validation set, then you know it is due to a mismatch of data. Second, if the model performs poorly on the train-dev set, this indicates overfitting, which means you can look into regularization, try to get more data, perform data cleaning, etc.

In general, the problem of data mismatch can be handled by performing "data preprocessing". In this case, that would mean taking the internet pictures and manipulating them to become more similar to the phone pictures (adjust size, color, etc.).

[//]: # (https://blogs.perficient.com/2018/01/29/machine-learning-vs-statistical-learning/#:~:text=Both%20methods%20are%20data%20dependent.&text=Statistical%20Learning%20is%20math%20intensive,way%20less%20of%20human%20effort.
)

[//]: # (https://towardsdatascience.com/the-actual-difference-between-statistics-and-machine-learning-64b49f07ea3)

- ML algorithm = the name of the specific math model/formula you want to train (linear, svm, decision tree, etc)
- ML model = the result of finalizing training (fully trained and tested model)
- ML system = what you get once you deploy your model to the real-world application you're working on
- ML project = the process of doing the above three


[anomaly_detection]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/anomaly_detection.png "illustration of anomaly detection"

[clustering_visualization]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/clustering_visualization.png "clustering visualization"

[machine_learning_automated]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/machine_learning_automated.png "machine learning automated"

[machine_learning_flow_chart]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/machine_learning_flow_chart.png "general machine learning flow chart"

[online_learning_flow_chart]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/online_learning_flowchart.png "online learning flow chart"

[semi_supervised_learning]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/semi_supervised_learning.png "illustration of semi-supervised learning"
