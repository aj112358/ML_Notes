https://blogs.perficient.com/2018/01/29/machine-learning-vs-statistical-learning/#:~:text=Both%20methods%20are%20data%20dependent.&text=Statistical%20Learning%20is%20math%20intensive,way%20less%20of%20human%20effort.


https://towardsdatascience.com/the-actual-difference-between-statistics-and-machine-learning-64b49f07ea3



### Ch 1 - The Machine Learning Landscape

* ML has already been around for decades! It has only recently become popular again.
* The first 'mainstream' application of ML was the beloved spam filter, created in the 1990s

## Chapter Objectives

1. Define "machine learning"
2. Compare the three different high-level categorizations of ML
3. Learn basic workflow of a ML project
4. Discuss some challenges of ML
5. Learn how to evaluate and improve a ML model


## 1.1 What is Machine Learning?

Here are a few ways that one might define "machine learning":

* *Machine learning is the science of programming computers so they can learn from data.*
* *Machine learning is the field of study that gives comptuers the ability to learn without being explicitly programmed* - Arthur Samuel (1959)
* *A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.* - Tom Mitchell (1997)

In essence, "machine learning" is the idea that a computer can become better at performing a task, if given sufficient amounts of data to "learn" from (what we mean by "learn" will be discussed later). This data may or may not come with an "answer key", and could be given to the computer all at once or continuously throughout some time interval. 

On this note, we can definitively say that simply downloading new data onto your computer is NOT considered ML (your computer has not gotten any better at performing a certain task, right!). Hence, this concept of "learning" is more nuanced than simply creating/copying new data onto the machine; indeed, we will be seeing various ways a computer may "learn" throughout our study.

## 1.2 Why Use Machine Learning?

As a thought experiment, let us consider the example of creating an email spam filter. Here are the steps that one would follow if implementing a traditional programming technique:

1. Figure out some common characteristics of spam emails (certain words, phrases, email addresses, etc.)
2. Write a program that, upon detection of these characteristics, classifies an email as spam
3. Do some testing, add other characteristics you observe
4. Launch the program!

As you may have picked up, there are a few problems with this problem solving scheme:

* You may not pick up all possible characteristics of spam emails
* Your program will not work for **new types** of spam emails
* Your program will require continuous updates/improvements/maintenance forever

That sounds pretty annoying to me! In contrast, consider instead the following steps that a ML implementation would take:

1. Gather existing data (your observations of spam vs ham email)
2. Train a ML algorithm to detect spam emails based on this data
3. Evaluate/improve the model -> Launch!
4. Feed new data into the ML algorithm (algorithm will improve by itself with new data)

In this problem solving scheme, we do not have to do as much maintanence work since the ML algorithm will continue to become better as new data is fed into it (which is a process that can be automated).

Another field in which implementing ML is useful are those which are too complex (in terms of scalability) or cannot be described via an algorithm. Some examples are: image recognition, speech recognition, online customer support, product recommendations, etc.


As a third use, we can even use ML to learn ourselves! Given a large data set with many characteristics, we may wish to know which 2-3 are the most important for predicting something about that data set. A common example that is seen is that of housing prices. We can get data on various characteristics on many different houses on the market, and we wish to know which few characteristics have the most effect on the price of a house. Applying ML techniques to big data can help us discover patterns or trends that are not immediately apparant.

Overall, most ML tasks are about making predictions (either numerical or categorical). This means that the ML system must be able to make a good prediction on NEW data, once it has been trained on the training data. Having good performance on the training set is good but not sufficient - we need the ML model to perform well on NEW data!

In summary, ML is very useful for:
* Problems for which traditional solutions would require a lot of maintanence and/or a lot of conditional statements
* Complex problems for which the traditional approach would not give a good solution
* Fluctuating environments, where your program would need to continually adapt to new data
* Gaining insight into your data sets

## 1.3 - Example Applications of ML

Here is a list of a few real-world applications of ML, of which we will be learning the techniques throughout our learning journey:

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


## 1.4 - Types of ML Systems

We classify ML systems into three broad categories:

* Supervised Learning vs. Unsupervised Learning (Whether or not they are trained with human supervision)
* Batch Learning vs. Online Learning (Whether or not they can learn continuously)
* Instance-Based Learning vs. Model-Based Learning (Whether they compare new data to old data OR detect patterns in input data and creating a predictive model)

We look at these in turn.

# 1.4.1 - Supervised Learning vs. Unsupervised Learning


We can classify ML systems based on the amount and type of human supervision they receive. In general, we have four major subcategories:

1. Supervised Learning

Here, the training set you feed your ML algorithm comes attached with the desired classification, which are called "labels".

A common example of a supervised learning task if "classification". Given a data set with many features, we wish to classify each sample point into one or more categories (called a "class").

Another common example is that of "regression". Given a data set, we wish to predict a target numerical value based on the features.


Some important supervised learning algorithms:
* Linear regression
* Logistic regression
* k-Nearest Neighbors
* Decision trees & Random forests
* Support vector machines
* Neural networks



2. Unsupervised Learning

Here, the training set that we feed to our ML model does NOT come attached with the desired output (either classification or numerical value). The ML system essentially tries to look for patterns in the data set that would allow it to categorize sample points together based on some rules.

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
  



3. Semi-Supervised Learning

Having to manually provide labels for a data set it very time-consuming and constly. As such, you may encounter data sets that have a few labelled samples and many unlabelled samples. Hence, mose semi-supervised learning algorithms are combinations of unsupervised algorithms and supervised algorithms.

Some important semi-supervised learning algorithms:
* Deep belief networks (DBN)
  * Based on unsupervised components called RBMs
  * RBMs are stacked and trained sequentially in an unsupervised manner
  * The entire stack is fine-tuned using supervised techniques
* Restricted Boltzmann machines (RBM)


4. Reinforcement Learning

This type of learning is very different from the above three. A learning system (called an "agent") is able to observe and environment, perform actions, and in turn get "rewards" or "penalties". It must then learn on its own to determine what the best strategy is (called a "policy") in order to maximize rewards. A policy defines what action the agent should choose when it is in a given situation.

Some examples of reinforcement learning include:
* Teaching robots how to walk
* DeepMind's AlphaGo



# 1.4.2 - Batch Learning vs. Online Learning


In "batch learning", the ML system is unable to learn in increments, hence all the training data must be used at once. As such, this is slower to execute (can potentially take many hours) and takes a lot of computing resources (CPU, memory, disk space, I/O, etc.) and so is performed offline. The system is trained and then immediately launched; it does not learn anymore. This is called "offline learning". If you want this ML model to learn about new data, you will need to re-train en entirely new model from scratch (using the old and new data)! Training a new system is normally done once every 24 hours or sometimes even once per week. That being said, if the scenario you are trying to model is rapidly changing, you may need a more reactive ML system. Moreover, since training takes a lot of resources, if you happen to be doing it on a cluster then this could end up costing a lof of money (even in opportunity cost). And if your data is truly big data, then it may be impossible to do batch learning!


In "online learning" (read: "incremental learning"), the ML system is trained in increments i.e. by feeding in the data sequentially, either individual samples or in small groups (called "mini-batches"). It's easy to see that this type of learning is very fast and cheap, and allows the ML system to learn on-the-fly as new data becomes available. So, even low memory systems can learn from big data! As such, this type of learning is very optimal for systems that receive data in a continuous manner (ex: stock prices, web server logs, time series data, etc.) and hence needs to adapt/improve quickly and independently. Moreover, since the ML system does not need to learn again from old data, the old data may be discarded once used which translates into more available disk space.


An important parameter of online learning systems is their "learning rate" - how frequently they should adapt to new data. A **high** learning rate means more frequent updates but may cause the system to forget old data before it has been processed. A **low** learning rate means less frequent updates but the ML system will also be less sensitive to noise/outliers in new data.

One challenge with online learning: if bad data is fed to the ML system, its performance will decline. To deal with this issue, you would need to monitor the ML system closely and shut it down if you detect a drop in performance. You could also monitor the input data and react to any abnormal data you see (you could use an anomaly detection algorithm if needed!).

# 1.4.3 - Instance-Based Learning vs. Model-Based Learning


"Instance-based learning" is probably the most simple way to learn - the system simply memorizes each data point in the training set along with its label. It then deals with new data by using a "measure of similarity" to compare the new data to each individual training sample it has memorized.

As an example, consider the task of classifying spam emails. A very basic measure of similarity would be to compare the total number of spam words in the new email with that in all training set emails. The reponse would be to flag an email as spam if it has many spam words in common with some training sample email.



"Model-based learning" involves using the training set to construct a mathematical model, then use it to make predictions (called "inferences") on new data. If your training set is large and contains mostly good data, then your model will have high prediction accuracy. If it does not make good predictions, then you can try to improve it by using more attributes, creating more informative features, get higher quality data, or try using a different model.




## Main Challenges of ML

# Insufficient Quantity of Training Data


# Non-Representative Training Data
# Poor-Quality Data
# Irrelevant Features
# Overfitting the Training Data
# Underfitting the Training Data






## Testing and Validating

# Hyperparameter Tuning & Model Selection
# Data Mismatch













