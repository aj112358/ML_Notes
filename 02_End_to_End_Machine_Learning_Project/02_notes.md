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


[anomaly_detection]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/anomaly_detection.png "illustration of anomaly detection"

[clustering_visualization]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/clustering_visualization.png "clustering visualization"

[machine_learning_automated]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/machine_learning_automated.png "machine learning automated"

[machine_learning_flow_chart]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/machine_learning_flow_chart.png "general machine learning flow chart"

[online_learning_flow_chart]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/online_learning_flowchart.png "online learning flow chart"

[semi_supervised_learning]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/semi_supervised_learning.png "illustration of semi-supervised learning"
