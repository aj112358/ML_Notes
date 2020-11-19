# Ch 7 - Ensemble Learning and Random Forests

## Chapter Objectives

1. Learn various popular ensemble methods
    1. Bagging
    2. Boosting
    3. Stacking
2. Learn why ensemble methods work
    1. Law of large numbers
3. Learn about bagging and pasting
    1. Compare and contrast
    2. Out-of-bag evaluation
4. Learn about random patches and random subspaces
5. Learn about random forests
    1. Implementation in SKL
6. Learn about boosting
    1. AdaBoost
    2. Gradient Boost
7. Learn about stacking

Websites like Stack Exchange, Stack Overflow, Wikipedia, Quora, and any other online forum provide you with multiple opinons/answers from different people, and (usually) these answers are better than what you would find when looking at a single source or asking a single person. This phenomenon is called *wisdom of the crowd* and we can apply it to create a new ML algorithm.

We have seen many different ML algorithms so far including linear regression, polynomial regression, logistic regression, SVM, and decision trees. Mirroring the above idea, we could make predictors using many different algorithms, and instead of using a single one we can aggregate each of their predictions. This will usually lead to a better prediction than you would get from each algorithm individually.

A group of predictors is called an **ensemble**, and the above technique is hence called **ensemble learning**. A particular ensemble learning ML algorithm is termed an **ensemble method**.

A particular example of an ensemble method is called a **random forest**. Here we train a group of decision trees individually on different random subsets of the training data. Then to make predictions, we get the predictions of each individual decision tree and finally predict the majority class. As simple as this sounds, a random forest model is one of the most powerful ML algorithms available!

In practice, an ensemble method is used at the end of a ML project. At this point you will have already created a few good predictors, and now you can simply combine them into a better predictor. 


## 7.1 - Voting Classifiers

We start learning about ensemble methods with a simple example involving a generic classification task.

Suppose you are given a classification ML problem, and have went ahead and worked through the ML process (as we have outlined at the beginning of this course), and now have developed various classification algorithms, say logistic classifier, SVM classifier, KNN classifier, and a random forest classifier. Each of these yield approximately 80% accuracy.

We can then use all these classifiers **at once** to make a prediction on a new input instance. We would pass this input instance through each individual classifier, get the predicted class from each, and then specify the prediction to be that class that was predicted most often by our group of classifiers. This type of ensemble classifier is called a **hard voting classifier**.

It turns out that such a hard voting classifier will often achieve a higher accuracy than even the best individual classifier in your ensemble. In the extreme case, if each of the individiual classifiers are **weak learners** (they only do slightly better than random guessing), the entire ensemble can still come out to be a **strong learner** (achieving high accuracy). This caveat is that there should be a sufficient number of weak laerners AND that they are diverse in nature.

The reason for this behaviour is explained by the statistical concept called the *law of large numbers*. As an illustration of this law, suppose we have built an ensemble consisting of 1000 classifiers, each with accuracy 51% (ie. weak learners). Using the law of large numbers and some basic probability theory, you can show that the probability of each classifer predicting the majority is approximately 75%. 

Although, we must be careful of our assumptions. The above illustration assumes that the ensemble of classifiers are mutually independent, which we know is false as each is trained on the same training set. Hence, they are capable of making the same errors, and so the true perceived accuracy would be less than the statistical ideal of 75%. The lesson learned is that **ensemble methods work best when the predictors (ie. individual classifiers) are as independent from one another as possible.** One way to accomplish this is to *train each predictor using very different algorithms*.

To illustrate the implementation of an ensemble method, we will create a classifier that implements hard voting. We can use SKL's 'VotingClassifier' class for this. We go to the JN now.

Finally, in an attempt to increase the ensemble accuracy even further, we can try an ensemble method called **soft voting**. Here, we proceed in a similar way - create various classifiers individually, and pass a new input instance to each to get their individual predictions.

NOW: Instead of doing a majority vote (hard voting), we would instead compute the class probabilities for each individual classifier, take the average of the probability for each class over the individual classifiers, and make the prediction for the class with the highest average.

In order to do this, each of the individual classifiers need to be equipped with the 'predict_proba()' method. As we know, the SVC does not come with this method by default, so for that particular classifier, we must get it to implement cross-validation by setting the hyperparameter 'probability' to "True". Although this will slow down training, you will be able to then compute class probabilities.

Finally, to implement soft voting, we simply specify the hyperparameter 'voting' to "soft", in the VotingClassifier class instantiation. We do this now in the JN.


## 7.2 - Bagging and Pasting

As we have mentioned above, in order for an ensemble method to have maximized accuracy, it is important to **use a diverse set of classifiers**. One way to do this is to simply use different training algorithms for each individual classifier. But another approach is to **use the *same training algorithm* but train each on *different random subsets* of the training set**.

To get these random subsets of the training data, we have two ways. **Bagging** (aka: "bootstrap aggregating") is a term that means sampling *with* replacement *(In statistics, sampling with replacement is called **bootstrapping**)*. **Pasting** is a term that means sampling *without* replacement. We can use one or the other to sample training instances for each individual predictor we choose to use. With _bagging_, the sampling with replacement is done on the *same predictor*. This means that, for a given predictor, it is possible that it randomly chooses repeated training instances to train on. This does **not mean** that *for each classifier*, the training subsets used are disjoint - each classifier starts its training with the *entire* training set. (*It is not like the first classifier uses some subset of the training set, and then passes on the remaining training instances to the next classifier*).

Once you have trained all the individual predictors, the ensemble makes predictions by using some type of **statistical aggregate function**. For classification, the mode is the most commonly used. For regression, the average is more commonly used.

Each individual predictor has a higher bias (than if it were trained on the *entire* training set), but the effects of aggregation are to reduce both the bias and variance. The ensemble then generally has a similar bias but a lower variance, than any single predictor that would have been trained on the entire training data.

One of the reasons that bagging and pasting are popular methods is that **they have good scalability**. Both training and making predictions can be done in parallel (different CPU cores, different servers, etc)


### 7.2.1 - Implementing Bagging and Pasting in SKL

We can implement both bagging and pasting in SKL using the classes 'BaggingClassifier' or 'BaggingRegressor' (both can perform bagging and pasting). We go to the JN now to see how this can be done.

When using either of these classes, we can specify the hyperparameter 'bootstrap' to be "True" if we want to implement bagging, and "False" if we want to implement pasting. With these classes we must specify which 'base_estimator' we want to use, and then we can specify the number we wish to train (via 'n_estimators') as well as the size of each random sample for each to train on (via 'max_samples').

Also, with the 'BaggingClassifier' class, it will **automatically perform *soft voting* if the base estimator is able to provide class probabilities**. Recall, for example, that this is the case for decision trees but not for SVMs.

From the JN code, we make the following two plots to compare bagging with using a single decision tree:

<insert plots here>

We can see that the single decision tree's decision boundary is very complex, whereas the bagging ensemble's decision boundary is more smooth. It's easy to see then that the bagging ensemble model will better generalize to new data. Also, as we discussed above, the ensemble will have a similar bias as the decision tree but will have smaller variance.

As is so happens, sampling *with* replacement (ie. bagging; bootstrapping) introdcues more diversity into the training subsets for each predictor, and so bagging actually will have a slightly higher bias than pasting. But this added diversity causes each predictor in the ensemble to be less correlated to the others, hence reducing the variance. **Overall, bagging often results in better models**.

**Remark**: Of course, you could use cross-validation to evaluate both bagging and pasting to make a better informed decision!


### 7.2.2 - Out-of-Bag Evaluation

For bagging, we know that training instances are chosen randomly to create the training subset for each predictor. Since we are performing sampling *with* replacement, it is possible that many instances are sampled multiple times, but there will surely be some training instances that are **not sampled**. It can be shown that ~63% of the training instances are sampled on average, for each predictor in the ensemble. The other ~37% of training instances that are not sampled are called **out-of-bag (oob) instances**. Of course, these need not be the same for each predictor!

So, during training the predictor does not have knowledge of these oob instances. As such, we can actually use them as an evaluation tool for our predictor! They essentially take the place of a validation set. We can then evaluate the ensemble by simply taking the average of these so-called **out-of-bag evaluations**. 

Let's see how we can implement oob evaluation with bagging in the JN.


## 7.3 - Random Patches and Random Subspaces

So far, we have discussed random samples of the training *instances*, but we can also do random sampling of the **features** as well. To do this, the 'BaggingClassifier' class allows us to specify the two hyperparameters 'max_features' and 'bootstrap_features'. The latter hyperparameter allows use to specify whether we wish to sample the features *with* replacement ("True") or *without* replacement ("False"). 

Sampling from both the training instances and training features is referred to as the **random patches method**.

If you choose to **use all training instances** (by setting 'boostrap' to "False, and 'max_samples' to "1.0") and simultaneously choose to **do feature sampling** (by setting 'bootstrap_features' to "True" and/or 'max_features' <1.0), this method is called the **random subspaces method**.

Implementing feature sampling is particularly useful if you are dealing with high-dimensional data, such as images. It can also provide you with greater diversity among your predictors in the ensemble, hence trading a little bias for a lower variance. In the next chapter of the book, we will look into the problems that high-dimensional data can cause to us as data scientists, and how to go about alleviating some of these problems.


## 7.4 - Random Forests

A **random forest** is simply an ensemble of decision trees, trained in the following way:
* via the bagging method (generally)
* 'max_samples' = 1.0 (ie. the entire training set)

Of course, we can simply use the 'BaggingClassifier' class and use a decision tree as the 'base_estimator'. But SKL also offers us the 'RandomForestClassifier' class for our use (there is also 'RandomForestRegressor') We quickly illustrate the use of this class in the JN.

The random forest algorithm introduces more randomness when growing trees by searching for the best feature *among a random subset of features* when splitting a node. With more diversity, it is able to trade more bias for smaller variance giving you better overall results.


### 7.4.1 - Extra-Trees

It is possible to go even further beyond and introduce more randomness into a random forest by *using random thresholds* for each feature itself, alongside the aforementioned method of using random features.

This type of random forest is fittingly called an **extremely randomized trees ensemble**, or simply **extra-trees** for short. Using extra-trees allows for faster training than a regular random forest, since we don't need to find *the best* possible threshold for each node.

SKL offers us the 'ExtraTreesClassifier' and 'ExtraTreesRegressor' classes to implement extra-trees into our ML problem. They the the exact same hyperparameters as both respective 'RandomForest' classes.


### 7.4.2 - Feature Importance

In any ML problem, it is very useful to have knowledge of which features are more important than others, especially if you are dealing with high-dimensional data. Random forests provide us with such information. To do this, SKL computes a weighted average over the nodes where the weights for a node are simply the number of training samples that correspond to it (the 'samples' attribute). It then scales all the scores so they sum to one. To access these feature importance scores, they are stored in the 'feature_importances_' variable.

We go to the JN to see an example, this time using the Iris data set.

**Random forests are an excellent and easy way to quickly determine which features are the important ones. This can help greatly during the EDA step of the ML process!**


## 7.5 - Boosting

The term **boosting** (originally called: "hypothesis boosting*) refers to any ensemble method that combines weak learners (slightly better than random) into a strong learner, and does this by **training each predictor sequentially, where subsequent predictors learn from previous ones and try to correct them**.

The two types of boosting methods we will personally look at are "AdaBoost" and "Gradient Boosting", although there are also many others available.


### 7.5.1 - AdaBoost

One popular boosting method is called **AdaBoost** (short for: **Adaptive Boosting**). In order for it to continue improving each successive predictor, it **uses the previous predictor's training instances that were *underfitted***. Hence, each subsequent predictor will focus moreso on the harder-to-identify training instances (from the previous predictor).

As a specific example, you first start with a base classifier and train that normally. Then you use the now trained ML model and evaluate it by making predictions on the *training set*. For any training instance that was *misclassified*, we increase that instances weight (relative to the other training instances). We then move on and train the second predictor on the training set, *now updated with different weights*. Once trained, we use this second predictor on the training set and check for misclassifications, and change their weights accordingly. This process is repeated for each predictor sequentially for the entire ensemble.

Once the entire ensemble has been created, **each now-trained predictor is itself assigned a relative weight** compared to all others based on its overall accuracy on the weighted training set.

This method is similar to the method of gradient descent we have seen previously. In this case, we are continually adding predictors to our ensemble to make it better overall, instead of working with only a single predictor and minimizing a cost function.

**Remark:** A very clear downside of sequential learning is that you *cannot parallelize the learning*. Each predictor must wait until the previous one is fully trained.


#### Mathematics of AdaBoost

We start by setting each instance's initial weight to $1/m$, where $m$ is the number of instances. Once a predictor is trained, we can compute its weighted error (when applied to the training set) via:

<img src="http://latex.codecogs.com/svg.latex?r_j=\frac{\displaystyle\sum_{\substack{i=1\\\hat{y}_j^{(i)}\neq&space;y^{(i)}}}^m\left(w^{(i)}\right&space;)}{\displaystyle\sum_{i=1}^m\left(w^{(i)}\right)}" title="http://latex.codecogs.com/svg.latex?r_j=\frac{\displaystyle\sum_{\substack{i=1\\\hat{y}_j^{(i)}\neq y^{(i)}}}^m\left(w^{(i)}\right )}{\displaystyle\sum_{i=1}^m\left(w^{(i)}\right)}" />


### 7.5.2 - Gradient Boosting

## 7.6 - Stacking

## - Concluding Remarks


[anomaly_detection]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/anomaly_detection.png "illustration of anomaly detection"
