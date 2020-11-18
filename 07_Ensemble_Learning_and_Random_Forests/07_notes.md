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




### 7.2.1 - Using SKL
### 7.2.2 - Out-of-Bag Evaluation


## 7.3 - Random Patches and Random Subspaces

## 7.4 - Random Forests
### 7.4.1 - Extra-Trees
### 7.4.2 - Feature Importance

## 7.5 - Boosting
### 7.5.1 - AdaBoost
### 7.5.2 - Gradient Boosting

## 7.6 - Stacking

## - Concluding Remarks


[anomaly_detection]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/anomaly_detection.png "illustration of anomaly detection"
