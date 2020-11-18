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

## 7.2 - Bagging and Pasting
### 7.2.1 - Using SKL
### 7.2.2 - Out-of-Bag Evaluation


## 7.3 - Random Patches and Random Subspaces

## 7.4 - Random Forests
## 7.4.1 - Extra-Trees
## 7.4.2 - Feature Importance

## 7.5 - Boosting
## 7.5.1 - AdaBoost
## 7.5.2 - Gradient Boosting

## 7.6 - Stacking

## - Concluding Remarks


[anomaly_detection]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/anomaly_detection.png "illustration of anomaly detection"
