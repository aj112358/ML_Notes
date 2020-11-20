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

Once the entire ensemble has been created, **each now-trained predictor itself has an assigned weight** compared to all others based on its overall accuracy on the weighted training set. With the ML model created, it makes predictions by taking an input instance and getting the individual predictions from all the predictors, sums the respective weights for each class across these predictions, then predicts the class as the one receiving majority of (weighted) votes.

This method is similar to the method of gradient descent we have seen previously. In this case, we are continually adding predictors to our ensemble to make it better overall, instead of working with only a single predictor and minimizing a cost function.

**Remark:** A very clear downside of sequential learning is that you *cannot parallelize the learning*. Each predictor must wait until the previous one is fully trained.


#### Mathematics of AdaBoost

We start by setting each instance's initial weight to $1/m$, where $m$ is the number of instances. Once a predictor is trained, we can compute its weighted error rate (when applied to the training set) via:

<img src="http://latex.codecogs.com/svg.latex?r_j=\frac{\displaystyle\sum_{\substack{i=1\\\hat{y}_j^{(i)}\neq&space;y^{(i)}}}^m\left(w^{(i)}\right&space;)}{\displaystyle\sum_{i=1}^m\left(w^{(i)}\right)}" title="Weighted error rate of j-th predictor" />

where $\hat{y}\_j^{(i)}$ is, for the j-th predictor, its prediction for the i-th instance.

We then compute that now-trained *predictor's* weight via:

<img src="http://latex.codecogs.com/svg.latex?\alpha_j=\eta\cdot\operatorname{log}\left(\frac{1-r_j}{r_j}\right)" title="Predictor weight" />

where eta is the **learning rate hyperparameter** (default 1).

From these two equations above, we can see that if a trained predictor has good performance on the training set, it's errot rate r_j will be low, and hence the predictor's weight \alpha_j will be high (since 1-r/r > 0, and logarithms are strictly increasing). If the predictor happens to be just guessing randomly, then it's weighted error rate will be close to 0.5 whence that predictor's weight will be close to 0 (since log(1)=0). Finally, if the predictor is wrong most of the time, then the weighted error rate will be high, so that predictor's weight will actually be negative (since log(x)<0 for x<1).

With these values computed, the next step is for AdaBoost to update the instance weights. Recall we wish to increase the weight of the *misclassified* instances. This is done via:

<img src="http://latex.codecogs.com/svg.latex?w^{(i)}\leftarrow\begin{cases}w^{(i)}&space;&\quad\text{if&space;}\hat{y}_j^{(i)}=y^{(i)}\\w^{(i)}\cdot\exp(\alpha_j)&space;&\quad\text{if&space;}\hat{y}_j^{(i)}\neq&space;y^{(i)}\\\end{cases}&space;" title="Weight update rule for new instance weights" />

for all instances i=1,2,...,m. Finally, these new instance weights are normalized by dividing each by the total sum.

Then, we simply repeat this entire process with the subsequent predictor. This AdaBoost algorithm will stop when either **the number of predictors is reached** or **a perfect predictor is found**.

Once the ensemble is fully trained, predictions on new data is computed via:

<img src="http://latex.codecogs.com/svg.latex?\hat{y}(x)=\operatorname{argmax}\left(\displaystyle\sum_{\substack{j=1\\\hat{y}_j(x)=k}}^N&space;\alpha_j\right)" title="http://latex.codecogs.com/svg.latex?\hat{y}(x)=\operatorname{argmax}\left(\displaystyle\sum_{\substack{j=1\\\hat{y}_j(x)=k}}^N \alpha_j\right)" />

where N is the number of predictors in the ensemble.


#### AdaBoost in SKL - SAMME

As it happens, SKL uses a *multi-class* version of AdaBoost called **SAMME**, which stands for **Stagewise Additive Modelling using a Multiclass Exponential loss function**. In the case of only two classes, SAMME is equivalent to AdaBoost.

If the predictors you select to implement are capable of producing class probabilities, then SKL has the ability to use a variant called **SAMME.R** (R stands for "Real"). This variant uses the class probabilities instead (rather than predictions), and so will generally perform better.

In any case, SKL offers the 'AdaBoostClassifier' class (and an analagous one for regression). Let's go to the JN and implement AdaBoost!

*NOTE: A decision tree with a height of one is called a "decision stump"!*

To regularize the AdaBoost model (ie. to reduce overfitting), we can either **reduce the number of estimators** or **most strongly regularize the base estimator**.


### 7.5.2 - Gradient Boosting

The second popular boosting method is called **gradient boosting**. As usual for all boosting algoriths, gradient boosting works by sequentially adding predictors to the ensemble, where each subsequent predictor's training is based on the previous one. 

With gradient boosting, we now have each subsequent predictor be trained on the **residual errors** made by the previous one (we do not change the weights of each training instance as is done with AdaBoost). Once we have trained each predictor, in order to make a prediction, we simply get the individual predictions and add them up.

To illustrate gradient boosting, we will implement it with a regression task (we have been doing a lot of classification thus far in the book!). We will use a decision tree as the base estimator. In this case, this type of ML algorithm/training is called **gradient boosted regression trees** (aka: "gradient tree boosting"). As our data set, we will use noisy quadratic data. And we will use three predictors in total, all decision trees. Let's go to the JN and see how it's done!

So, we implemented gradient boosting from scratch. We can also use SKL's 'GradientBoostingRegressor' class to create GBRT ensembles. It comes equipped with hyperparameters that we can fine tune to regularize the model (the usual ones associated with decision trees). It also has hyperparameters that allows us to control the actual *ensemble training process* itself (ex: number of trees, etc). We quickly show an implementation of this class in the JN (for the same regression problem stated above).

In particular, the SKL class has the hyperparameter 'learning_rate' which allows you to scale the contribution of each tree. If you set it to a low value, you will need to balance that with a high number of predictors, in order to ensure a good fit to the training data and ensure good generalization. This is a regularization technique called **shrinkage**. 

#### How to Determine Number of Trees to Train

So with the knowledge of these boosting techniques, we can ask the question: **How to find the optimal number of trees to use?**

One good answer to this question is to implement **early stopping** - we can look at the validation errors and select the number of trees that minimize them. 

A quick and easy way to do this is to train a full GBRT ensemble and then manually find the minimum validation error and the number of corresponding trees. Luckily, we can take advantage of SKL and use GBRTs build-in 'staged_predict()' method to compute the validation errors. This method returns an iterator over the predictions made by the (currently build) ensemble at each stage of training. Once we know the optimal number of trees, we have to train an **entirely new GBRT ensemble** using this optimal number! We go to the JN to show how this is done.

With the above strategy, we are forced to train **two entirely separate** GBRT ensembles. We can also **truly** implement early stopping by actually stopping the training cycle once a minimum validation error is reached. This is done easily by simply specifying the GBRT hyperparameter 'warm_start' to "True" when you instantiate the gradient boosting class. This causes SKL to perform *incremental training* - it will keep any currently existing trees when you call the 'fit()' method. We go to the JN to see how this is done.

*Note: I don't think SKL has a special early stopping method for the GBRT classes, so you can just do it manually (using if-else statements!)*

#### One Final Note

SKL's 'GradientBoostingRegressor' class has a hyperparameter called "subsample", which allows us to specify what fraction of the training data do we wish to train each tree on (so it's a number between 0.0 and 1.0). As usual, the training samples are selected radomly for each tree, and hence we have another technique by which to trade a higher bias for lower variance, thereby (hopefully) improving the predictive power of our ensemble.

Of course, this also allows the training to complete much quicker. This technique (as you can probably guess) is called **stochastic gradient boosting**. It is also possible to use different cost functions with gradient boosting by specifying the 'loss' hyperparameter.


### 7.5.3 - XGBoost (A Quick Note)

In 2014, an individual names *Tianqi Chen* released a Python library that actually implements a more optimized version of gradient boosting. He called this method **extremem gradient boosting**, or simply **XGBoost** for short. It allows gradient boosting to be very fast and scalable.

Since it's built off the idea of gradient boosting, its API is similar to that of most SKL classes. We won't look into depth with XGBoost but only provide a basic illustration in the JN. ***I could not get XGBoost to install properly***

**NOTE: XGBoost is involved/used in many winning ML competitions, so you should definitely check it out!!!**


## 7.6 - Stacking

Another ensemble method is **stacking**, short for **stacked generalization**. Here, instead of using a trivial function to aggregate each predictor's predictions, we actually **train another model entirely for the purposes of aggregation**. 

The image below shows how to make predictions with a stacked ensemble model:

<insert image here>

We see that a new input instance is passed to each predictor in or ensemble, thereby yielding one prediction from each. Then we pass each individual prediction into a final ML model called a **blender** (aka: "meta learner"), and thus get the final prediction for the input instance.


#### How to Make the Blender

That all seems simple enough, so the question now is how to make the blender. One common approach is to use a so-called **hold-out set**, and another is to make **out-of-fold predictions**. We will look at implementing a hold-out set.

The two diagrams below show the process of making the blender:

<insert pictures here>
    
We start by splitting the training set into two subsets, and only use the first one to train each individual predictor. The other subset is said to be **held out**. We can refer to these predictors as the **first layer**.

Once the predictors are trained, we then evaluate them each on the held-out set (this set contains new data the predictors have not seen) which will yield three predictions. We now use these predicted values to create a new training set. This new training set will have the same number of instances as the held-out set, will be N-dimensional, where N is the number of predictors, and will have the same target values as the held-out set. We then decide what ML algorithm we wish to use as our blender, and train it using this new training set.


#### Implementing Multiple Blenders/Layers

Now, as you may have imagined, it is possible to create many different blenders, and thus have many different **layers** in your stacking ensemble, each layer essentially being composed of intermediary blenders and the last layer being a single final blender. To do this, we simply partition our training data into the same number of subsets that we want blenders.

For example, let's say you wanted three blenders. To start you would split the training data into three subsets. We use the first one to train the first layer of predictors, and then use these predictors to make predictions on the second subset. This would then allow use to create a new training set. We use this new training set to train predictors in the second layer. Once the second layer predictors are made, we pass them the third subset to get another new training set. We then use this final new training set to train the final blender. (*That sounds pretty complicated!*)

The image below shows what such a stacking ensemble would look like:

<insert image>

Finally, to use it for making predictions, we take an input instance and pass it to the predictors in the first layer. Each predictor produces its prediction, and passes it to *each* predictor in the second layer (each of which can be thought of a blender itself). These predictors make their predictions and pass them to the final blender, which then yields the final prediction.


#### SKL Implementations

It so happens that SKL does **not** come equipped with classes for implemnting stacking ensembles. We would have to do this from scratch ourselves. Once you understand the idea behind stacking and can visualize it in your mind it shouldn't be a problem to do this. You also have the option of using an already made library, one such option is **DESlib**. I would personally write it from scratch myself.


## - Concluding Remarks

In this chapter, we learned about ensemble learning and the many tools we have for implementing it including random forsts, bagging, boosting, and stacking. We implemented each in SKL, and talked briefly about the statistical theory as to why an ensemble method would provide better predictive power. With bagging, we saw how we can make use of the non-chosen training instances in a validation set. With boosting, we say how it takes advantage of previous predictors' mistakes to become better. With random forests, they have the ability to help use with feature selection. We also looked at early stopping and how to implement it in SKL. Nice work!


[anomaly_detection]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/anomaly_detection.png "illustration of anomaly detection"
