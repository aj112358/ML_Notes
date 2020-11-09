# Ch 5 - Support Vector Machines

## Chapter Objectives

1. Understand intuition behind SVMs
2. Learn how to implement an SVM algorithm for classification
    1. Batch gradient descent
3. Learn how to implement an SVM algorithm for regression
4. Learn the mathematical theory underlying SVM

A "support vector machine" is a ML algorithm that is both powerful and versatile. It can be used for both classification and regression tasks, and is more well suited for the former, especially when you have small/medium sized datasets. It can also handle both linear and non-linear data sets! All-in-all, this is definitely a tool you want to have in your ML toolbox.


## 5.1 - Linear SVM Classification

In order to gain some intuition of how a SVM can be used for a classification task, we look at the visual below:

<insert visual of various decision boundaries>
    
This is some data from the Iris data set. Our goal with this data set (as we have already seen in the last chapter!) is to classify instances as being in the so-called 'versicolor' or 'setosa' class. To do this we need to construct a decision boundary, and we can see three possible candidates in the left diagram: solid-red, solid-pink, dashed-green. Because we can physically separate the classes with a straight line, we say the data is **"linearly separable"**.

Clearly, the dashed-green decision boundary is terrible as it does not even distinguish between training instances with the same labels! The two solid lines are better - they will fit the training data well. BUT! They will not allow for good generalization to new data. Indeed, look at the green dots (representing the 'setosa' class) and look at the instance that is just on the edge of the solid-red decision boundary. If you encountered new data where the petal length was the same as this instance but the petal *width* was slightly larger, then with this solid-red decision boundary your ML model would classify that new data as being in the 'versicolor' class (which of course if very **un**likely!). The same argument can be applied to the solid-pink decision boundary.

Now consider instead the right diagram, which shows a single solid-black decision boundary (the dashed lines are something else as we'll see). This line clearly separates the classes with no ambiguity and it does not fall into the trap/mistake we talked about above. It is at maximum distance from the closest training data instances in each class, these closest instances represented by the pink circles. The distance is visualized by drawing the two dashed lines parallel to the decision boundary. 

The common visualization of an SVM classifier is one of fitting the **widest possible street** between the two classes. This is called "**large margin classification**". One key observation is that **the decision boundary is fully defined by the two instances located on the 'edges' of this street.** These instances are called **support vectors** (the same ones highlighted by the pink circles).

**IMPORTANT NOTE:** SVM models are highly sensitive to each features scales. It is important to perform feature scaling if you intend to investigate the applicability of an SVM model to your data problem!!


### 5.1.1 - Soft Margin Classification



## 5.2 - Non-Linear SVM Classification
### 5.2.1 - Polynomial Kernal
### 5.2.2 - Similarity Features
### 5.2.3 - Gaussian RBF Kernal
### 5.2.4 - Computational Complexity

## 5.3 - SVM Regression

## 5.4 - Under the Hood
### 5.4.1 - Decision Function & Predictions
### 5.4.2 - Training Objective
### 5.4.3 - Quadratic Programming
### 5.4.4 - The Dual Problem
### 5.4.5 - Kernalized SVMs
### 5.4.6 - Online SVMs

## - Concluding Remarks







[anomaly_detection]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/anomaly_detection.png "illustration of anomaly detection"
