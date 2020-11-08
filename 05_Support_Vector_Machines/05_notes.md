# Ch 4 - Training Models

## Chapter Objectives

1. Formalize the linear regression ML algorithm
2. Learn about gradient descent approaches and its variants
    1. Batch gradient descent
    2. Mini-batch gradient descent
    3. Stochastic gradient descent
3. Learn about polynomial fitting
4. Learn how to detect overfitting with polynomial fitting
5. Learn about regularization techniques to reduce risk of overfitting
6. Formalize the logistic regression ML algorithm (for classification)
7. Formalize the softmax regression ML algorithm (for classification)

Thus far in our learning endeavors, we have learned about the overall ML process, created a full regression ML model, and created a classification ML model, all the while learning about various SKL functions and ways of thinking when implementing ML theory.

Now, we will invest some time in learning about the mathematics behind some of the ML algorithms we have been using. 


## 4.1 - Linear Regression

### 4.1.1 - The Normal Equation

### 4.1.2 - Computational Complexity

## 4.2 - Gradient Descent

### 4.2.1 - Batch Gradient Descent

## - Concluding Remarks

In this chapter we have learned some of the foundational mathematical theory underlying linear regression, polynomial regression, and logistic regression. We learned about the iterative technique known as gradient descent (and its variants), and implemented with each of the above three ML algorithms. We plotted and used learning curves to investigate how our ML models were learning which gave us insight into underfitting and overfitting. In particular, for logistic regression as well as the more general softmax regression techniques, we learned how to visualize the resulting ML model's decision boundaries.

We also learned the very important concept of regularization (and three variants: ridge, lasso, elastic net) and how it can be used to reduce overfitting during training. We also looked briefly into early stopping as another technique to control the training of our ML model.

Finally, we implemented all the above in Python in our JN. Nice work!





[anomaly_detection]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/anomaly_detection.png "illustration of anomaly detection"
