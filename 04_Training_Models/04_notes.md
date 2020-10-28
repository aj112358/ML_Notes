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

The standard form of a linear model is

<img src="http://latex.codecogs.com/svg.latex?\hat{y}&space;=&space;\theta_0&space;&plus;&space;\theta_1x_1&space;&plus;&space;\cdots&space;&plus;&space;\theta_nx_n&space;" title="http://latex.codecogs.com/svg.latex?\hat{y} = \theta_0 + \theta_1x_1 + \cdots + \theta_nx_n " />

where:

<img src="http://latex.codecogs.com/svg.latex?\begin{align*}\hat{y}&space;&\text{&space;-&space;predicted&space;value}\\n&space;&\text{&space;-&space;number&space;of&space;features}\\x_i&space;&\text{&space;-&space;$i^{th}$&space;feature&space;value}&space;\\\theta_j&space;&\text{&space;-&space;$j^{th}$&space;model&space;parameter}&space;\\\end{align*}" title="http://latex.codecogs.com/svg.latex?\begin{align*}\hat{y} &\text{ - predicted value}\\n &\text{ - number of features}\\x_i &\text{ - $i^{th}$ feature value} \\\theta_j &\text{ - $j^{th}$ model parameter} \\\end{align*}" />

The constant term is called the "bias term". The model parameters can be thought of as weights for each feature value, and so we think of the linear model making predictions by multiplying each feature value by the respective weight and summing everything along with the bias term.

Using linear algebra, we can re-express the linear model as follows. Define the following vectors:

<img src="http://latex.codecogs.com/svg.latex?\boldmath{\theta}&space;=&space;\begin{bmatrix}\theta_0&space;\\\theta_1&space;\\\vdots&space;\\\theta_n&space;\\\end{bmatrix}" title="http://latex.codecogs.com/svg.latex?\boldmath{\theta} = \begin{bmatrix}\theta_0 \\\theta_1 \\\vdots \\\theta_n \\\end{bmatrix}" />, <img src="http://latex.codecogs.com/svg.latex?\boldmath{x}&space;=&space;\begin{bmatrix}x_0&space;\\x_1&space;\\\vdots&space;\\x_n&space;\\\end{bmatrix}&space;" title="http://latex.codecogs.com/svg.latex?\boldmath{x} = \begin{bmatrix}x_0 \\x_1 \\\vdots \\x_n \\\end{bmatrix} " />

*insert better looking equations and variable definitions here????*

Now that we have the equation for a linear regression algorithm, in order to "train" it we need to determine what parameter values to select so that our model "fits" the training data the best. We recall that to measure this fit we need to decide on a performance measure to use, and the common one for any regression task is the RMSE. Hence, we can now discuss how to find the parameter values that would minimize the RMSE. As it happens, we can instead discuss how to minimize the MSE, as we know from calculus that the square-root function is minimized if, and only if, it's argument is minimized.

Here is the general formula for the MSE, as we have seen before:

<img src="http://latex.codecogs.com/svg.latex?\operatorname{MSE}(\mathbf{X},h)&space;=&space;\frac{1}{m}\sum_{i=1}^m\left(h(\mathbf{x}^{(i)})-y^{(i)}\right)^2" title="MSE formula" />


### 4.1.1 - The Normal Equation
### 4.1.2 - Computational Complexity

## 4.2 - Gradient Descent

### 4.2.1 - Batch Gradient Descent
### 4.2.2 - Stochastic Gradient Descent
### 4.2.3 - Mini-Batch Gradient Descent





## 4.3 - Polynomial Regression


## 4.4 - Learning Curves

## 4.5 - Regularized Linear Models

### 4.5.1 - Ridge Regression
### 4.5.2 - Lasso Regression
### 4.5.3 - Elastic Net
### 4.5.4 - Early Stopping


## 4.6 - Logistic Regression
### 4.6.1 - Estimating Probabilities
### 4.6.2 - Training and Cost Function
### 4.6.3 - Decision Boundaries
### 4.6.4 - Softmax Regression

## - Concluding Remarks

In this chapter we have learned the many aspects of a classification task.





[anomaly_detection]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/anomaly_detection.png "illustration of anomaly detection"
