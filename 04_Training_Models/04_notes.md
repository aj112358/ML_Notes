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

<img src="http://latex.codecogs.com/svg.latex?\operatorname{MSE}(\mathbf{X},h)=\frac{1}{m}\sum_{i=1}^m\left(h(\mathbf{x}^{(i)})-y^{(i)}\right)^2" title="MSE formula" />

Substituting our linear regression model for <img src="http://latex.codecogs.com/svg.latex?h(\mathbf{x}^{(i)})" title="blah" /> gives use the form of the MSE for linear regression (we simplify the notation also by only writing theta as the argument in the formula):
    
<img src="http://latex.codecogs.com/svg.latex?\operatorname{MSE}(\theta)=\frac{1}{m}\sum_{i=1}^m\left(\theta^Tx^{(i)}-y^{(i)}\right)^2" title="MSE formula for linear regression" />


### 4.1.1 - The Normal Equation

To minimize the MSE cost function, we must find the appropriate value of \theta. As it happens, using some calculus and linear algebra, one can show that the solution of this minimization problem can be written in closed-form as:

<img src="http://latex.codecogs.com/svg.latex?\hat{\theta}=(X^TX)^{-1}*X^Ty" />

where:

- \hat{\theta} - value of \theta that minimizes the MSE
- y - column vector of target values

This equation is called the "normal equation". Let's implement this as Python code in the JN:

<see JN>
    

### 4.1.2 - Computational Complexity

When doing computational mathematics, it is important to consider the time and space complexity of your code and computations. In our case, to use the normal equation requires us to compute the inverse of a (n+1)x(n+1) sized matrix (n is the number of features), which has a computational complexity (time?) somewhere between O(n^2.4) and O(n^3). Since SKL first computes the SVD of the matrix, it's computational complexity happens to be around O(n^2).

Unfortunately, even using the SVD can get very slow if your ML problem has a large enough amount of features. The good news is that since both the normal equation and using the SVD are both linear problems, the space complexity is usually not an issue as long as you have enough memory!

Finally, once you have created your linear regression model, using it to make predictions has very low computational complexity (it is a linear model after all; it's only computing sums and products!).

Now that we know the basics of linear regression, we look at a completly different way to **train** a linear regression model, which uses a limiting technique as opposed to direct computation of the model parameters.


## 4.2 - Gradient Descent

The method of **Gradient Descent** is a more general *optimization algorithm* that can be applied to a variety of problems in order to find optimized solutions. It uses the gradient of the function as the steepest path to a local minimum or local maximum, and it does this in an iterative fashion.

We start with an initial guess for \theta, simply using random values as the model parameters. This step is called "random initialization". We then use a pre-defined (or iteratively computed) step size along with the gradient to compute the next iteration. Eventually, the iterations will converge to the local minimum.

Here, the step size is a hyperparameter called the "learning rate". If chosen too small, the algorithm will take longer to converge. If chosen too large, the algorithm may overshoot the target local minimum and the iterations will diverge.

One main difficulty that could arise when implementing gradient descent is the cost function (in the case of linear regression, the MSE) may be discontinuous or non-differentiable. The following diagram depicts two scenarios that could affect the execution of gradient descent:

<insert image>
    
As can be seen, if the random initialization is too far left, then the gradient descent algorithm will instead reach the local minimum, confusing it for the global minimum. If the random initialization is too far right, then the gradient descent algorithm will take a long time to execute as it has to travel along the portion of the graph with near-zero slope.



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
