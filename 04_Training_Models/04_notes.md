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

As it so happens, the MSE cost function for linear regression is a convex function, hence it only has a single global minimum. Moreover, it is Lipschitz continuous meaning its output values do not change abruptly, which translates into its slope also only changing gradually. As such, the gradient descent method is guaranteed to reach the global minimum with any amount of precision (given you let the algorithm run long enough).

In three-dimensions, the MSE cost function is shaped like a paraboloid. Hence it is possible that it may be stretched along one of its axes. If you consider how gradient descent uses the gradient as the measure of how/where to travel, we can look at the paraboloids level curves as shown in the diagram below:

<insert level curve diagram>
    
As can be seen, if there is a stretch along one of the axes (right-hand diagram), the gradient descent method takes a long time to converge to the minimum. On the other hand, if both axes have the same scale (no stretching), then gradient descent essentially goes directly towards the minimum. This is why **feature scaling is important**!

Finally, when we are training a model, we must remember that we are essentailly searching through the parameter space for the most optimal parameters to use. So if your ML problem has lots of parameters this search will be more difficult and take a longer time. Later in this book we will learn techniques we can implement to reduce the number of parameters that are needed.


### 4.2.1 - Batch Gradient Descent

As you may have alluded to, the general gradient descent algorithm requires us to compute the gradient vector of the cost function. Considering the linear regresson MSE formula above, we can easily compute the partial derivative of an arbitrary parameter \theta_j to be:

<partial derivative formula>
    
and hence, the gradient vector is simply:

<gradient vector> = \frac{2}{m}X^T(X\theta - y)

*NOTE: This formula involves doing computations with the entire training data set at each iteration, hence the name **batch** gradient descent. Unfortunately, one consequence is that gradient descent executes very slowly for large data sets. Luckily, as we know, its good for use when there are a lot of features.*

Finally, to compute the next iterative value, we simply apply the learning rate \eta as shown:

\theta_new = \theta_old - \eta * <gradient>
    
Let's jump to the JN and see what this looks like in practice!

Indeed, we get the same values for the model parameters as when we used the normal equation, as well as from the SKL built-in 'LinearRegression()' class! The question that you should be asking now is: 'What would happen if we used a higher or lower learning rate?'. Here is a visual that depicts, for three different learning rates, the first 10 iterations of gradient descent. The red dashed line is our random initialization, and the blue lines are the result of each iteration (i.e. we take the theta parameters of that iteration and plot the line defined by them).

<insert learning rate visual>
    
As you can see, with the very small \eta=0.02 learning rate, the speed of convergence of gradient descent is much slower. With \eta = 0.5, we greatly overshoot the optimal solution and the lines actually diverge away from our data points!

As we know, we can utilize a grid search to find the optimal learning rate. A good idea is to **limit the number of iterations so that grid search can eliminate models that take too long to converge** (indicating that learning rate was not the optimal one).

Something else to take into consideration is the total number of iterations you wish to run gradient descent for. Too few iterations would mean that your have not reached the minimum, but too many iterations mean you would be wasting time (the model parameters have already, with regards to decimal place accuracy, reached the minimum). To solve this issue, we simply use a very large number of iterations and also implement an example of 'early stopping', where we break the algorithm/loop once the gradient vector has norm smaller than some specified number called the "tolerance" (denoted with \epsilon).

**Note:** It may take O(1/\epsilon) iterations to reach the minimum within your specified tolerance \epsilon. As such, if you decrease the tolerance by a factor of 10, the number of iterations will increase by a factor of 10 also!


### 4.2.2 - Stochastic Gradient Descent

The main problem with batch gradient descent is that it uses the entire training data set at each iteration which causes it to execute more slowly. This is in contrast to "stochastic" gradient descent, which selects a single random training instance at each iteration. This makes the stochastic gradient descent algorithm much faster than the batch version, but the trade off is that its movement towards the minimum is more random and chaotic. As a result, when it does get close to the minimum it will not be able to get arbitrarily close to it due to its chaotic nature, and hence the model parameters it returns are only a rough approximation of the true minimum. On the other hand, there are benefits to implementing stochastic gradient descent. Because of its random behaviour it can "jump out/away" from any local minima and does not get caught like batch gradient descent would. 

So how can we implement stochastic gradient descent and take advantage of the randomness but also solve the issue of not being able to settle on the optimal model parameter values? The solution is to **gradually reduce the learning rate** over each iteration. We start with a large learning rate which allows the algorithm to make progress quickly and possibly get past any local minima. Then we reduce the learning rate to make the jumps smaller which will help the algorithm to more accurately reach the global minimum. The learning rate is governed by a "learning schedule" which is simply a function that decides what the learning rate should be for that iteration.

Potential problems could occur if you try to reduce the learning rate too quickly (it may get stuck in a local minimum) and if it is reduced too fast (it may bounce around the global minimum without getting closer to it, hence returning an even less optimal value).

We implement some code in the JN to illustrate this algorithm:

As you can see, we only get a close **approximation** to the true optimal model parameter values. Also, you can see in the code that we actually run through the training set only 50 times; each round is called an "epoch", and within each epoch the algorithm goes through the training data only 50 times (as opposed to batch gradient descent, which goes through the training set 1000 times).

The final observation is that selecting instances randomly for each iteration may result in some instances not being selected at all and still others being selected multiple times. To make sure that the algorithm goes through each instance during each epoch, one approach is to **shuffle the training set (instances and labels!) at the beginning of each epoch** and then simply go through each instance one at a time. Of course, this takes more time for convergence to the global minimum.

**NOTE:** If you choose not to shuffle the instances during each epoch, then the algorithm will optimize each label one at a time and hence may/will not settle near the global minimum!

Finally, as usual we can use SKL to implement stochastic gradient descent for us using its 'SGDRegressor' class.


### 4.2.3 - Mini-Batch Gradient Descent

"Mini-batch" gradient descent has properties of both batch and stochastic gradient descent algorithms. At each iteration, it uses a random set of training instances called "mini-batches". Using more than one instance per iteration allows it to be less chaotic when trying to find the global minimum, and so when it reaches it its steps will be more near to it (but will still be chaotic). But, as a negative consequence, since its step sizes are smaller it may get stuck at a local minimum and may not be able to "jump out".


### 4.2.4 - Gradient Descent Concluding Remarks

The following figure depcts the iterative movement of each algorithm in the parameter space. As you can see, the batch gradient descent algorithm goes right towards the minimum value, where as the stochastic and mini-batch algorithms still jump around the minimum, with the mini-batch algorithm staying closer to the minimum. 

**Note: Both stochastic and mini-batch gradient descent still have the ability to reach the global minimum, so long as you find and use a good learning schedule!**

<insert gradient descent comparison graph>
    
The table below compares various properties of each algorithm. Don't forget that **scaling is required for gradient descent!**

<insert GD table here>
    

## 4.3 - Polynomial Regression

In cases where a simple line will not fit your data, you can look into doing a "polynomial regression" instead. This involves creating new features as *powers* of old features. Then, with these new features, you **can** create a linear regression model on your non-linear data set! 

Let's see an implementation in the JN.

A very good use of polynomial regression is to **find relationships between features**, which is something that linear regression cannot do. Indeed, when you instantiate a 'PolynomialFeatures' class by specifying the 'degree' parameter, the class will actually create all possible combinations of features up to that degree. For example, if you had two features a & b and you specified degree=3, then the new features would be all of b^3, ab, ab^2, a^2, a^2b, a^3 (i.e. it will make the full spectrum of all cubic terms).

Something to take note of is the complexity of doing a polynomial regression: Specifying degree=d will take your n features and turn them into (n+d)!/n!d! number of features, which can get very large very quickly!


## 4.4 - Learning Curves

So far we have only looked at fitting linearly-related data and quadratically-related data. If we wanted to, we could have fitted the quadratic data with a very high-degree polynomial (ex: 300-degree polynomial). Here is a visual of a linear, quadratic, and 300-degree polynomial fit to the same quadratic data:

<insert visual here>
    
As can be seen, the 300-degree polynomial is grossly overfitting the data, whereas the linear model is underfitting the data. It seems that the quadratic model is the right fit.

In this example, we had already known that a quadratic model would be a good fit, as that is the type of model that was underlying the data we generated! But in practice, the question is **how do we decide on the complexity of our model**? A related question is **how can we measure the level of underfitting or overfitting that may be occurring**?

One way to investigate underfitting or overfitting is to implement cross-validation, as we have seen in the past. If your model performs well on the training set but performs poorly on the validation set(s), this is an indication of overfitting. If it performs poorly on both the training and validation sets, then it is underfitting.

Here, we look into another way to investigate underfitting or overfitting - by plotting the "learning curve" of the ML algorithm. This is a plot that depcits the models performance on the training data as well as on the validation set, both as a function of the training data size (i.e. training iteration). A learning curve is created by essentially training the ML algorithm several times on different sized subsets of the training set. 

We implement some code in the JN to illustrate learning curves for underfitting and overfitting. Here is the learning curve we get for underfitting:

<insert your graph for underfitting>
    red curve -> training data
    blue curve -> validation data

From this plot, we see a few defining characteristics of **underfitting**:
* Performance curve for the training data:
    * Starts with zero RMSE as the model can fit the training data perfectly
    * RMSE starts to increase as perfect fitting cannot happen with more training data (noise & data is not linear in the first place)
    * Training error increases, then plateaus (*hence, simply adding more instances won't help!*)
* Performance  curve for the validation data:
    * Starts with high RMSE as the model cannot make good predictions when having only trained on a few instances
    * RMSE starts to decrease as the model becomes better at making predictions (due to more training data)
    * Validation error decreases, then plateaus (*because a linear model is insufficient for our data set*)
* Both curves, when plateaued, are very close to one another AND have high RMSE

**Recall: A solution for resolving underfitting the training data is to either try a more complex model or create better features**

Next, we look at the learning curves for an overfitting scenario:

<insert graph of overfitting>
    
From this plot, we see it looks very similar to the aboe underfitting plot, but here we also have a few differences that characterize of **overfitting**:
* The RMSE values are much lower
* There is a clear gap between the two curves
    * The model performs better on the (red) training set than it does on the (blue) validation set -> overfitting is occurring!
    
**Recall: A solution for resolving overfitting is to use a larger training set until the validation error and training error converge.**


## ASIDE: The Bias-Variance Trade-Off

An important thing to keep in mind is something called the "bias-variance trade-off". Any ML model's generalization error is the sum of the following three kinds of error:

1. **Bias**: This comes about due to incorrect assumptions about the data (ex: assuming it's linear when it's actually quadratic). A **high-bias model** is most likely to **underfit the training data**.

2. **Variance**: This comes about from the model's excessive sensitivity to small variations in the training data. A model with many degrees-of-freedom is likely to have a **high variance** and hence will **overfit the training data**.

3. **Irreducible error**: This comes about from noise in the actual data set itself. There is usually no way to completely eliminate this error (hence, 'irreducible') but steps can be taken to try and minimize it (ex: fix source of the data, remove outliers)

Based on these definitions, here is a general rule-of-thumb:

**Increasing a model's complexity will (usually) increase the variance and decrease its bias.**

**Decreasing a model's complexity will (usually) decrease the variance and increase its bias.**

When designing ML systems, you will have to decide which of variance or bias is more important to minimize and/or will need to decide how to balance each. Since we cannot minimize both, this is why it is called a *trade-off*.


## 4.5 - Regularized Linear Models

As we know, a good way to **reduce overfitting** is to "regularize" your ML model, meaning to constrain it in some way with respect to its degrees of freedom (ex: reduce the degree of a polynomial regression). With fewer degrees of freedom, it will make it less easy for the ML algorithm to overfit your data.

When it comes to linear models, we normally implement regularization by constraining the weights (i.e. model parameter values) of the model. There are three methods to achieve this and we learn about each in turn:
* Ridge regression
* Lasso regression
* Elastic net

We also look at the notion of "early stopping" as a way to break our .fit method once our validation error reaches a minimum value.


### 4.5.1 - Ridge Regression

"Ridge regression" (aka: "Tikhonov regularization") is a method of regularization used to solve the issue of multi-collinearity when doing linear regression. To the usual MSE cost function we simply add a "regularization term" 

<img src="http://latex.codecogs.com/svg.latex?\alpha\sum_{i=1}^n\theta_i^2"  />

which gives us a new cost function which we can express as follows:

<img src="http://latex.codecogs.com/svg.latex?J(\theta)=\operatorname{MSE}(\theta)+\frac{1}{2}\alpha\sum_{i=1}^n\theta_i^2"  />

where we include the 1/2 factor so that the derivative looks nicer. As a note, the J(\theta) is a standard notation for any general cost function. Also note that the **sum starts with i=1, *not* i=0**, so the \theta_0 term is not regularized. We can express the regularization term in vector form as 

<img src="http://latex.codecogs.com/svg.latex?\frac{1}{2}||w||_2^2"  />

and in regards to gradient descent, we simply add (\alpha w) to the MSE gradient vector we saw previously.

Now, how does this regularization term help to constrain the ML model? Well, since the we are taking the sum of positive terms, in order to minimize the cost function, we now need to consider that we need to also minimize this summation, which means that the values of the model parameters \theta_1, ..., \theta_n will have to be constrained and kept as small as possible!

**IMPORTANT NOTE: The regularization term should only be included in the cost function DURING TRAINING. Once you have trained the model, we use the original cost function (without the regularization term) as the performance measure.**

**Aside: It is common for the cost function to take different forms during training and during evaluation. For training purposes, you want to use one that is easily differentiated and can perform computations with easily. During evaluation, you would want to use a cost function that is a good measure of performance as to how your model will be used. (ex: Classifiers use something called the "log loss" as their cost function, but then use precision & recall for evaluating performance)**

Now, in the expression above, the value \alpha is a hyperparameter that dictates "how strongly" you want to regularize your model. If \alpha=0, then you get the usual cost function, hence you are performing no regularization. If \alpha is large, then to minimize the summation the model parameter values will need to be accordingly very small (potentially very close to 0 if \alpha is large enough). Hence, since your model parameters are very small, your linear model will visually be a near-horizontal line passing through the mean of the data set. Thus, you will need to find a value for \alpha that provides balance to your model.

In the visual below, we have two graphs. The left-hand one is ridge regression applied to a linear model. The right-hand one is ridge regression with a 10-degree polynomial model, and in this case the data has been scaled before regularization. Both graphs have the same original data set, and various values of \alpha:

<insert visual here>
    
As you can see, increasing the value of \alpha does indeed cause the ML model (ie. regression curve) to become more flat. As a result, these higher-\alpha models have reduced variance but increased bias.

Just as with linear regression, we can perform ridge regression either through its closed-form solution or by implementing gradient descent. The closed-form solution to the regularized cost function J(\theta) is:

<img src="http://latex.codecogs.com/svg.latex?  \hat{\theta}=(X^TX+\alphaI_{n+1})^{-1}X^Ty"   />

where I_{n+1} is the usual identity matrix but with a 0 as the (1,1)-entry.

Of course, we can also use SKL's built-in 'Ridge' class to implement ridge regression (using both the closed-form solution AND gradient descent) for us. We go to the JN to see the code


### 4.5.2 - Lasso Regression

Another regularized technique we can implement with linear regression is called "lasso regression" (aka: "Least Absolute Shrinkage & Selection Operator" regression). In Lasso regression, we add a different type of regularization term using the $l1$-norm instead:

<img src="http://latex.codecogs.com/svg.latex?J(\theta)=\operatorname{MSE}(\theta)+\alpha\sum_{i=1}^n|\theta_i|"  />

An important characterizing feature of doing lasso regression is that it will **eliminate the weights (ie. set to zero) of the least important features**. So lasso regression automatically performs feature selection and it outputs a sparse model with very few non-zero feature weights. To prevent the gradient descent implementation of lasso regression from being chaotic around the minimum, you will need to gradually reduce the learning rate (as discussed previously).

You may note that the lasso cost function above is not differentiable at each \theta_i, 1\leq i\leq n (by the nature of the absolute value). So to implement gradient descent, we replace the use of the gradient vector with a "subgradient vector" (which is a generalization of the concept of gradient to non-differentiable functions) whenever any of these model parameters are zero.

The equation below shows the lasso cost function with a subgradient vector that can be used to implement gradient descent:

<insert math equation here>
    
As usual, SKL offers us its 'Lasso' class that we can use to implement lasso regression. See the JN for a simple implementation.


### 4.5.3 - Elastic Net

"Elastic Net" is a regularization technique that is "in between" ridge and lasso regressions. Its regularization term is a combination of those from ridge and lasso as seen below:

<img src="http://latex.codecogs.com/svg.latex?J(\theta)=\operatorname{MSE}(\theta)+r\alpha\sum_{i=1}^n|\theta_i|+\frac{1-r}{2}\alpha\sum_{i=1}^n\theta_i^2"  />

Here the r value is called the "mix ratio" and it allows us to control the extent to which we want the cost function to behave more like ridge regression or lasso regression. As you can see, if r=0 we simply get back the ridge regression cost function, and when r=1, we get the lasso regression cost function.

We implement SKL's 'ElasticNet' class in the JN as a quick illustration.

#### When to use Ridge vs. Lasso vs. Elastic Net vs. Non-regularized linear regression?

First, you definitely want to implement some amount of regularization in your ML models. Ridge regression is a good default method to use, but if you know that most of your features are irrelevant/not useful, then implementing lasso or elastic net would be beneficial as they will essentially get rid of these features by taking their weights close to zero. With these latter two, elastic net is preferred as pure lasso's behaviour is more chaotic (especially so when the number of features is greater than the number of training instances OR when several features are strongly correlated - *but if several features are strongly correlated, you should have already picked up on this in your EDA, right?!*).


### 4.5.4 - Early Stopping

This method of regularization is very different from the above three. With "early stopping" we simply monitor the learning curves and break the execution of the .fit method once we observe a minimum in the validation error. Of course, both the validation and training errors will (on average **over each epoch**) be decreasing. Eventually the validation error will begin to increase which is indicative of the ML algorithm starting to overfit the data. It is at this point we stop the training. Here is an easy visual depicting such a scenario:

<insert pic here>
    
Now, as we discussed and observed with stochastic and mini-batch gradient descent methods, their learning curves are not smooth, so it is usually more difficult to determine when you have reached the minimum validation error. A good solution to follow is to **stop training when the validation error has been increasing for some number of epochs**. Then you simply use the model parameters that were computed step that is the same number of epochs back.

We implement some code for early stopping in the JN.


## 4.6 - Logistic Regression

Some regression algorithms can infact be used for classification tasks, and vice versa. "Logistic regression" (aka: "logit regression") is commonly used as a binary classifier, where a computed probablilty of >50% yields the prediction of that particular instance being the the "positive class" (labelled '1'), and a probability <50% yields the prediciton that the instance is in the "negative class" (labelled '0'). 


### 4.6.1 - Estimating Probabilities

Just as with linear regression, logistic regression assigns weights (the model parameters) to each input instance then adds a bias term. Then logistic regression does one extra step where it passes this weighted result to the the "logistic function" (which is a special case of the more general family of sigmoid functions). Here are the equations:

<img src="http://latex.codecogs.com/svg.latex?\sigma(t)=\frac{1}{1+e^{-t}}"  />

<img src="http://latex.codecogs.com/svg.latex?\hat{p}=h_\theta(x)=\sigma(x^T\theta)"  />

<insert image of logistic function>
    
As you can see, this ML algorithm first computes the weighted sum $x^T\theta$ then uses this as the input to the logistic function. Since the logistic functions codomain is [0,1], we can interpret the outputs as probabilities (*although the logistic curve is NOT a probability density function!*). Finally, to make decisions we simply use the decision function:

<img src="http://latex.codecogs.com/svg.latex?\hat{y}=\begin{cases}0\quad\text{if}\quad&space;p<0.5\\1\quad\text{if}\quad&space;p\geq&space;0.5\\\end{cases}&space;" title="http://latex.codecogs.com/svg.latex?\hat{y}=\begin{cases}0\quad\text{if}\quad p<0.5\\1\quad\text{if}\quad p\geq 0.5\\\end{cases} " />

Note from the graph that if \sigma(t)<0.5 then t<0, and also if \sigma(t)\geq 0.5 then t\geq 0. So if the weighted sum $x^T\theta$ is positive we know the logistic function will be positive, hence the predicted classification will be in the positive class (and vice versa for the negative class).

*Remark: The input variable of the logistic function, t, is called the "logit" which comes from the "logit function" which is the inverse of the logistic function. The logit is also called the "log-odds" since it computes the log of the ratio of the two probabilities.*


### 4.6.2 - Training and Cost Function

We need to construct a cost function that we can use to describe the training process for logistic regression. Then we can look into how to minimize it. The cost function used for logistic regression is as follows:

<img src="http://latex.codecogs.com/svg.latex?c(\theta)=\begin{cases}-log(\hat{p})&\quad\text{if}\quad&space;y=1\\-log(1-\hat{p})&\quad\text{if}\quad&space;y=0\\\end{cases}&space;" title="http://latex.codecogs.com/svg.latex?c(\theta)=\begin{cases}-log(\hat{p})&\quad\text{if}\quad y=1\\-log(1-\hat{p})&\quad\text{if}\quad y=0\\\end{cases} " />

and expressed over the entire set of training data, this then takes the usual average cost structure:

<img src="http://latex.codecogs.com/svg.latex?\operatorname{J}(\theta)=-\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\operatorname{log}(\hat{p}^{(i)})+(1-y^{(i)})\operatorname{log}(1-\hat{p}^{(i)})\right]"  />

It turns out that no closed-form solution for minimizing this cost function exists. Luckily, it can be shown that this cost function is convex and hence, through gradient descent, will converge to its global minimum. So, in order to compute the gradient, we need its partial derivative which can be compute to be:

<img src="http://latex.codecogs.com/svg.latex?\frac{\partial}{\partial&space;\theta_j}\operatorname{J}(\theta)=\frac{1}{m}\sum_{i=1}^{m}\left[\left(\sigma\left(\theta^Tx^{(i)}\right)-y^{(i)}\right)x_j^{(i)}\right]" />

And thus we can now implement logistic regression! We can utilize any of the three gradient descent methods we discussed earlier (batch, stochastic, mini-batch).


### 4.6.3 - Decision Boundaries

As an example of using logistic regression, we apply it to the well-known classification problem involving the "Iris data set". We will construct a binary classifier to predict whether an instance is of the type Iris virginica or is not, and will only use petal width as a single feature. Let's jump to the JN

Finally, we add that like any ML algorithm, we can apply regularization techniques to logistic regression also, using either $l1$ or $l2$ penalties. SKL uses an $l2$ penalty by default.

**NOTE:** The hyperparameter that controls the amount/strength of regularization is *not* alpha, but infact the reciprocal of alpha labelled as C=1/\alpha. So, if you want **less regularization**, you would decrease alpha, hence would need to **increase the hyperparameter C**


### 4.6.4 - Softmax Regression

In the previous JN example, we only made a binary classifer but logistic regression can infact be generalized to multiple classes. This is called "softmax regression" (aka: multinomial logistic regression").

The idea is the same as when we learned about 'multiclass classifiers' (cf. Ch 3 notes). The softmax regression algorithm computes a decision score $s_k(x)$ for each class k as per the formula:

<img src="http://latex.codecogs.com/svg.latex?s_k(x)=x^T\theta^{(k)}" />

Once you have computed all the class scores for each input instance, you then pass the scores (one instance at a time) to the "softmax function" (aka: "normalized exponential") seen below:

<img src="http://latex.codecogs.com/svg.latex?\hat{p}_k=\sigma(s(x))_k=\frac{exp(s_k(x))}{\sum_{j=1}^K(exp(s_j(x)))"  />

This gives us the probability $\hat{p}^k$ that the particular instance belongs to class k. Finally, the prediction is that the instance belongs to the class exhibiting the highest probability, which we can express mathematically as follows:

<img src="http://latex.codecogs.com/svg.latex?\hat{y}=\operatorname{argmax}_k\left(\hat{p}_k\right)=\operatorname{argmax}_k\left(\sigma(s(x))_k\right)=\operatorname{argmax}_k\left(s_k(x)\right)"  />

where the last equality follows from the fact that the standard exponential function is maximized when its argument is maximized. In NumPy, the 'argmax' operator returns the value of a variable that maximizes a function, and in this case will return the class value k.

**Note: The softmax regression algorithm can make predictions with only one class at a time. Hence you must be sure to only use it when you know the classes are mutually exclusive!**

Now, we need to construct a cost function that models the behaviour of our multiclass logistic regression scenario, and hence are able to optimize. The cost function we use for softmax regression is called the "cross entropy" and it is given by the equation below:

<img src="http://latex.codecogs.com/svg.latex?\operatorname{J}(\theta)=-\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^K\left[y_k^{(i)}\operatorname{log}(\hat{p}_k^{(i)})\right]"  />

**Cross entropy is frequently used to measure how well a set of estimated class probabilities matches the target classes.**

**Note: If the number of classes K=2, we simply get back the usual cost function for logistic regression!**

As usual, we require the gradient vector in order to find the minimum of this cost function:

<img src="http://latex.codecogs.com/svg.latex?\nabla_{\theta^{(k)}}\operatorname{J}(\Theta)=\frac{1}{m}\sum_{i=1}^m\left[\left(\hat{p}_k^{(i)}-y_k^{(i)}\right)x^{(i)}\right]"  />

We will illustrate the use of the softmax regression ML algorithm by attempting to classify the entire Iris data set.


## - Concluding Remarks

In this chapter we have learned some of the foundational mathematical theory underlying linear regression, polynomial regression, and logistic regression. We learned about the iterative technique known as gradient descent (and its variants), and implemented with each of the above three ML algorithms. We plotted and used learning curves to investigate how our ML models were learning which gave us insight into underfitting and overfitting. In particular, for logistic regression as well as the more general softmax regression techniques, we learned how to visualize the resulting ML model's decision boundaries.

We also learned the very important concept of regularization (and three variants: ridge, lasso, elastic net) and how it can be used to reduce overfitting during training. We also looked briefly into early stopping as another technique to control the training of our ML model.

Finally, we implemented all the above in Python in our JN. Nice work!





[anomaly_detection]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/anomaly_detection.png "illustration of anomaly detection"
