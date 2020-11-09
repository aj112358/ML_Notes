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

There are two main ways one can draw a decision boundary for linearly separable data. The first is called **hard margin classification**, in which we have the very strict properties that (i) all instances must be "off the street", and (ii) all instances must be on their appropriate side of the street. As anything in life, the more restrictions you place on something, the harder it is to deal with it. In this case, there are two main issues with hard margin classification:

1. It only works if the data is actually linearly separable
2. It is very sensitive to outliers

These problems are exhibited in the two plots below:

<insert plots of problems>

As you can see, the left plot contains an outlier that is located near the other label instances. In this case, the data is NOT linearly separable and hence it is impossible to satisfy criteria (i) in the definition (i.e. there does not exist a street in which all instances would be off is). In the right picture, we see that there is a green outlier nearer to the blue instances. In this case, the data is still linearly separable BUT if we continue to construct a decision boundary as well as the parallel dashed lines, we see this causes the street to be very narrow and hence our SVM model will not generalize well.

To avoid these issues, we simply relax our strict criteria and use a more flexible model. We must consider the trade-offs and find a balance between **keeping the street as wide as possible** and **limiting any so-called "margin violations"** (instances that fall on the street, or on the wrong side). This flexible model is achieved with **soft margin classification**.

We can use SKL's 'LinearSVC' class to implement a SVM model in Python. This class allows for many hyperparameters, one of which is aptly names **C**. To obtain a **softer** margin classifcation use a **smaller value of C**; for **harder** margin classification, use **larger values of C**. In general, if you have a lot of margin violations, you're gonna have a bad time! So it is a good idea to try and minimize the amount of margin violations, but of course the tradeoff is that your SVM model will not generalize as well. **If you are finding that your SVM model is overfitting your data, you can implement regularization by simply using a smaller C value**.

The above discussion can be summarized with the visual below:

<insert plots of various C values>

We now go to the JN to (very quickly and easily, because we have experience!) implement an SVM classifier that will distinguish between 'virginica' and 'non-virginica' instances using only petal length and width, just as we had done in the last chapter.

**NOTE**: The 'LinearSVC' class regularizes the bias term, so you should *center the training set first* (by subtracting the mean). This is automatically done for you if you choose to use 'StandardScaler' to scale your data first. 


## 5.2 - Non-Linear SVM Classification

Of course, not all your data will exhibit a linear behaviour. For non-linear data sets, you can create new polynomial features (i.e. powers of original features), and thereby hopefully the resulting features will have a linear profile (as we had done in the last chapter!).

As we have seen, we can use SKL's 'PolynomialFeatures' class to do such a thing for us. We can then scale the data (as is very important with SVM!) and finally train our SVM model on the new (hopefully linear) features. We go to the JN to do this now, and will use the "moons data set" from the SKL dataset library (if you plot the data, it essentially looks like a yin-yang symbol).


### 5.2.1 - Polynomial Kernal

Wanting to implement polynomial features begs the question "**what degree should we use**". As with anything in ML, there are trade-offs that have to be taken into consideration.

If we use a lower-degree polynomial, the SVM model will not be able to handle very complex data sets (for example, if your data set is of the 10th-degree but you only create polynomial features up to the 3rd-degree). On the other hand, if you try to use a higher-degree polynomial, then the SVM model will take a very very long time to fit.

To find a balance between these two extremes, we implement a mathematical technique called the "**kernal trick**" (see here: https://en.wikipedia.org/wiki/Kernel_method). This is a way to train an SVM model using many polynomial features of high-degree *without actually creating them*. We can implement the kernal trick using SKL's 'SVC' class, and we do it now in the JN.

**NOTE:** If you find your model is overfitting (underfitting), just try reducing (increasing) the degree when implementing the kernel trick.

As usual, in order to determine some appropriate values of the hyperparameters to use, you can perform a grid search. A good technique is to **start with a wider search, then narrow your search down to the more promising hyperparameter values**.


### 5.2.2 - Similarity Features

Another technique in handling non-linear problems is to create features using a **similarity function**. This is a function that measures *how much a particular instance 'resembles' a specified **landmark***. Essentially, the similarity function provides us with a metric of how similar an instance to some other pre-specified instance (which is called the **landmark**). We can then compute values of this similarity function by inputting every instance into it, and the output will be a new feature value. If you choose to use multiple landmarks, then for each input instance it will have the same number of *new* feature values (i.e. for each input instance, you need to compute the similarity function for *each* landmark).

The benefit of using a similarity function to create so-called **similarity features** is that your non-linearly separable data set will (may?) be transformed into being linearly separable. 

A question to ask is "how to select the landmarks". The obvious choice is to use *every* instance as a landmark, and hence for each of your $m$ input instances you will be creating $m$ new features (because you are comparing that input instance to all other instances). Hence, you will be increasing the dimensionality of your data set, and so it's more likely that your transformed training data will be linearly separable. Of course, the cost of doing this is the greater dimensionality. **Note: We normally drop the original features**.

Luckily, the computational issues are once again bypassed as the 'SVM' class uses the kernel trick.


### 5.2.3 - Gaussian RBF Kernal

One common similarity function is called the "**Gaussian Radial Basis Function (RBF)**". Its equation is below:

<img src="http://latex.codecogs.com/svg.latex?\phi_\gamma(\mathbf{x},l)=\operatorname{exp}\left(-\gamma\lVert&space;x-l\rVert^2\right)" title="Gaussian RBF" />

As you can infer from this equation, instances that are closer to the landmark will have value closer to 1, and instances that are further away will have value closer to 0. Also, the value of gamma dictates the horizontal stretch of the Gaussian. Smaller values of gamma will cause the Gaussian curve to be more narrow, and this translates into each instance's range-of-influence being *smaller*. This will result in a decision boundary that is more irregular and less smooth. Larger gamma values will produce a more smooth decision boundary.

In this way, the parameter gamma **behaves like a regularization hyperparameter. If your model is overfitting, reduce it; with underfitting, increase it.**

We can easily implement this particular similarity function by using the 'SVC' class and specifying the 'kernel' hyperparameter to be "rbf". We do this in the JN now. Here is an image of the decision boundarys for various values of gamma:

<insert image here>

There do exist many other kinds of kernels you can use as similarity functions. The Gaussian RBF is a common one. You should use one that is appropriate for your ML problem. For example, when classifying text documents or DNA sequences you may wish to investigate using some type of "**string kernel**". Your choice will depend on what data structure you are working with.

As usual, you can look into implementing various kernels and use cross-validation and grid search as tools to determine which one works best for your data set.


### 5.2.4 - Computational Complexity

The two SKL classes we have looked at for classification are the 'LinearSVC' class and the 'SVC' class.

The 'LinearSVC' class is based on the 'liblinear' library which implements an optimized algorithm for linear SVMs but does **not** support the kernel trick. However, it does scale (almost) linearly with the number of training instances and features. Training takes longer if you need greater precision (can experiement with the 'tol' hyperparameter).

The 'SVC' class is based on the 'libsvm' library which **does** support the kernel trick. The time complexity is large and hence this algorithm should be used for small/medium sized data sets. It does scale nicely with the number of features.

Here is a summary table for comparison:

<insert table here>


## 5.3 - SVM Regression

We now turn our attention to implementing SVMs for regression tasks. SVM supports both linear **and** non-linear regression. In order to implement SVMs for regression, we look at the reverse of the widest-street analogy: SVM for regression can be thought of as making a street that **contains as many instances as possible on the street** while still limiting "margin violations" (which are, in this case, the instances **off** the street). The width of the street is controlled by the hyperparameter 'tol' - larger values correspond to wider streets.

SVM regression models are said to be **"$\epsilon$-sensitive"** because adding more training instances *inside* the margins (which are defined by the support vectors) does not affect the model's predictions.

We can implement SVM for a regression task using SKL's 'LinearSVR' class. We go to the JN to do that now.

Finally, for non-linear SVM regression, we can simply implement a polynomial kernel just as we had done for classification above. In this case, we instead use SKL's 'SVR' class which does happen to support the kernel trick.

On a final note, SVMs can also be used for outlier detection. See the SKL documention for more details if you are interested.


## 5.4 - Under the Hood
### 5.4.1 - Decision Function & Predictions
### 5.4.2 - Training Objective
### 5.4.3 - Quadratic Programming
### 5.4.4 - The Dual Problem
### 5.4.5 - Kernalized SVMs
### 5.4.6 - Online SVMs

## - Concluding Remarks







[anomaly_detection]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/anomaly_detection.png "illustration of anomaly detection"
