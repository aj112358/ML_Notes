# Ch 6 - Decision Trees

## Chapter Objectives

1. Learn how to visualize a decision tree once trained
    1. Will need the 'graphviz' library installed!
2. Learn how a decision tree makes predictions
    1. For classification tasks
    2. For regression tasks
3. Learn about impurity of a node
    1. Gini impurity metric
4. Learn how a decision tree estimates class probabilities
5. Learn and implement SKL's CART training algorithm for decision trees
6. Compare and contrast the Gini impurity and entropy metrics
7. Learn how to implement regularization

A *decision tree* is another ML algorithm, one that can be used for both regression and classification tasks. They are very powerful algorithms, can be used for multioutput tasks, and are capable of dealing with complex data sets. They are the fundamental component of *random forest* ML algorithms, which we will see in the next section (Ch8).


## 6.1 - Training and Visualizing a Decision Tree

To begin our study of the decision tree ML algorithm, we will start by implementing one and seeing how it makes predictions. As we have been so far, we will use the Iris data set and create a classification algorithm. To the JN!

Once trained, the actual decision *tree* can be visualized using the powerful 'graphviz' program. SKL uses the 'export_graphviz' method to create a .dot file containing the code for the tree. This file can then be converted to PDF or PNG using the graphviz library via the command:

dot -Tpng iris_tree.dot -o iris_tree.png

We can also simply "open" this file and compile the code to create the visual in the JN itself. Here is what the decision tree looks like for our Iris classification model:

<insert tree here>

**Note: One of the very nice features of using a decision tree ML algorithm is that they require very little data preparation. They don't even require *feature scaling or cerntering*!**

## 6.2 - Making Predictions

To see how a decision tree model makes predictions (for classification tasks), we can use the visual of the decision tree above. 

We see that all non-leaf nodes begin with a statement - in our case, either "petal length (cm) <= 2.45" or "petal width (cm) <= 2.45". Given any input instance, we simply traverse through the tree depending on if the answer to each non-leaf node's statement is true (traverse left) or false (traverse right). The ultimate leaf node we finish at is the prediction class for that input instance.

*Some Terminology: You can refer to the nodes as "depth-2 right", which would mean the right child node at a depth of 2 into the tree.*

Other than the statement in the non-leaf nodes, each node in a decision tree has four attributes:

1. **samples**: A count of how many training instances the *node* applies to.
    + *ex: There are 50 instances with petal length <=2.45; there are 54 instances of petal length >2.45 and petal width <=1.75*
2. **value**: A count of how many training instances of *each class* this node applies to.
    + *ex: For the purple node, of the 46 samples, 0 apply to the setosa class, 1 applies to the versicolor class, 45 apply to the virginica class*
3. **gini**: A metric that measures the "impurity" of a node (Wikipedia: *a measure of how often a randomly chosen element from the (training) set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset*)
4: **class**: The predicted class for that node.

**Note:** SKL uses the 'CART Algorithm' when training a decision tree, which produces only binary tree. Other algorithms can be used to produce decision trees with more than two child nodes (ex: ID3 algorithm).


With regards to the Gini impurity, a node is said to be *pure* if the gini metric evaluates to zero (meaning all training instances that node applies to belong to the same class). The equation to copute the Gini impurity is:

<img src="http://latex.codecogs.com/svg.latex?G_i=1-\sum_{k=1}^n\left(p_{i,k}\,^2\right)" title="Gini Impurity" />

where $p_{i,k}$ is the ratio of class-k instances to the samples in the i-th node. As a sample calculation, the Gini impurity for the green versicolor class node is:

<img src="http://latex.codecogs.com/svg.latex?G_1=1-\sum_{k=1}^3\left(p_{1,k}\,^2\right)=1-\left[p_{1,1}\,^2&plus;p_{1,2}\,^2&plus;p_{1,3}\,^2&space;\right]=1-\left[\left(\frac{0}{54}\right)^2&plus;\left(\frac{49}{54}\right)^2&plus;\left(\frac{5}{54}\right)^2\right]=0.1680384088\approx0.168" title="Sample Gini calculation" />

Finally, being used as a classification algorithm, we can visualize a decision tree ML model's decision boundaries:

<insert picture of decision boundaries>

Each depth of the decision tree produces its own decision boundary. For the root node (depth-0), the decision boundary is the thick vertical line. Since the depth-1 left node was pure (Gini impurity of zero), it cannot be split further. The depth-1 right node howevery has a non-zero Gini impurity, and hence can be separated (it's also not a leaf node). The right-hand region is split based on the decision statement 'petal width (cm) <= 1.75'.

Since we had specified the hyperparameter 'max_depth' as 2, we only have these two decision boundary. Had we set this hyperparameter as 3, we would have gotten another (disjoint) decision boundary (indicated by the vertical dotted lines), splitting each of the two regions separated by the depth-1 decision boundary.


## 6.3 - Estimating Class Probabilities

For a classification task, we can compute the class probabilities for an input instance. We simply traverse the tree to the appropriate leaf node, and can then compute the class probabilities by dividing each number in the value list by the samples quantity.

With regards to the decision boundary image aboe, each of the three colored regions (not including the vertical dotted lines) will yield the **same class probabilities**, regardless of the input instance features.

And, as usual, we can use SKL's '.predict_proba()' and '.predict()' methods to automatically compute class probabilities and predictions, respectively.


## 6.4 - The CART Training Algorithm

In order to train a decision tree, SKL uses the so-called "CART Algorithm" (*Classification and Regression Tree*) to train (aka: "grow") a decision tree. This algorithm executes in a recursive fashion, and performs the following steps at each iteration.

It starts by splitting the training set into two subsets, by using a statement (these statements are written in each non-leaf node of the decision tree visual) invovling a single feature $k$ and a single threshold value $t_k$. It selects these two by attempting to search for the pair that yields the purest subsets. The cost function associated with this optimization process is:

<img src="http://latex.codecogs.com/svg.latex?J(k,t_k)=\frac{m_{left}}{m}\cdot&space;G_{left}&plus;\frac{m_{right}}{m}\cdot&space;G_{right}" title="CART algorithm cost function" />

where $G$ measures the impurity of the corresponding subset, and $m$ is the number of instances in that subset. Once the algorithm has found a split, it applies the above steps recursively to each of the two subsets, continuing until it reaches the maximum specified tree depth OR if it cannot find a split that will reduce the Gini impurity metric.

As we will discuss below, there do exist a few other hyperparameters that can be used to control the stopping of this recursive algorithm:
* min_samples_split
* min_samples_leaf
* min_weight_fraction_leaf
* max_leaf_nodes

**Remark:** This CART Algorithm is an example of a "greedy" algorithm, since it searches for the optimal values for the current iteration without consideration to later iterations. As such, the output will only be a *reasonably good* solution, but perhaps not the true optimal solution. The problem of finding the true optimal solution (i.e. optimal tree) is actually an NP-Complete problem and is hence intractable. 


## 6.5 - Computational Complexity

In order to make predictions, the decision tree must be traversed through to a leaf node. We can assume that a decision tree is approximately balanced, hence this traversal has time complexity $O(log_2(m))$, which we observe to be independent of the number of features! Hence, predictions are usually computed very quickly, even with very large data sets (because of the logarithm).

With regards to training, the CART algorithm compares all features on all samples at each node (less if 'max_features' hyperparameter is specified). As such, training a decision tree has time complexity $O(n x m\*log_2(m))$. It's possible to have SKL speed up the training process by *pre-sorting the data* (set 'presort=True'). Since we know that sorting algorithms themselves can be very time intensive, this is really only useful for small training sets (less than a few thousand nodes), and even then the pre-sorting will slow down the training (even moreso for larger data sets!).


## 6.6 - Gini Impurity or Entropy?

By default, SKL uses the Gini impurity metric by default. There is another metric that can be used called "entropy", set by defining the hyperparameter 'criterion' to "entropy". The concept of entropy was founded in the study of thermodynamics, and the general concept later was introduced in the branch of computer science known as "information theory", the founder of which was Claude Shannon in the early 90's.

In ML, entropy is frequently used as an impurity measure - a set's entropy is zero if it contains instances of only a single class. The mathematical formulation of entropy (in the context of computer science) is:

<insert formula here>

So the question to ask ourselves is which should we use? In truth, it doesn't matter in the end. Both will yield similar decision trees. But there are a few key differences:
* Gini impurity tends to isolate the most frequent class in its own branch (**it is also slightly faster to compute** (because no logarithm?))
* Entropy tends to produce slightly more balanced trees


## 6.7 - Regularization Hyperparameters





## 6.8 - Regression
## 6.9 - Instability

## - Concluding Remarks




[anomaly_detection]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/anomaly_detection.png "illustration of anomaly detection"
