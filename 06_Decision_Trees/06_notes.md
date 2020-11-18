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

## 6.3 - Estimating Class Probabilities
## 6.4 - The CART Training Algorithm
## 6.5 - Computational Complexity
## 6.6 - Gini Impurity or Entropy?
## 6.7 - Regularization Hyperparameters
## 6.8 - Regression
## 6.9 - Instability

## - Concluding Remarks




[anomaly_detection]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/anomaly_detection.png "illustration of anomaly detection"
