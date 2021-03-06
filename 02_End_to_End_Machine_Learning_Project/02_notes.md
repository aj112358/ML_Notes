# Ch 2 - End-to-End Machine Learning Project

## Chapter Objectives

1. Learn the main steps of a ML project
2. Illustrate these steps on a ML project
3. Learn the details of each step
4. Write sample code for each step


## 2.0 - Steps in a ML Project

Here are the main steps in any ML project:

1. Look at the big picture
2. Get the data
3. Discover and visualize the data to gain insights
4. Prepare the data for ML algorithms
5. Select a model and train it
6. Fine-tune your model
7. Present your solution
8. Launch, monitor, and maintain your ML system

To illustrate these steps, we will go through an entire ML project from start to finish using a real data set, one that is very commonly used as an example for any beginning ML student, called the "California Housing Prices dataset". This is a data set that contains data on prices of houses in California back in 1990 which was collected through a census survey. This data set can be found on Kaggle here: https://www.kaggle.com/camnugent/california-housing-prices

_NOTE: For the purposes of this course, we will use a modified version of this data set (see later...)_

There any many different sources where one can procure data sets from. Here are some online sources:

* Open data repositories
  * UC Irvine ML Repository
  * Kaggle
  * Amazon's AWS data sets
* Meta portals (list of data repositories)
  * Data Portals
  * OpenDataMonitor
  * Quandl
* Other websites
  * Wikipedia's list of ML data sets
  * Quora
  * Datasets subreddit
  
Of course, it is also possible to collect your own data! Here are some ways you could do so:

* Sensor data
  * Arduino Uno microcontroller
    * Has sensors for temperature, humidity, water level, sound, etc.)
    * Will have to make your own circuit
* Web scraping
  * Write Python code to scrape a website
    * Libraries to use: BeautifulSoup, Requests, lxml, Selenium
* In-Person data collection
  * Create your own surveys (with qualitative and quantitative questions)
  * Hold focus groups
  

## 2.1 - Look at the Big Picture

This first step is to get a very quick high-level overview of our data set, and to set out our objective. 

To start, we can go to the Kaggle page (linked above) to see what's written about it, what variables are included in the data, and other information. If you scroll down the webpage to the "Data Explorer" section, you can see a high-level quick overview of the data using either of the three visualization options ('Detail', 'Compact', 'Column'). 

Under the 'Detail' tab, we see there are 10 variables in this data set including location (longitude & latitude coordinates, proximity to the ocean), various descriptive features of a house (age, total number of rooms/bedrooms), and population data (number of residents in a block, median income). Any units pertaining to numerical data is also described. We can also see some initial histograms of each variable which gives us some insight to the distribution of values for that variable (ex: all clustered about the same value, or spread out evenly)

The 'Compact' tab shows us the actual tabulated data. We will do a more in-depth data analysis in the later stages of a ML project, but for now, the main thing we can see is the decimal accuracy of each variable.

The 'Column' tab gives a bit more detail than the 'Detail' tab, including some basic descriptive statistics, but otherwise has mostly the same information as the 'Detail' tab.

One thing to note is that the data in this data set is for so-called "block groups" in California. A "block group" (aka: "district") is defined to be the smallest geographical unit for which the US Census Bureau publishes sample data.

Next, we must carefully understand what our objective is.


### 2.1.1 - Frame the Problem

With this data set, our objective is to **predict the median housing price in any given district**. Although, as data scientists, this may be our mathematical objective, there may be more business objectives that we need to also carefully take into consideration.

If you work for an organization, you may wish to ask your manager some more probing questions like:
* How your model will be used?
* How does the organization expect to benefit from your model?
* What do they consider a 'success' and a 'failure'?
* What is the timeline for the project?
* How is the data coming in?
* Where are your predictions going?
* What computational resources are you given to create your model?
* What type of audience will you be presenting your solution to?
* Is the data stored on a cluster?
* Will the data be streaming in continuously over time?
* Has the organization done a similar project in the past?
* How are you solving your problem currently?


Knowing the answer to as many of these questions as possible will help you in your endeavor to create an appropriate and accurate ML model - it will determine what algorithms you choose to investigate, how much time and effort you spend fine-tuning, etc.

For the purposes of this course, we assume the scenario that our ML model's output will be fed into yet another ML system. This second ML system will ultimately make the decision of whether a certain block group is worth looking into for future investment.

Finally, we can think about what _type_ of ML problem this is. It is clearly a supervised learning problem (we are given the desired output labels for each input vector). It is also clearly a regression problem (we are trying to predict a numerical value). More specifically, we can classify this as a _multiple regression_ problem since we have multiple input features, and can also classify it as a _univariate regression_ problem since we are only predicting a single output value. Finally, since we are using a static data set and are not expecting data to flow continuously into our ML system, this is a batch learning problem.


### 2.1.2 - Select a Performance Measure

Once we have solidified the objective, our next step is to select a performance measure that we will use to assess the performance of our ML model. 

A common performance measure used with regression problems is the "root-mean-square error" (RMSE), which is written mathematically as:

<img src="http://latex.codecogs.com/svg.latex?\operatorname{RMSE}(\mathbf{X},h)&space;=&space;\sqrt{\frac{1}{m}\sum_{i=1}^m\left(h(\mathbf{x}^{(i)})-y^{(i)}\right)^2}" title="RMSE formula" />

The logic behind this formula is that if the regular error is larger, then the square comes into affect and causes that particular error to be more "exposed" than smaller errors. We then take the average of all the errors, by adding then dividing by m. The squaring process causes the units of measurement to also be squared, hence we take the square root to revert to the normal units.

Another common performance measure is the "mean-absolute error" (MAE), which is written mathematically as:

<img src="http://latex.codecogs.com/svg.latex?\operatorname{MAE}(\mathbf{X},h)&space;=&space;\frac{1}{m}\sum_{i=1}^m\left|h(\mathbf{x}^{(i)})-y^{(i)}\right|" title="http://latex.codecogs.com/svg.latex?\operatorname{MAE}(\mathbf{X},h) = \frac{1}{m}\sum_{i=1}^m\left|h(\mathbf{x}^{(i)})-y^{(i)}\right|" />

Both these formulas are essentially different ways to measure the "distance" between two vectors using different metrics (c.f. norms on metric spaces). In this context, we are finding the distance between each prediction vector and the corresponding vector of target values.


### 2.1.3 - Check the Assumptions

Finally, before we actually start writing any code, we should perform a fail-safe check with our assumptions for the problem. We don't want to be writing code for months on end only to later realize we've been working under a false/undesired assumption and/or realizing we could have done something easier, less time consuming, more cost effective, etc.

Some assumptions you want to consider are:
* Is this a regression or classification task?
* How will our ML model's output be used downstream?
* What is the level of accuracy our model needs? (Very strict or more loose?)
* What is the deadline(s) for this project?

Once you are certain about your objective, performance measure, and assumptions, it's time to start coding!


## 2.2 - Get the Data


### 2.2.1 - Create the Workspace

I will assume that you have already set up your entire workspace, set up your Python environment (as you see fit), installed the appropriate Python libraries, and are able to start a new Jupyter Notebook (JN) and execute code without any problems.


### 2.2.2 - Download the Data

There are many ways that data could be stored. A common way is in a relation database (ex: MySQL, postgreSQL, Microsoft SQL Server, Oracle, etc.). These databases themselves could be spread over a distrubuted cluster (ex: Hadoop), or even in multiple files (ex: different locations' financial data in Excel sheets). You need to make sure that you get all the data at your disposable from any source - the more data means (potentially) a better ML model. You may need to ask your manager for access if you do not have the clearance to the data, and should also look into any legal constraints/concerns as necessary (ex: private information should not get leaked!). Once you have your data, you should take some time to learn about its "data schema", the structure of how the data is stored and the connections between variables/tables in the data.

For this course, we will use a reduced version of the California housing data set which has been simplified by removing many features.

We now continue this discussion in more detail in the associated JN. Here in these notes we simply summarize the main ideas and our findings. **(Go to "ML Step2 2 in JN)**

To download the data, we write our code inside a function, and then simply call this function. We do the same when loading the data into the JN. This allows us to easily do the same in the future for updated data, or if we want to use a new machine.


### 2.2.3 - Take a Quick Look at the Data Structure

With the data downloaded into the JN, we can have a quick initial look at it. We can use various methods provided by the pandas library including .head(), .info(), .describe(), and .columns. We can also take advantage of many features provided by the matplotlib and seaborn libraries including creating a basic histogram of relationships between each attribute, creating a heatmap, etc. 


### 2.2.4 - Create a Test Set

Once you are done with your initial look, the next thing you should do is *immediately split the data into a training set and test set.* It is important to do this before the more in-depth EDA as we don't want to accidentally "discover" some pattern in the test set that may bias our choice of ML algorithm. This is called "data snooping" bias.

We can also think about creating multiple subsets of the test set, each for specific parts of the data we may wish to evaluate our model on *(ex: a subset of each type of ocean proximity)*. This could potentially give us a deeper understanding of our model's strengths and weaknesses.

In order to split our data, we will **use a unique identifier for each instance to dictate whether that instance is allocated to the training or test set**. This method has a few benefits:

1. Random splitting will cause the test set to be different each time our ML program is executed. This method maintains consistency.
2. New data can easily be included in the training and test sets, without affecting the allocation of old data.

In the case that the data does not come with a unique identifier column, we can simply make our own for instance by using the most stable features in the data set. 

Further, we can implement a method of sampling called **stratified sampling**, which involves partitioning a population into sub-populations each of which is homogeneous with respect to a certain characteristic/attribute. Then, we can ensure that we sample data (into the training and test set) in the same ratios of these sub-populations.

In our data, we infer that the median_income attribute is very important for predicting the median_house_value. So we perform stratified sampling with this attribute. We convert this numerical attribute into categorical attributes defined by income levels as seen below:

Category | Income Level
---|---
1 | 0 - 1.5
2 | 1.5 - 3.0
3 | 3.0 - 4.5
4 | 4.5 - 6.0
5 | 6.0+

Then, we use Scikit-Learn to split our data into a training set and test set using the method of stratified sampling. Having finally split our data, we can now move on to our in-depth EDA.


## 2.3 - Discover and Visualize the Data to Gain Insights

So far, we have only taken a very high-level look at the data when we downloaded it (cf. 2.2.3 above). In this step, we invest our time in looking deeper into the data.

**NOTE: We must only explore the _training set_ and NOT the test set!!!**

In case we want to add things to the stratified training set we wish to explore, we can simply create a copy of it to fool around with. This way, the original is kept unchanged.

### 2.3.1 - Visualizing Geographical Data
### 2.3.2 - Looking for Correlations

### 2.3.3 - Experimenting with Attribute Combinations


## 2.4 - Prepare the Data for ML Algorithms


### 2.4.1 - Data Cleaning
### 2.4.2 - Handling Text and Categorical Attributes
### 2.4.3 - Custom Transformers
### 2.4.4 - Feature Scaling
### 2.4.5 - Transformation Pipelines


## 2.5 - Select and Train a Model

We have invested a lot of time in understanding the problem, exploring our data set, and creating a transformation pipeline. We are finally ready to take our cleaned data and train a model using it!

### 2.5.1 - Training and Evaluating on the Training Set
### 2.5.2 - Better Evaluation Using Cross-Validation


## 2.6 - Fine-Tune Your Model

### 2.6.1 - Grid Search
### 2.6.2 - Randomized Search
### 2.6.3 - Ensemble Methods
### 2.6.4 - Analyze the Best Models and Their Errors
### 2.6.5 - Evaluate Your System on the Test Set


## 2.7 - Present Your Solution

Now that we are finally done the model creation process, the next step is to **get approval for launch**. This is usually done in the form of a presentation, which should be concise, have clear visualizations and easy-to-remember statements. Some things to may wish to discuss:
* your solution
* highlight what you have learned about the problem and the data
* what assumptions were made
* how well your model performs
* what your model's limitations are

Hopefully you have been documenting your entire process throughout all your hard work!

For our example, our ML model doesn't seem to perform that much better than the original method of prediction (which was simply the experts' price estimates - usually off by 20%). It may still be a good idea to launch though, as it would allow the experts' time to be used for other things.


## 2.8 - Launch, Monitor, and Maintain Your System

Now that we have gotten approval for launch, we need to get our ML model ready for production. Some things we should do:
* Clean our code
* Write documentation
* Create tests (unit tests???)

We can now deploy our ML model into our production environment. The simplest way to do this is to save our model (as a pickle file) and load it directly into the production environment. It can then be used to make predictions by simply calling the 'predict()' method.

If we happen to be launching this on a web application (phone app; website) we can look into setting up a query code between the app and our model. Another option is to 'wrap' your model in some online web service (Flask; Docker) so that the web application can query it through a REST API. This has the added benefit of allowing you to make updates when necessary without disrupting the web app, and also allows you to use as many servers on the web service as needed to balance incoming requests.

Yet another option is to deploy the model onto a cloud platform like 'Google Cloud AI Platform'. This provides a web service that automatically takes care of load balancing for you. You can use this web service in your organizations website or on a phone app.

**This is not the end...**

We need to think about writing code to monitor our ML system's live performance, and trigger alerts if there is a drop. A drop could be quick/steep, but could also be a slow decay over time. This is common as ML systems get old as time goes on and need to be continually updated to our changing world.

One way to monitor our ML system's live performance is to simply infer it from downstream metrics. For example, if we had designed a recommender system which provides suggestions for users, and we observe that the number of suggested products sold drops below the same for non-suggested products, this means something is awry. The main suspect is the model - perhaps our data pipeline is broken, or the model needs to be re-trained on fresh data.

Unfortunately, it is not always possible to determine the model's performance _without_ human analysis. For example, if we had designed an image classification system used to detect product defects on a production line, then it would be very important to get alerts quickly (otherwise we could accidentally ship defective products to thousands of customers!). One solution, with regards to human analysis, is to get people to check various pictures that the model had classified (especially those it was unsure about).

So! We need to implement a monitoring system (with or without human analysis) that not only detects drops in performance but also defines procedures to follow in case of such occurences. Unfortunately, **this can be a lot of work**, and may even be more work than constructing the model itself!

Not only that, but if the data keeps updating, you will have to keep up and re-train your ML model regularly. Of course, the easiest thing is to automate this whole process:
* Collect fresh data regularly and label it
* Write a script to train and fine-tune a model automatically and periodically
* Write a script to evaluate both new and old model on an updated test set (only deploy if new one is better)

Another thing we should do is **evaluate the quality of input data**. It's possible the model's performance is dropping because of poor-quality inputs (ex: bad sensor, upstream data is stale). Even though it may take some time for your ML system's performance to drop enough to trigger an alert, we can still try to catch such issues early. Some things to look out for:
* Increasingly more inputs are missing a feature
* Mean/standard deviation drifts too far from the training set
* Categorical feature starts containing new/unexpected categories

Finally, as is good practice in programming, keep backups of all your models, pipelines, etc. Have these on the ready in case you need to revert to an older model; you can even automate this process using a script as well! You could also consider keeping backups of each version of the data sets used - you can rollback to a previous one if the new one (during the collection process?) gets corrupted. Although, this would take up a lot of memory so always think of the trade-offs! But, having backups allows you to evaluate your model on any of them too.

_Note: Something you can think about is creating several subsets of the test set as well, so that you can evaluate how your model performs on specific parts of the data. (ex: subsets each for a specific kind of ocean proximity). This will give you a deeper understanding of your model's strengths and weaknesses._


## - Concluding Remarks

Working through the entire 8-step ML process is quite an ordeal! It takes a lot of time investment as well as setting up a lot of initial infrastructure. There are a lot of tools at our disposal for each step of the process. Most of the work was contained in the data cleaning steps and the model creation process (model selection and fine-tuning). Once we have a pipeline built, it is easy to automate this task for production. 

**Important Note:** The ML algorithm themselves are important, but **NOT as important as** having a good understanding of the entire ML process. It is better to simply be comfortable with a few ML algorithms and be very efficient with the ML process, than it is to spend lots of time learning new/advanced algorithms and not getting more experience with the ML process. 

GL;HF :)




[anomaly_detection]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/anomaly_detection.png "illustration of anomaly detection"
