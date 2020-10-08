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


### 2.2.2 - Download the Data


### 2.2.3 - Take a Quick Look at the Data Structure


### 2.2.4 - Create a Test Set


## 2.3 - Discover and Visualize the Data to GAin Insights














[anomaly_detection]: https://github.com/aj112358/ML_Notes/blob/main/01_The_Machine_Learning_Landscape/01_images/anomaly_detection.png "illustration of anomaly detection"
