### Ch 1 - The Machine Learning Landscape

* ML has already been around for decades! It has only recently become popular again.
* The first 'mainstream' application of ML was the beloved spam filter, created in the 1990s

## Chapter Objectives

1. Define "machine learning"
2. Compare the three different high-level categorizations of ML
3. Learn basic workflow of a ML project
4. Discuss some challenges of ML
5. Learn how to evaluate and improve a ML model


## 1.1 What is Machine Learning?

Here are a few ways that one might define "machine learning":

* *Machine learning is the science of programming computers so they can learn from data.*
* *Machine learning is the field of study that gives comptuers the ability to learn without being explicitly programmed* - Arthur Samuel (1959)
* *A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.* - Tom Mitchell (1997)

In essence, "machine learning" is the idea that a computer can become better at performing a task, if given sufficient amounts of data to "learn" from (what we mean by "learn" will be discussed later). This data may or may not come with an "answer key", and could be given to the computer all at once or continuously throughout some time interval. 

On this note, we can definitively say that simply downloading new data onto your computer is NOT considered ML (your computer has not gotten any better at performing a certain task, right!). Hence, this concept of "learning" is more nuanced than simply creating/copying new data onto the machine; indeed, we will be seeing various ways a computer may "learn" throughout our study.

## 1.2 Why Use Machine Learning?

As a thought experiment, let us consider the example of creating an email spam filter. Here are the steps that one would follow if implementing a traditional programming technique:

1. Figure out some common characteristics of spam emails (certain words, phrases, email addresses, etc.)
2. Write a program that, upon detection of these characteristics, classifies an email as spam
3. Do some testing, add other characteristics you observe
4. Launch the program!

As you may have picked up, there are a few problems with this problem solving scheme:

* You may not pick up all possible characteristics of spam emails
* Your program will not work for **new types** of spam emails
* Your program will require continuous updates/improvements/maintenance forever

That sounds pretty annoying to me! In contrast, consider instead the following steps that a ML implementation would take:

1. Gather existing data (your observations of spam vs ham email)
2. Train a ML algorithm to detect spam emails based on this data
3. Evaluate/improve the model -> Launch!
4. Feed new data into the ML algorithm (algorithm will improve by itself with new data)

In this problem solving scheme, we do not have to do as much maintanence work since the ML algorithm will continue to become better as new data is fed into it (which is a process that can be automated).

Another field in which implementing ML is useful are those which are too complex (in terms of scalability) or cannot be described via an algorithm. Some examples are: image recognition, speech recognition, online customer support, product recommendations, etc.


As a third use, we can even use ML to learn ourselves! Given a large data set with many characteristics, we may wish to know which 2-3 are the most important for predicting something about that data set. A common example that is seen is that of housing prices. We can get data on various characteristics on many different houses on the market, and we wish to know which few characteristics have the most effect on the price of a house. Applying ML techniques to big data can help us discover patterns or trends that are not immediately apparant.

In summary, ML is very useful for:
* Problems for which traditional solutions would require a lot of maintanence and/or a lot of conditional statements
* Complex problems for which the traditional approach would not give a good solution
* Fluctuating environments, where your program would need to continually adapt to new data
* Gaining insight into your data sets

## 1.3 - Example Applications of ML

Here is a list of a few real-world applications of ML, of which we will be learning the techniques throughout our learning journey:

* Analyzing images of products on a production line to classify them
* Detecting brain tumors
* Flagging offensive/off-topic comments in online forums/chats
* Summarizing long documents
* Forecasting a company's revenue
* Creating voice command features in phones, call centers, etc.
* Detecting credit card fraud
* Representing complex, high-dimensional data in a clear and insightful diagram
* Creating a recommendation system for books, movies, restaurants, etc.
* Creating bots for games, web crawling, etc.


## 1.4 - Types of ML Systems

We classify ML systems into three broad categories:

* Whether or not they are trained with human supervision (supervised, semi-supervised, unsupervised, reinforcement learning)
* Whether or not they can learn continuously (online learning vs. batch learning)
* Whether they compare new data to old data OR detect patterns in input data and creating a predictive model (instance-based learning vs. model-based learning






















