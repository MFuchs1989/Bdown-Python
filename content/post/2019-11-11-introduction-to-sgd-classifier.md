---
title: Introduction to SGD Classifier
author: Michael Fuchs
date: '2019-11-11'
slug: introduction-to-sgd-classifier
categories:
  - R
tags:
  - R Markdown
output:
  blogdown::html_page:
    toc: true
    toc_depth: 5
---
 


# 1 Introduction

The name Stochastic Gradient Descent - Classifier (SGD-Classifier) might mislead some user to think that SGD is a classifier. But that's not the case! SGD Classifier is a linear classifier (SVM, logistic regression, a.o.) optimized by the SGD.
These are two different concepts. While SGD is a optimization method, Logistic Regression or linear Support Vector Machine is a machine learning algorithm/model. You can think of that a machine learning model defines a loss function, and the optimization method minimizes/maximizes it. 


For this post the dataset *Run or Walk* from the statistic platform ["Kaggle"](https://www.kaggle.com/) was used. You can download it from my [GitHub Repository](https://github.com/MFuchs1989/Datasets-and-Miscellaneous/tree/main/datasets).


# 2 Background information on SGD Classifiers


**Gradient Descent**

First of all let's talk about Gradient descent in general.

![](/post/2019-11-11-introduction-to-sgd-classifier_files/p32s1.png)

In a nutshell gradient descent is used to minimize a cost function. 
Gradient descent is one of the most popular algorithms to perform optimization and by far the most common way to optimize neural networks. But we can also use these kinds of algorithms to optimize our linear classifier such as Logistic Regression and linear Support Vecotor Machines.

There are three well known types of gradient decent:

1. Batch gradient descent
2. Stochastic gradient descent
3. Mini-batch gradient descent

Batch gradient descent computes the gradient using the whole dataset to find the minimum located in it's basin of attraction.

Stochastic gradient descent (SGD) computes the gradient using a single sample. 

Mini-batch gradient descent finally takes the best of both worlds and performs an update for every mini-batch of n training examples.


**Why do we use SGD classifiers, when we already have linear classifiers such as LogReg or SVM?**

As we can read from the previous text, SGD allows minibatch (online/out-of-core) learning. Therefore, it makes sense to use SGD for large scale problems where it's very efficient.

The minimum of the cost function of Logistic Regression cannot be calculated directly, so we try to minimize it via Stochastic Gradient Descent, also known as Online Gradient Descent. In this process we descend along the cost function towards its minimum (please have a look at the diagram above) for each training observation we encounter.

Another reason to use SGD Classifier is that SVM or logistic regression will not work if you cannot keep the record in RAM. However, SGD Classifier continues to work.

# 3 Loading the libraries and the data


```r
import numpy as np
import pandas as pd

# For chapter 4
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# For chapter 5
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import time

# For chapter 6
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
```



```r
run_walk = pd.read_csv("path/to/file/run_or_walk.csv")
```


```r
run_walk.head()
```

![](/post/2019-11-11-introduction-to-sgd-classifier_files/p32p1.png)


# 4 Data pre-processing

In the first step we split up the data set for the model training. Columns 'date', 'time' and 'username' are not required for further analysis.


```r
x = run_walk.drop(['date', 'time', 'username', 'activity'], axis=1)
y = run_walk['activity']

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)
```

It is particularly important to scale the features when using the SGD Classifier. You can read about how scaling works with Scikit-learn in the following post of mine: ["Feature Scaling with Scikit-Learn"](https://michael-fuchs-python.netlify.com/2019/08/31/feature-scaling-with-scikit-learn/)


```r
scaler = StandardScaler()
scaler.fit(trainX)
trainX = scaler.transform(trainX)
testX = scaler.transform(testX)
```


# 5 SGD-Classifier

As already mentioned above SGD-Classifier is a Linear classifier with SGD training. Which linear classifier is used is determined with the hypter parameter loss.
So, if I write *clf = SGDClassifier(loss='hinge')* it is an implementation of Linear SVM and if I write *clf = SGDClassifier(loss='log')* it is an implementation of Logisitic regression. 

Let's see how both types work:

## 5.1 Logistic Regression with SGD training


```r
clf = SGDClassifier(loss="log", penalty="l2")
clf.fit(trainX, trainY)
```

![](/post/2019-11-11-introduction-to-sgd-classifier_files/p32p2.png)


```r
y_pred = clf.predict(testX)

print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))
```

![](/post/2019-11-11-introduction-to-sgd-classifier_files/p32p3.png)

By default the maximum number of passes over the training data (aka epochs) is set to 1,000.
Let's see what influence this parameter has on our score (accuracy):


```r
n_iters = [5, 10, 20, 50, 100, 1000]
scores = []
for n_iter in n_iters:
    clf = SGDClassifier(loss="log", penalty="l2", max_iter=n_iter)
    clf.fit(trainX, trainY)
    scores.append(clf.score(testX, testY))
  
plt.title("Effect of n_iter")
plt.xlabel("n_iter")
plt.ylabel("score")
plt.plot(n_iters, scores) 
```

![](/post/2019-11-11-introduction-to-sgd-classifier_files/p32p4.png)


## 5.2 Linear SVM with SGD training

Now we do the same calculation for the linear model of the SVM.


```r
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(trainX, trainY)
```

![](/post/2019-11-11-introduction-to-sgd-classifier_files/p32p5.png)





```r
y_pred = clf.predict(testX)

print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))
```

![](/post/2019-11-11-introduction-to-sgd-classifier_files/p32p6.png)

The accuracy is a little bit less.

Let's take another look at the influence of the number of iterations:


```r
n_iters = [5, 10, 20, 50, 100, 1000]
scores = []
for n_iter in n_iters:
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=n_iter)
    clf.fit(trainX, trainY)
    scores.append(clf.score(testX, testY))
  
plt.title("Effect of n_iter")
plt.xlabel("n_iter")
plt.ylabel("score")
plt.plot(n_iters, scores)
```

![](/post/2019-11-11-introduction-to-sgd-classifier_files/p32p7.png)



If you look at the training time, it becomes clear how much faster the SGD classifier works compared to the linear SVM:


```r
start = time.time()
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(trainX, trainY)
stop = time.time()
print(f"Training time for linear SVM with SGD training: {stop - start}s")

start = time.time()
clf = SVC(kernel='linear')
clf.fit(trainX, trainY)
stop = time.time()
print(f"Training time for linear SVM without SGD training: {stop - start}s")
```

![](/post/2019-11-11-introduction-to-sgd-classifier_files/p32p8.png)


# 6 Model improvement

## 6.1 Performance comparison of the different linear models


Let's take a look at the performance of the different linear classifiers


```r
losses = ["hinge", "log", "modified_huber", "perceptron", "squared_hinge"]
scores = []
for loss in losses:
    clf = SGDClassifier(loss=loss, penalty="l2", max_iter=1000)
    clf.fit(trainX, trainY)
    scores.append(clf.score(testX, testY))
  
plt.title("Effect of loss")
plt.xlabel("loss")
plt.ylabel("score")
x = np.arange(len(losses))
plt.xticks(x, losses)
plt.plot(x, scores) 
```

![](/post/2019-11-11-introduction-to-sgd-classifier_files/p32p9.png)


It becomes clear that 'hinge' (which stands for the use of a linear SVM) gives the best score and the use of the perceptron gives the worst value.



## 6.2 GridSearch

We use the popular GridSearch method to find the most suitable hyperparameters.


```r
params = {
    "loss" : ["hinge", "log", "squared_hinge", "modified_huber", "perceptron"],
    "alpha" : [0.0001, 0.001, 0.01, 0.1],
    "penalty" : ["l2", "l1", "elasticnet", "none"],
}

clf = SGDClassifier(max_iter=1000)
grid = GridSearchCV(clf, param_grid=params, cv=10)


grid.fit(trainX, trainY)

print(grid.best_params_) 
```

![](/post/2019-11-11-introduction-to-sgd-classifier_files/p32p10.png)



```r
grid_predictions = grid.predict(testX) 

print('Accuracy: {:.2f}'.format(accuracy_score(testY, grid_predictions)))
```

![](/post/2019-11-11-introduction-to-sgd-classifier_files/p32p11.png)


Although the accuracy could not be increased further, we get the confirmation that hinge (aka linear SVM) with the parameters shown above is the best choice.


# 7 Conclusion

In this post we covered gradient decent in general and how it can be used to improve linear classifiers. It was worked out why there is SGD Classifier at all and what advantages they have over simple linear models. Furthermore the functionality of hypterparameter tuning was explained.



