---
title: Roadmap for Classification Tasks
author: Michael Fuchs
date: '2020-02-19'
slug: roadmap-for-classification-tasks
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


Another big chapter from the supervised machine learning area comes to an end. In the past 4 months I wrote in detail about the functionality and use of the most common classification algorithms within data science.

Analogous to my post ["Roadmap for Regression Analysis"](https://michael-fuchs-python.netlify.com/2019/10/14/roadmap-for-regression-analysis/), I will give again an overview of the handling of classification tasks.



# 2 Roadmap for Classification Tasks

## 2.1 Data pre-processing

![](/post/2020-02-19-roadmap-for-classification-tasks_files/p40p1.png)



Here are the links to the individual topics:

+ [Dealing with outliers](https://michael-fuchs-python.netlify.com/2019/08/20/dealing-with-outliers/)
+ [Handling Missing Values](https://michael-fuchs-python.netlify.com/2019/03/18/dealing-with-missing-values/)
+ [Feature Encoding](https://michael-fuchs-python.netlify.com/2019/06/16/types-of-encoder/)
+ [Feature Scaling](https://michael-fuchs-python.netlify.com/2019/08/31/feature-scaling-with-scikit-learn/)
+ [Dealing with imbalanced classes](https://michael-fuchs-python.netlify.com/2020/01/16/dealing-with-imbalanced-classes/)



## 2.2 Feature Selection Methods


![](/post/2020-02-19-roadmap-for-classification-tasks_files/p40p2.png)


Here are the links to the individual topics:


Filter methods:

+ [Dealing with highly correlated features](https://michael-fuchs-python.netlify.com/2019/07/28/dealing-with-highly-correlated-features/)
+ [Dealing with constant features](https://michael-fuchs-python.netlify.com/2019/08/09/dealing-with-constant-and-duplicate-features/)
+ [Dealing with duplicate features](https://michael-fuchs-python.netlify.com/2019/08/09/dealing-with-constant-and-duplicate-features/)



Wrapper methods:

+ [SelectKBest](https://michael-fuchs-python.netlify.com/2020/01/31/feature-selection-methods-for-classification-tasks/)
+ [Step Forward Feature Selection](https://michael-fuchs-python.netlify.com/2020/01/31/feature-selection-methods-for-classification-tasks/)
+ [Backward Elimination](https://michael-fuchs-python.netlify.com/2020/01/31/feature-selection-methods-for-classification-tasks/)
+ [Recursive Feature Elimination (RFE)](https://michael-fuchs-python.netlify.com/2020/01/31/feature-selection-methods-for-classification-tasks/)
+ [Exhaustive Feature Selection](https://michael-fuchs-python.netlify.com/2020/01/31/feature-selection-methods-for-classification-tasks/)




## 2.3 Algorithms

### 2.3.1 Classification Algorithms


![](/post/2020-02-19-roadmap-for-classification-tasks_files/p40p3.png)

Here are the links to the individual topics:

+ [Logistic Regression](https://michael-fuchs-python.netlify.com/2019/10/31/introduction-to-logistic-regression/)
+ [Support Vector Machines](https://michael-fuchs-python.netlify.com/2019/11/08/introduction-to-support-vector-machines/)
+ [Perceptron](https://michael-fuchs-python.netlify.com/2019/11/14/introduction-to-perceptron-algorithm/)
+ [SGD Classifier](https://michael-fuchs-python.netlify.com/2019/11/11/introduction-to-sgd-classifier/)
+ [OvO and OvR Classifier](https://michael-fuchs-python.netlify.com/2019/11/13/ovo-and-ovr-classifier/)
+ [Softmax Regression](https://michael-fuchs-python.netlify.com/2019/11/15/multinomial-logistic-regression/)
+ [Decision Trees](https://michael-fuchs-python.netlify.com/2019/11/30/introduction-to-decision-trees/)
+ [Naive Bayes Classifier](https://michael-fuchs-python.netlify.com/2019/12/15/introduction-to-naive-bayes-classifier/)
+ [K Nearest Neighbor Classifier](https://michael-fuchs-python.netlify.com/2019/12/27/introduction-to-knn-classifier/)
+ [Bagging](https://michael-fuchs-python.netlify.app/2020/03/07/ensemble-modeling-bagging/)
+ [Boosting](https://michael-fuchs-python.netlify.app/2020/03/26/ensemble-modeling-boosting/)
+ [XGBoost](https://michael-fuchs-python.netlify.app/2020/04/01/ensemble-modeling-xgboost/#xgboost-for-classification)
+ [Stacking](https://michael-fuchs-python.netlify.app/2020/04/24/ensemble-modeling-stacking/)
+ [Stacking with scikit-learn](https://michael-fuchs-python.netlify.app/2020/04/29/stacking-with-scikit-learn/)
+ [Voting](https://michael-fuchs-python.netlify.app/2020/05/05/ensemble-modeling-voting//)


**Notes on the special classifiers:**

As described in the [SGD Classifier](https://michael-fuchs-python.netlify.com/2019/11/11/introduction-to-sgd-classifier/) post, this is not a classifier. It's a linear classifier optimized by the Stochastic Gradient Descent.


With the [One-vs-One and One-vs-Rest](https://michael-fuchs-python.netlify.com/2019/11/13/ovo-and-ovr-classifier/) method it is possible to make binary classifiers multiple.


**Notes on ensemble methods:**

Depending on the underlying problem with the predictions I choose the following ensemble method:

+ Bagging: Decrease Variance
+	Boosting: Decrease Bias
+	Stacking: Improve Predictions


### 2.3.2 Classification with Neural Networks

Of course, in addition to traditional classification algorithms, neural networks can be used to solve classification problems. 

Here again are the links to the respective publications: 

+ [Multi-layer Perceptron Classifier (MLPClassifier)](https://michael-fuchs-python.netlify.app/2021/02/03/nn-multi-layer-perceptron-classifier-mlpclassifier/)
+ [Artificial Neural Network for binary Classification](https://michael-fuchs-python.netlify.app/2021/02/16/nn-artificial-neural-network-for-binary-classification/)
+ [Artificial Neural Network for Multi-Class Classfication](https://michael-fuchs-python.netlify.app/2021/02/23/nn-artificial-neural-network-for-multi-class-classfication/)


### 2.3.3 AutoML

The use of automated machine learning libraries is becoming increasingly popular. 
Here is a guide on how classification problems can be solved with PyCaret:

+ [AutoML using PyCaret - Classification](https://michael-fuchs-python.netlify.app/2022/01/01/automl-using-pycaret-classification/)



# 3 Conclusion

The methods and algorithms shown in the overviews are described in detail in the respective publications with regard to theory and practical application. Just click on the respective link.


