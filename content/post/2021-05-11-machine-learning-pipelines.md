---
title: Machine Learning Pipelines
author: Michael Fuchs
date: '2021-05-11'
slug: machine-learning-pipelines
categories: []
tags: []
output:
  blogdown::html_page:
    toc: true
    toc_depth: 5
---


# 1 Introduction


![](/post/2021-05-11-machine-learning-pipelines_files/p121s1.png)


Some time ago I had written the post [The Data Science Process (CRISP-DM)](https://michael-fuchs-python.netlify.app/2020/08/21/the-data-science-process-crisp-dm/), which was about the correct development of Machine Learning algorithms. As you have seen here, this is quite a time-consuming matter if done correctly.

In order to quickly check which algorithm fits the data best, it is recommended to use machine learning pipelines. Once you have found a promising algorithm, you can start fine tuning with it and go through the process as described [here](https://michael-fuchs-python.netlify.app/2020/08/21/the-data-science-process-crisp-dm/#data-science-best-practice-guidlines-for-ml-model-development).  



For this post the dataset *bird* from the statistic platform ["Kaggle"](https://www.kaggle.com) was used. You can download it from my ["GitHub Repository"](https://github.com/MFuchs1989/Datasets-and-Miscellaneous/tree/main/datasets).



# 2 Loading the libraries and classes



```r
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from sklearn.decomposition import PCA


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score
```



```r
class Color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
```



# 3 Loading the data



```r
bird_df = pd.read_csv('bird.csv').dropna()
bird_df
```

![](/post/2021-05-11-machine-learning-pipelines_files/p121p1.png)


Description of predictors:

+ Length and Diameter of Humerus
+ Length and Diameter of Ulna
+ Length and Diameter of Femur
+ Length and Diameter of Tibiotarsus
+ Length and Diameter of Tarsometatarsus



```r
bird_df['type'].value_counts()
```

![](/post/2021-05-11-machine-learning-pipelines_files/p121p2.png)


Description of the target variable:

+ SW: Swimming Birds
+ W: Wading Birds
+ T: Terrestrial Birds
+ R: Raptors
+ P: Scansorial Birds
+ SO: Singing Birds



```r
bird_df['type'].nunique()
```

![](/post/2021-05-11-machine-learning-pipelines_files/p121p3.png)




```r
x = bird_df.drop(['type', 'id'], axis=1)
y = bird_df['type']

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)
```




# 4 ML Pipelines


## 4.1 A simple Pipeline


Let's start with a simple pipeline.

In the following, I would like to perform a classification of bird species using [Logistic Regression](https://michael-fuchs-python.netlify.app/2019/10/31/introduction-to-logistic-regression/). For this purpose, the data should be scaled beforehand using the StandardScaler of scikit-learn.


**Creation of the pipeline:**


```r
pipe_lr = Pipeline([
    ('ss', StandardScaler()),
    ('lr', LogisticRegression())
    ])
```

**Fit and Evaluate the Pipeline:**


```r
pipe_lr.fit(trainX, trainY)
```



```r
y_pred = pipe_lr.predict(testX)


print('Test Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))
```

![](/post/2021-05-11-machine-learning-pipelines_files/p121p4.png)


OK, .75% not bad. Let's see if we can improve the result by choosing a different scaler. 




## 4.2 Determination of the best Scaler



### 4.2.1 Creation of the Pipeline


```r
pipe_lr_wo = Pipeline([
    ('lr', LogisticRegression())
    ])

pipe_lr_ss = Pipeline([
    ('ss', StandardScaler()),
    ('lr', LogisticRegression())
    ])

pipe_lr_mms = Pipeline([
    ('mms', MinMaxScaler()),
    ('lr', LogisticRegression())
    ])

pipe_lr_rs = Pipeline([
    ('rs', RobustScaler()),
    ('lr', LogisticRegression())
    ])
```



### 4.2.2 Creation of a Pipeline Dictionary

To be able to present the later results better, I always create a suitable dictionary at this point. 


```r
pipe_dic = {
    0: 'LogReg wo scaler',
    1: 'LogReg with StandardScaler',
    2: 'LogReg with MinMaxScaler',
    3: 'LogReg with RobustScaler',
    }

pipe_dic
```

![](/post/2021-05-11-machine-learning-pipelines_files/p121p5.png)



### 4.2.3 Fit the Pipeline

To be able to fit the pipelines, I first need to group the pipelines into a list:


```r
pipelines = [pipe_lr_wo, pipe_lr_ss, pipe_lr_mms, pipe_lr_rs]
```


Now we are going to fit the created pipelines:


```r
for pipe in pipelines:
    pipe.fit(trainX, trainY)
```



### 4.2.4 Evaluate the Pipeline



```r
for idx, val in enumerate(pipelines):
    print('%s pipeline Test Accuracy: %.2f' % (pipe_dic[idx], accuracy_score(testY, val.predict(testX))))
```

![](/post/2021-05-11-machine-learning-pipelines_files/p121p6.png)



We can also use the .score function:


```r
for idx, val in enumerate(pipelines):
    print('%s pipeline Test Accuracy: %.2f' % (pipe_dic[idx], val.score(testX, testY)))
```

![](/post/2021-05-11-machine-learning-pipelines_files/p121p7.png)


I always like to have the results displayed in a dataframe so I can sort and filter:



```r
result = []

for idx, val in enumerate(pipelines):
    result.append(accuracy_score(testY, val.predict(testX)))

    
result_df = pd.DataFrame(list(pipe_dic.items()),columns = ['Idx','Estimator'])

# Add Test Accuracy to result_df
result_df['Test_Accuracy'] = result
# print result_df
result_df 
```

![](/post/2021-05-11-machine-learning-pipelines_files/p121p8.png)


Let's take a look at our best model:



```r
best_model = result_df.sort_values(by='Test_Accuracy', ascending=False)
best_model
```

![](/post/2021-05-11-machine-learning-pipelines_files/p121p9.png)




```r
print(best_model['Estimator'].iloc[0] +
      ' Classifier has the best Test Accuracy of ' + 
      str(round(best_model['Test_Accuracy'].iloc[0], 2)) + '%')
```

![](/post/2021-05-11-machine-learning-pipelines_files/p121p10.png)

Or the print statement still a little bit spiffed up:


```r
print(Color.RED + best_model['Estimator'].iloc[0] + Color.END +
      ' Classifier has the best Test Accuracy of ' + 
      Color.GREEN + Color.BOLD + str(round(best_model['Test_Accuracy'].iloc[0], 2)) + '%')
```

![](/post/2021-05-11-machine-learning-pipelines_files/p121p11.png)



## 4.3 Determination of the best Estimator


Let's try this time with different estimators to improve the result.


### 4.3.1 Creation of the Pipeline



```r
pipe_lr = Pipeline([
    ('ss1', StandardScaler()),
    ('lr', LogisticRegression())
    ])

pipe_svm_lin = Pipeline([
    ('ss2', StandardScaler()),
    ('svm_lin', SVC(kernel='linear'))
    ])

pipe_svm_sig = Pipeline([
    ('ss3', StandardScaler()),
    ('svm_sig', SVC(kernel='sigmoid'))
    ])


pipe_knn = Pipeline([
    ('ss4', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=7))
    ])

pipe_dt = Pipeline([
    ('ss5', StandardScaler()),
    ('dt', DecisionTreeClassifier())
    ])

pipe_rf = Pipeline([
    ('ss6', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=100))
    ])
```


### 4.3.2 Creation of a Pipeline Dictionary


```r
pipe_dic = {
    0: 'lr',
    1: 'svm_lin',
    2: 'svm_sig',
    3: 'knn',
    4: 'dt',
    5: 'rf'
    }

pipe_dic
```

![](/post/2021-05-11-machine-learning-pipelines_files/p121p12.png)


### 4.3.3 Fit the Pipeline



```r
pipelines = [pipe_lr, pipe_svm_lin, pipe_svm_sig, pipe_knn, pipe_dt, pipe_rf]
```



```r
for pipe in pipelines:
    pipe.fit(trainX, trainY)
```


### 4.3.4 Evaluate the Pipeline



```r
for idx, val in enumerate(pipelines):
    print('%s pipeline Test Accuracy: %.2f' % (pipe_dic[idx], accuracy_score(testY, val.predict(testX))))
```

![](/post/2021-05-11-machine-learning-pipelines_files/p121p13.png)




```r
result = []

for idx, val in enumerate(pipelines):
    result.append(accuracy_score(testY, val.predict(testX)))

    
result_df = pd.DataFrame(list(pipe_dic.items()),columns = ['Idx','Estimator'])

# Add Test Accuracy to result_df
result_df['Test_Accuracy'] = result
# print result_df
result_df 
```

![](/post/2021-05-11-machine-learning-pipelines_files/p121p14.png)




```r
best_model = result_df.sort_values(by='Test_Accuracy', ascending=False)
best_model
```

![](/post/2021-05-11-machine-learning-pipelines_files/p121p15.png)




```r
print(Color.RED + best_model['Estimator'].iloc[0] + Color.END +
      ' Classifier has the best Test Accuracy of ' + 
      Color.GREEN + Color.BOLD + str(round(best_model['Test_Accuracy'].iloc[0], 2)) + '%')
```

![](/post/2021-05-11-machine-learning-pipelines_files/p121p16.png)

Done. The linear support vector classifier has improved the accuracy. 




## 4.4 ML Pipelines with further Components


At this point you can now play wonderfully. You can add different scalers to the Estimators or even try including a [PCA](https://michael-fuchs-python.netlify.app/2020/07/22/principal-component-analysis-pca/). 

I use the same pipeline as in the previous example and add a PCA with n_components=2 to the estimators.


```r
pipe_lr = Pipeline([
    ('ss1', StandardScaler()),
    ('pca1', PCA(n_components=2)),
    ('lr', LogisticRegression())
    ])

pipe_svm_lin = Pipeline([
    ('ss2', StandardScaler()),
    ('pca2', PCA(n_components=2)),
    ('svm_lin', SVC(kernel='linear'))
    ])

pipe_svm_sig = Pipeline([
    ('ss3', StandardScaler()),
    ('pca3', PCA(n_components=2)),
    ('svm_sig', SVC(kernel='sigmoid'))
    ])


pipe_knn = Pipeline([
    ('ss4', StandardScaler()),
    ('pca4', PCA(n_components=2)),
    ('knn', KNeighborsClassifier(n_neighbors=7))
    ])

pipe_dt = Pipeline([
    ('ss5', StandardScaler()),
    ('pca5', PCA(n_components=2)),
    ('dt', DecisionTreeClassifier())
    ])

pipe_rf = Pipeline([
    ('ss6', StandardScaler()),
    ('pca6', PCA(n_components=2)),
    ('rf', RandomForestClassifier(n_estimators=100))
    ])
```


```r
pipe_dic = {
    0: 'lr',
    1: 'svm_lin',
    2: 'svm_sig',
    3: 'knn',
    4: 'dt',
    5: 'rf'
    }
```


```r
pipelines = [pipe_lr, pipe_svm_lin, pipe_svm_sig, pipe_knn, pipe_dt, pipe_rf]
```


```r
for pipe in pipelines:
    pipe.fit(trainX, trainY)
```



```r
for idx, val in enumerate(pipelines):
    print('%s pipeline Test Accuracy: %.2f' % (pipe_dic[idx], accuracy_score(testY, val.predict(testX))))
```

![](/post/2021-05-11-machine-learning-pipelines_files/p121p17.png)



```r
result = []

for idx, val in enumerate(pipelines):
    result.append(accuracy_score(testY, val.predict(testX)))

    
result_df = pd.DataFrame(list(pipe_dic.items()),columns = ['Idx','Estimator'])

# Add Test Accuracy to result_df
result_df['Test_Accuracy'] = result
# print result_df
result_df 
```

![](/post/2021-05-11-machine-learning-pipelines_files/p121p18.png)



```r
best_model = result_df.sort_values(by='Test_Accuracy', ascending=False)
best_model
```

![](/post/2021-05-11-machine-learning-pipelines_files/p121p19.png)




```r
print(Color.RED + best_model['Estimator'].iloc[0] + Color.END +
      ' Classifier has the best Test Accuracy of ' + 
      Color.GREEN + Color.BOLD + str(round(best_model['Test_Accuracy'].iloc[0], 2)) + '%')
```

![](/post/2021-05-11-machine-learning-pipelines_files/p121p20.png)

The use of a PCA has not worked out any improvement. We can therefore fall back on the linear Support Vector Classifier at this point and try to improve the result again with Fine Tuning. 


# 5 Conclusion

In this post, I showed how to use machine learning pipelines to quickly and efficiently run different scenarios to get a first impression of which algorithm fits my data best. 








