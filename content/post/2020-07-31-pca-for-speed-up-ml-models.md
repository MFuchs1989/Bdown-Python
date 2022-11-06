---
title: PCA for speed up ML models
author: Michael Fuchs
date: '2020-07-31'
slug: pca-for-speed-up-ml-models
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

As already announced in post about ["PCA"](https://michael-fuchs-python.netlify.app/2020/07/22/principal-component-analysis-pca/), we now come to the second main application of a PCA: Principal Component Analysis for speed up machine learning models.


For this post the dataset *MNIST* from the statistic platform ["Kaggle"](https://www.kaggle.com) was used. A copy of the record is available at <https://drive.google.com/open?id=1Bfquk0uKnh6B3Yjh2N87qh0QcmLokrVk>.



# 2 Loading the libraries and the dataset



```r
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

import pickle as pk
```


```r
mnist = pd.read_csv('mnist_train.csv')
mnist
```

![](/post/2020-07-31-pca-for-speed-up-ml-models_files/p57p1.png)


```r
mnist['label'].value_counts().T
```

![](/post/2020-07-31-pca-for-speed-up-ml-models_files/p57p2.png)


# 3 LogReg

If you want to know how the algorithm of the logistic regression works exactly see ["this post"](https://michael-fuchs-python.netlify.app/2019/10/31/introduction-to-logistic-regression/) of mine.



```r
x = mnist.drop(['label'], axis=1)
y = mnist['label']

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)
```



```r
sc=StandardScaler()

# Fit on training set only!
sc.fit(trainX)

# Apply transform to both the training set and the test set.
trainX_scaled = sc.transform(trainX)
testX_scaled = sc.transform(testX)
```


```r
# all parameters not specified are set to their defaults

logReg = LogisticRegression()
```



```r
import time

start = time.time()

print(logReg.fit(trainX_scaled, trainY))

end = time.time()
print()
print('Calculation time: ' + str(end - start) + ' seconds')
```

![](/post/2020-07-31-pca-for-speed-up-ml-models_files/p57p3.png)



```r
y_pred = logReg.predict(testX_scaled)
```



```r
print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))
```

![](/post/2020-07-31-pca-for-speed-up-ml-models_files/p57p4.png)


# 4 LogReg with PCA

## 4.1 PCA with 95% variance explanation

Notice the code below has .95 for the number of components parameter. It means that scikit-learn choose the minimum number of principal components such that 95% of the variance is retained.



```r
pca = PCA(.95)
```


```r
# Fitting PCA on the training set only
pca.fit(trainX_scaled)
```

You can find out how many components PCA choose after fitting the model using pca.n_components_ . In this case, 95% of the variance amounts to 326 principal components.


```r
pca.n_components_
```

![](/post/2020-07-31-pca-for-speed-up-ml-models_files/p57p5.png)


```r
trainX_pca = pca.transform(trainX_scaled)
testX_pca = pca.transform(testX_scaled)
```


```r
# all parameters not specified are set to their defaults

logReg = LogisticRegression()
```


```r
import time

start = time.time()

print(logReg.fit(trainX_pca, trainY))

end = time.time()
print()
print('Calculation time: ' + str(end - start) + ' seconds')
```

![](/post/2020-07-31-pca-for-speed-up-ml-models_files/p57p6.png)


```r
y_pred = logReg.predict(testX_pca)
```



```r
print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))
```

![](/post/2020-07-31-pca-for-speed-up-ml-models_files/p57p7.png)


Now let's try 80% variance explanation.


## 4.2 PCA with 80% variance explanation



```r
pca = PCA(.80)
```


```r
# Fitting PCA on the training set only
pca.fit(trainX_scaled)
```


```r
pca.n_components_
```

![](/post/2020-07-31-pca-for-speed-up-ml-models_files/p57p8.png)


```r
trainX_pca = pca.transform(trainX_scaled)
testX_pca = pca.transform(testX_scaled)
```


```r
# all parameters not specified are set to their defaults

logReg = LogisticRegression()
```


```r
import time

start = time.time()

print(logReg.fit(trainX_pca, trainY))

end = time.time()
print()
print('Calculation time: ' + str(end - start) + ' seconds')
```

![](/post/2020-07-31-pca-for-speed-up-ml-models_files/p57p9.png)



```r
y_pred = logReg.predict(testX_pca)
```



```r
print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))
```

![](/post/2020-07-31-pca-for-speed-up-ml-models_files/p57p10.png)


## 4.3 Summary


As we can see in the overview below, not only has the training time has been reduced by PCA, but the prediction accuracy of the trained model has also increased.


![](/post/2020-07-31-pca-for-speed-up-ml-models_files/p57s1.png)



# 5 Export PCA to use in another program

For a nice example we create the following artificial data set:



```r
df = pd.DataFrame({'Col1': [5464, 2484, 846546],
                   'Col2': [5687,78455,845684],
                   'Col3': [8754,7686,4585],
                   'Col4': [49864, 89481, 92254],
                   'Col5': [22168, 63689, 5223]})
df
```

![](/post/2020-07-31-pca-for-speed-up-ml-models_files/p57p11.png)



```r
df['Target'] = df.sum(axis=1)
df
```

![](/post/2020-07-31-pca-for-speed-up-ml-models_files/p57p12.png)


**Note:** We skip the scaling step and the train test split here. In the following, we only want to train the algorithms as well as their storage and use in other programs. Validation is also not a focus here.



```r
X = df.drop(['Target'], axis=1)
Y = df['Target']
```


```r
pca = PCA(n_components=2)
```


```r
pca.fit(X)
result = pca.transform(X)
```


```r
components = pd.DataFrame(pca.components_, columns = X.columns, index=[1, 2])
components = components.T
components.columns = ['Principle_Component_1', 'Principle_Component_2']
components
```

![](/post/2020-07-31-pca-for-speed-up-ml-models_files/p57p13.png)



```r
# all parameters not specified are set to their defaults

logReg = LogisticRegression()

logReg.fit(result, Y)
```



```r
pk.dump(pca, open("pca.pkl","wb"))
pk.dump(logReg, open("logReg.pkl","wb"))
```

The models are saved in the corresponding path and should look like this:

![](/post/2020-07-31-pca-for-speed-up-ml-models_files/p57s2.png)

In order to show that the principal component analysis has been saved with the correct weightings and reloaded accordingly, we create exactly the same artificial data set (only without target variable) as at the beginning of this exercise.


```r
df_new = pd.DataFrame({'Col1': [5464, 2484, 846546],
                   'Col2': [5687,78455,845684],
                   'Col3': [8754,7686,4585],
                   'Col4': [49864, 89481, 92254],
                   'Col5': [22168, 63689, 5223]})
df_new
```

![](/post/2020-07-31-pca-for-speed-up-ml-models_files/p57p14.png)

Now we reload the saved models:



```r
pca_reload = pk.load(open("pca.pkl",'rb'))
logReg_reload = pk.load(open("logReg.pkl",'rb'))
```


```r
result_new = pca_reload .transform(df_new)
```


```r
components = pd.DataFrame(pca.components_, columns = X.columns, index=[1, 2])
components = components.T
components.columns = ['Principle_Component_1', 'Principle_Component_2']
components
```

![](/post/2020-07-31-pca-for-speed-up-ml-models_files/p57p15.png)


We see that the weights have been adopted, as we can compare this output with the first transformation (see above).



```r
y_pred = logReg_reload.predict(result_new)
y_pred
```

![](/post/2020-07-31-pca-for-speed-up-ml-models_files/p57p16.png)

Last but not least we'll add the predicted values to our original dataframe.


```r
df_y_pred = pd.DataFrame(y_pred)
df_result_new = pd.DataFrame(result_new)

result_new = pd.concat([df_result_new, df_y_pred], axis=1)
result_new.columns = ['Principle_Component_1', 'Principle_Component_2', 'Prediction']
result_new
```

![](/post/2020-07-31-pca-for-speed-up-ml-models_files/p57p17.png)




# 6 Conclusion

In this post, I showed how much a PCA can improve the training speed of machine learning algorithms and also increase the quality of the forecast.
I also showed how the weights of principal component analysis can be saved and reused for future pre-processing steps.

