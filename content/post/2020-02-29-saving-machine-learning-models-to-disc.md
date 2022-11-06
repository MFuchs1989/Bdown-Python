---
title: Saving machine learning models to disc
author: Michael Fuchs
date: '2020-02-29'
slug: saving-machine-learning-models-to-disc
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


We have seen how to train and use different types of machine learning models. But how do we proceed when we have developed and trained a model with the desired performance? 
Due to the fact that the training of large machine learning models can sometimes take many hours, it is a good tip to save your trained models regularly so that you can access them later.

For this post the dataset *Iris* from the statistic platform ["Kaggle"](https://www.kaggle.com) was used. You can download it from my ["GitHub Repository"](https://github.com/MFuchs1989/Datasets-and-Miscellaneous/tree/main/datasets). 




# 2 Loading the libraries and the data


```r
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

#with joblib we'll safe our trained model
import joblib
```





```r
iris = pd.read_csv("Iris_Data.csv")
iris = iris[['sepal_length', 'sepal_width', 'species']]

iris
```

![](/post/2020-02-29-saving-machine-learning-models-to-disc_files/p41p1.png)


# 3 Visualization of the data


```r
g = sns.pairplot(iris, hue='species', markers='+')
plt.show()
```


![](/post/2020-02-29-saving-machine-learning-models-to-disc_files/p41p2.png)



# 4 Model training


```r
x = iris.drop('species', axis=1)
y = iris['species']

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)

clf = SVC(kernel='linear')
clf.fit(trainX, trainY)
```


![](/post/2020-02-29-saving-machine-learning-models-to-disc_files/p41p3.png)





```r
y_pred = clf.predict(testX)
```


```r
print('Accuracy: {:.2f}'.format(accuracy_score(testY, y_pred)))
print('Error rate: {:.2f}'.format(1 - accuracy_score(testY, y_pred)))
```


![](/post/2020-02-29-saving-machine-learning-models-to-disc_files/p41p4.png)



# 5 Safe a model to disc

Now we have trained our model (here linear SVM). It's time to safe the trained model to disc.



```r
# save the model to disk

filename = 'final_svm_model.sav'
joblib.dump(clf, filename)
```


![](/post/2020-02-29-saving-machine-learning-models-to-disc_files/p41p5.png)


Of course you can specify a different location under (here) filename.





# 6 Load a model from disc


The trained model can be reloaded at any later time.


```r
# load the model from disk

filename = 'final_svm_model.sav'
loaded_model = joblib.load(filename)
```


Ok let's test the loaded model. In advance we trained the distinction between three types of flowers.
Let's see what prediction I get for values sepal_length & sepal_width of 4.0 each.



```r
my_flower = [[4.0, 4.0]]
```


```r
my_pred = loaded_model.predict(my_flower)
my_pred
```


![](/post/2020-02-29-saving-machine-learning-models-to-disc_files/p41p6.png)


# 7 Conclusion

A Jupyter notebook, or whatever IDE you like to use, crashes easily. It is therefore advisable to save your trained model regularly.


