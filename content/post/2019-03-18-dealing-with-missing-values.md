---
title: Dealing with missing values
author: Michael Fuchs
date: '2019-03-18'
slug: dealing-with-missing-values
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


In the real world, there is virtually no record that has no missing values. Dealing with missing values can be done differently. In the following several methods will be presented how to deal with them.


# 2 Loading the Libraries and the Data


```r
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.impute import KNNImputer

from sklearn.model_selection import train_test_split
import pickle as pk
import random
```


```r
df = pd.DataFrame({'Name': ['Anton', 'Moni', np.NaN, 'Renate', 'Justus'],
                   'Age': [32,22,62,np.NaN,18],
                   'Salary': [np.NaN, np.NaN,4500,2500,3800],
                   'Job': ['Student', np.NaN, 'Manager', 'Teacher', 'Student']})
df
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p1.png)


# 3 Checking for missing values



```r
df.isnull().sum()
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p2.png)



```r
def mv_overview_func(df):
    '''
    Gives an overview of the total number and percentage of missing values in a data set
    
    Args:
        df (DataFrame): Dataframe to which the function is to be applied
        
    Returns:
        Overview of the total number and percentage of missing values
    '''
    # Total missing values
    mis_val = df.isnull().sum()
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    # Data Types
    data_types = df.dtypes
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent, data_types], axis=1)
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values', 2 : 'Data Type'})
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    # Return the dataframe with missing information
    return mis_val_table_ren_columns
```


```r
mv_overview_func(df)
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p3.png)

# 4 Droping of Missing Values



```r
df_drop = df.copy()
df_drop
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p4.png)


All rows with minimum one NaN will be dropped:


```r
df_drop.dropna()
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p5.png)


All rows from the defined columns with a NaN will be dropped:



```r
df_drop.dropna(subset=['Name', 'Age'])
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p6.png)


# 5 Imputations

## 5.1 for **NUMERIC** Features

### 5.1.1 Replace np.NaN with specific values



```r
df_replace_1 = df.copy()
df_replace_1
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p7.png)


Missing values from only one column (here 'Name') are replaced:



```r
df_replace_1['Name'] = df_replace_1['Name'].fillna(0)
df_replace_1
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p8.png)


Missing values from the complete dataset will be replaced:


```r
df_replace_1.fillna(0, inplace=True)
df_replace_1
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p9.png)


Does that make sense? Probably not. So let's look at imputations that follow a certain logic.


### 5.1.2 Replace np.NaN with MEAN

A popular metric for replacing missing values is the use of mean. 




```r
df_replace_2 = df.copy()
df_replace_2
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p10.png)




```r
df_replace_2['Age'].mean()
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p11.png)


Here all missing values of the column 'Age' are replaced by their mean value.


```r
df_replace_2['Age'] = df_replace_2['Age'].fillna(df_replace_2['Age'].mean())
df_replace_2
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p12.png)


**scikit-learn - SimpleImputer**

Always keep in mind that you will need all the steps you take to prepare for model training to make predictions later. 

What do I mean by that exactly?
If the data set you have available for model training already has missing values, it is quite possible that future data sets for which predictions are to be made will also contain missing values. 
In order for the prediction model to work, these missing values must be replaced by metrics that were also used in the model training. 



```r
df_replace_3 = df.copy()
df_replace_3
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p13.png)



```r
imp_age_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

imp_age_mean.fit(df_replace_3[['Age']])
df_replace_3['Age'] = imp_age_mean.transform(df_replace_3[['Age']])
df_replace_3
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p14.png)


In the steps shown before, I used the .fit and .transform functions separately. If it's not about model training, you can also combine these two steps and save yourself another line of code. 


```r
df_replace_4 = df.copy()
df_replace_4
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p15.png)



```r
imp_age_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

df_replace_4['Age'] = imp_age_mean.fit_transform(df_replace_4[['Age']])
df_replace_4
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p16.png)


This way you can see which value is behind imp_age_mean concretely: 


```r
 imp_age_mean.statistics_
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p17.png)


### 5.1.3 Replace np.NaN with MEDIAN

Other metrics such as the median can also be used instead of missing values:


```r
df_replace_5 = df.copy()
df_replace_5
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p18.png)



```r
imp_age_median = SimpleImputer(missing_values=np.nan, strategy='median')

df_replace_5['Age'] = imp_age_median.fit_transform(df_replace_5[['Age']])
df_replace_5
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p19.png)



```r
imp_age_median.statistics_
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p20.png)


### 5.1.4 Replace np.NaN with most_frequent

For some variables, it makes sense to use the most frequently occurring value for NaNs instead of mean or median. 



```r
df_replace_6 = df.copy()
df_replace_6
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p21.png)




```r
imp_age_mfreq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

df_replace_6['Age'] = imp_age_mfreq.fit_transform(df_replace_6[['Age']])
df_replace_6
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p22.png)



Since there is no value in the variable 'Age' that occurs twice or more often, the lowest value is automatically taken. The same would apply if there were two equally frequent values.


## 5.2 for **CATEGORICAL** Features

### 5.2.1 Replace np.NaN with most_frequent

The most_frequent function can be used for numeric variables as well as categorical variables. 



```r
df_replace_7 = df.copy()
df_replace_7
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p23.png)




```r
imp_job_mfreq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

df_replace_7['Job'] = imp_job_mfreq.fit_transform(df_replace_7[['Job']])
df_replace_7
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p24.png)


Here we see that with a frequency of 2, the job 'student' is the most common, so this is used for the missing value here. 


```r
imp_job_mfreq.statistics_
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p25.png)


But what happens if we just don't have a most frequent value in a categorical column like in our example within the column 'Name'?



```r
df_replace_8 = df.copy()
df_replace_8
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p26.png)



```r
imp_name_mfreq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

df_replace_8['Name'] = imp_name_mfreq.fit_transform(df_replace_8[['Name']])
df_replace_8
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p27.png)


Again, the principle that the lowest value is used applies. In our example, this is the name Anton, since it begins with A and thus comes before all other names in the alphabet. 


### 5.2.2 Replace np.NaN with specific values


```r
df_replace_9 = df.copy()
df_replace_9
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p28.png)


However, we also have the option of using certain values:


```r
imp_job_const = SimpleImputer(missing_values=np.nan, 
                              strategy='constant',
                              fill_value='others')

df_replace_9['Job'] = imp_job_const.fit_transform(df_replace_9[['Job']])
df_replace_9
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p29.png)


## 5.3 for specific Values

Not only a certain kind of values like NaN values can be replaced, this is also possible with specific values. 


### 5.3.1 single values

For the following example we take the last version of the last used data set, here df_replace_9:



```r
df_replace_9
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p30.png)



```r
rep_job_const = SimpleImputer(missing_values='others', 
                              strategy='constant',
                              fill_value='not_in_scope')

df_replace_9['Job'] = rep_job_const.fit_transform(df_replace_9[['Job']])
df_replace_9
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p31.png)


As we can see, 'others' became 'not_in_scope'.


### 5.3.2 multiple values

Unfortunately, we cannot work with lists for multiple values. But with the use of the pipeline function it works. We use for our following example again the last state of the dataset 'df_replace_9':




```r
rep_pipe = Pipeline([('si1',SimpleImputer(missing_values = 'Manager', 
                                          strategy='constant',
                                          fill_value='not_relevant')),
                     ('si2', SimpleImputer(missing_values = 'Teacher', 
                                           strategy='constant', 
                                           fill_value='not_relevant'))])

df_replace_9['Job'] = rep_pipe.fit_transform(df_replace_9[['Job']])
df_replace_9
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p32.png)


In my opinion, however, this approach has two disadvantages. Firstly, the values used are not saved (so cannot be reused automatically) and secondly, this is a lot of code to write. With an if-else function you would be faster:


```r
def rep_func(col):

    if col == 'Student':
        return 'useless'
    if col == 'not_relevant':
        return 'useless'
    else:
        return 'useless'

df_replace_9['Job'] = df_replace_9['Job'].apply(rep_func)
df_replace_9
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p33.png)


But again, the values used cannot be easily reused. You would have to create your own dictionary. 


# 6 Further Imputation Methods

In the following chapter I would like to present some more imputation methods. 
They differ from the previous ones because they use different values instead of NaN values and not specific values as before. 

In some cases, this can lead to getting a little closer to the truth and thus improve the model training. 


## 6.1 with ffill

Here, the missing value is replaced by the preceding non-missing value. 


```r
df_replace_10 = df.copy()
df_replace_10
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p34.png)



```r
df_replace_10['Age'] = df_replace_10['Age'].fillna(method='ffill')
df_replace_10
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p35.png)


The ffill function also works for categorical variables. 


```r
df_replace_10['Job'] = df_replace_10['Job'].fillna(method='ffill')
df_replace_10
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p36.png)


The condition here is that there is a first value. Let's have a look at the column 'Salary'. Here we have two missing values right at the beginning. Here ffill does not work:



```r
df_replace_10['Salary'] = df_replace_10['Salary'].fillna(method='ffill')
df_replace_10
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p37.png)

## 6.2 with backfill

What does not work with ffill works with backfill. Backfill replaces the missing value with the upcoming non-missing value. 


```r
df_replace_10['Salary'] = df_replace_10['Salary'].fillna(method='backfill')
df_replace_10
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p38.png)

## 6.3 Note on this Chapter

Due to the fact that different values are used for the missing values, it is not possible to define a uniform metric that should replace future missing values from the column.

However, one can proceed as follows for a model training. You start by using the functions ffill or backfill and then calculate a desired metric of your choice (e.g. mean) and save it for future missing values from the respective column.

I will explain the just described procedure in chapter 8.2 in more detail.



# 7 KNNImputer

Here, the KNN algorithm is used to replace missing values. 
If you want to know how KNN works exactly, check out this post of mine: [Introduction to KNN Classifier.](https://michael-fuchs-python.netlify.app/2019/12/27/introduction-to-knn-classifier/) For this chapter, I have again created a sample data set:


```r
df_knn = pd.DataFrame({'Name': ['Anton', 'Moni', 'Sven', 'Renate', 'Justus', 
                                'Sarah', 'Jon', 'Alex', 'Jenny', 'Jo'],
                       'Age': [32,22,62,np.NaN,18,63,np.NaN,44,23,71],
                       'Salary': [4000, np.NaN,4500,2500,3800,5500,7000,np.NaN,4800,3700]})
df_knn
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p39.png)

## 7.1 on single columns



```r
df_knn_1 = df_knn.copy()


imp_age_knn = KNNImputer(n_neighbors=2)

df_knn_1['Age'] = imp_age_knn.fit_transform(df_knn_1[['Age']])
df_knn_1
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p40.png)




```r
imp_salary_knn = KNNImputer(n_neighbors=2)

df_knn_1['Salary'] = imp_salary_knn.fit_transform(df_knn_1[['Salary']])
df_knn_1
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p41.png)


So far so good. However, this is not how the KNNImputer is used in practice. I will show you why in the following chapter.


## 7.2 on multiple columns



```r
df_knn_2 = df_knn.copy()
df_knn_2
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p42.png)




```r
imp_age_salary_knn = KNNImputer(n_neighbors=2)

df_knn_2[['Age', 'Salary']] = imp_age_salary_knn.fit_transform(df_knn_2[['Age', 'Salary']])
df_knn_2
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p43.png)


As you can see from the comparison below, the two methods use different values from the KNNImputer (see index 1,3 and 7).


```r
print()
print('df_knn_1')
print()
print(df_knn_1)
print('-------------------------')
print()
print('df_knn_2')
print()
print(df_knn_2)
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p44.png)


It should be mentioned here that sometimes it is better to choose this statistical approach in order to achieve better results in the later model training. 


## 7.3 Note on this Chapter

KNNImputer stores the calculated metrics for each column added to it. **The number and the order of the columns must remain the same!**


# 8 Imputation in Practice 

As already announced, I would like to show again how I use the replacement of missing values in practice during model training. It should be noted that I use a simple illustrative example below. In practice, there would most likely be additional steps like using encoders or feature scaling. This will be omitted at this point. But I will show in which case which order should be followed.

In the following, I use a modified version of the created dataset from the KNN example:



```r
df_practice = df_knn.copy()
Target_Var = [0,0,1,0,1,1,0,1,1,0]
df_practice['Target_Var'] = Target_Var

df_practice
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p45.png)

## 8.1 SimpleImputer in Practice

### 8.1.1 Train-Test Split


```r
df_practice_simpl_imp = df_practice.copy()
df_practice_simpl_imp
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p46.png)


In model training, I first divide the data set into a training part and a test part.


```r
x = df_practice_simpl_imp.drop('Target_Var', axis=1)
y = df_practice_simpl_imp['Target_Var']

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)
```




```r
print()
print('trainX')
print()
print(trainX)
print('-------------------------')
print()
print('testX')
print()
print(testX)
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p47.png)


### 8.1.2 Fit&Transform (trainX)

Then I check if there are any Missing Values in trainX.



```r
trainX.isnull().sum()
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p48.png)

As we can see, we need to replace missing values in the columns 'Age' and 'Salary'. For this I use the SimpleImputer with the strategy='mean'.




```r
# Fit and Transform trainX column 'Age' with strategy='mean'
imp_age_mean1 = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_age_mean1.fit(trainX[['Age']])
trainX['Age'] = imp_age_mean1.transform(trainX[['Age']])

# Fit and Transform trainX column 'Salary' with strategy='mean'
imp_salary_mean1 = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_salary_mean1.fit(trainX[['Salary']])
trainX['Salary'] = imp_salary_mean1.transform(trainX[['Salary']])

print(trainX)
print()
print('Number of missing values:')
print(trainX.isnull().sum())
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p49.png)


When I use the SimpleImputer, **I save it separately** to be able to use it again later. 


```r
pk.dump(imp_age_mean1, open('imp_age_mean1.pkl', 'wb'))
pk.dump(imp_salary_mean1, open('imp_salary_mean1.pkl', 'wb'))
```


### 8.1.3 Model Training

I **won't do** the model training at this point, because that would still require me to either remove the categorical variable 'name' or convert it to a numeric one. This would only cause confusion at this point.
Let's assume we have done the model training like this. 



```r
dt = DecisionTreeClassifier()
dt.fit(trainX, trainY)
```


The execution of the prediction function (apart from the still existing categorical variable) would not work like this.


```r
y_pred = df.predict(testX)
```

Because we also have Missing Values in the testX part.



```r
testX.isnull().sum()
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p50.png)


In order to be able to test the created model, we also need to replace them. For this purpose, we saved the metrics of the two SimpleImputers used in the previous step and can use them again here.


### 8.1.4 Transform (testX)

In the following, I will show the syntax to replace missing values for both columns ('Age' and 'Salary'). I am aware that in this example only the 'Age' column contains a missing value. But in practice the data sets are usually larger than 10 observations. 



```r
# Transform testX column 'Age'
testX['Age'] = imp_age_mean1.transform(testX[['Age']])

# Transform testX column 'Salary'
testX['Salary'] = imp_salary_mean1.transform(testX[['Salary']])

print(testX)
print()
print('Number of missing values:')
print(testX.isnull().sum())
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p51.png)

Now the test of our model training would also work. 


## 8.2 ffill & backfill in Practice

As already noted in chapter 6.3, we cannot directly save a metric for further use with the ffill or backfill method. Therefore, in this part I show how I proceed in such a situation.

Here I will not go into each step individually, as they have been sufficiently explained in the previous chapter.


### 8.2.1 Train-Test Split



```r
df_practice_ffill_bfill = df_practice.copy()
df_practice_ffill_bfill
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p52.png)


```r
x = df_practice_ffill_bfill.drop('Target_Var', axis=1)
y = df_practice_ffill_bfill['Target_Var']

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)
```




```r
print()
print('trainX')
print()
print(trainX)
print('-------------------------')
print()
print('testX')
print()
print(testX)
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p53.png)


### 8.2.2 Use ffill or backfill

Now I will replace the missing values in our example with the ffill method. 


```r
# ffill column 'Age'
trainX['Age'] = trainX['Age'].fillna(method='ffill')
# ffill column 'Salary'
trainX['Salary'] = trainX['Salary'].fillna(method='ffill')

print(trainX)
print()
print('Number of missing values:')
print(trainX.isnull().sum())
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p54.png)

### 8.2.3 Fit (trainX)


```r
# Fit trainX column 'Age' with strategy='mean'
imp_age_mean2 = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_age_mean2.fit(trainX[['Age']])

# Fit trainX column 'Salary' with strategy='mean'
imp_salary_mean2 = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_salary_mean2.fit(trainX[['Salary']])
```



```r
pk.dump(imp_age_mean, open('imp_age_mean2.pkl', 'wb'))
pk.dump(imp_salary_mean, open('imp_salary_mean2.pkl', 'wb'))
```


### 8.2.4 Transform (testX)

I'll leave out the part about model training at this point, since it would only be fictitiously presented anyway. 


```r
testX.isnull().sum()
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p55.png)




```r
# Transform testX column 'Age'
testX['Age'] = imp_age_mean2.transform(testX[['Age']])

# Transform testX column 'Salary'
testX['Salary'] = imp_salary_mean2.transform(testX[['Salary']])

print(testX)
print()
print('Number of missing values:')
print(testX.isnull().sum())
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p56.png)

## 8.3 KNNImputer in Practice

Now we come to the last method described in this post for replacing missing values in practice.


### 8.3.1 Train-Test Split


```r
df_practice_knn = df_practice.copy()
df_practice_knn
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p57.png)



```r
x = df_practice_knn.drop('Target_Var', axis=1)
y = df_practice_knn['Target_Var']

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)
```



```r
print()
print('trainX')
print()
print(trainX)
print('-------------------------')
print()
print('testX')
print()
print(testX)
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p58.png)

### 8.3.2 Fit&Transform (trainX)



```r
# Fit and Transform trainX column 'Age' and 'Salary'
imp_age_salary_knn1 = KNNImputer(n_neighbors=2)
trainX[['Age', 'Salary']] = imp_age_salary_knn1.fit_transform(trainX[['Age', 'Salary']])

print(trainX)
print()
print('Number of missing values:')
print(trainX.isnull().sum())
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p59.png)



```r
pk.dump(imp_age_salary_knn1, open('imp_age_salary_knn1.pkl', 'wb'))
```



### 8.3.3 Transform (testX)


I'll leave out again the part about model training at this point, since it would only be fictitiously presented anyway.



```r
testX.isnull().sum()
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p60.png)




```r
# Transform testX column 'Age' and 'Salary'
testX[['Age', 'Salary']] = imp_age_salary_knn1.transform(testX[['Age', 'Salary']])

print(testX)
print()
print('Number of missing values:')
print(testX.isnull().sum())
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p61.png)


## 8.4 Recommendation


What works for me in practice is a method from statistics that can be used when the data of a variable is normally distributed.


We know from [normal distributions (Bell Curve)](https://www.simplypsychology.org/normal-distribution.html) that 68% of the data lie between Z-1 and Z1. 

![](/post/2019-03-18-dealing-with-missing-values_files/p4s1.png)

Source: [SimplyPsychology](https://www.simplypsychology.org/normal-distribution.html)



That is, they have a mean value of 0 +- 1sd with a standard normal distribution. 


![](/post/2019-03-18-dealing-with-missing-values_files/p4s2.png)

Source: [SimplyPsychology](https://www.simplypsychology.org/normal-distribution.html)



Therefore, for a variable with a normal distribution, we can replace the missing values with random values that have a range from mean - 1sd to mean + 1sd. 

This method, is a little more cumbersome than the functions we used before, but it provides slightly more accurate values.




```r
df_recom = df_knn.copy()
df_recom = df_recom[['Name', 'Age']]
df_recom
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p62.png)




```r
mean_age = df_recom['Age'].mean()
sd_age = df_recom['Age'].std()

print('Mean of columne "Age": ' + str(mean_age))
print('Standard deviation of columne "Age": ' + str(sd_age))
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p63.png)



Fast saving of the values:


```r
pk.dump(mean_age, open('mean_age.pkl', 'wb'))
pk.dump(sd_age, open('sd_age.pkl', 'wb'))
```

With the random.uniform function I can output floats for a certain range.



```r
random.uniform(mean_age-sd_age, 
               mean_age+sd_age)

print('Lower limit of the range: ' + str(mean_age-sd_age))
print('Upper limit of the range: ' + str(mean_age+sd_age))
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p64.png)





```r
def fill_missings_gaussian_func(col, mean, sd):
    if np.isnan(col) == True: 
        col = random.uniform(mean-sd, mean+sd)
    else:
         col = col
    return col
```




```r
df_recom['Age'] = df_recom['Age'].apply(fill_missings_gaussian_func, args=(mean_age, sd_age)) 
df_recom
```

![](/post/2019-03-18-dealing-with-missing-values_files/p4p65.png)



Voilá, we have inserted different new values for the missing values of the column 'Age', which are between the defined upper and lower limit of the rage. Now, if you want to be very precise, you can round the 'Age' column to whole numbers to make it consistent.




# 9 Conclusion

In this post, I have shown different methods of replacing missing values in a dataset in a useful way. Furthermore, I have shown how these procedures should be applied in practice during a model training. 





