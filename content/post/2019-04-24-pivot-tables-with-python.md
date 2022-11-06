---
title: Pivot Tables with Python
author: Michael Fuchs
date: '2019-04-24'
slug: pivot-tables-with-python
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

Many people like to work with pivot tables in Excel. This possibility also exists in Python.

For this post the dataset *WorldHappinessReport* from the statistic platform ["Kaggle"](https://www.kaggle.com) was used. You can download it from my [GitHub Repository](https://github.com/MFuchs1989/Datasets-and-Miscellaneous/tree/main/datasets).



**Loading the libraries and the data**


```r
import pandas as pd
import numpy as np
```


```r
happy = pd.read_csv("path/to/file/WorldHappinessReport.csv")
```


# 2 Getting an overview of our data



```r
happy.head(2)
```

![](/post/2019-04-24-pivot-tables-with-python_files/p8p1.png)


Getting an overview of our data and checking for missing values:

```r
print("Our data has {0} rows and {1} columns".format(happy.shape[0], happy.shape[1]))
print("Are there missing values? {}".format(happy.isnull().any().any()))
```

![](/post/2019-04-24-pivot-tables-with-python_files/p8p2.png)

 
# 3 Categorizing the data by Year and Region


```r
pd.pivot_table(happy, index= 'Year', values= "Happiness Score")
```

![](/post/2019-04-24-pivot-tables-with-python_files/p8p3.png)



```r
pd.pivot_table(happy, index = 'Region', values="Happiness Score").head()
```

![](/post/2019-04-24-pivot-tables-with-python_files/p8p4.png)

# 4 Creating a multi-index pivot table


```r
pd.pivot_table(happy, index = ['Region', 'Year'], values="Happiness Score").head(9)
```

![](/post/2019-04-24-pivot-tables-with-python_files/p8p5.png)



```r
pd.pivot_table(happy, index= 'Region', columns='Year', values="Happiness Score")
```

![](/post/2019-04-24-pivot-tables-with-python_files/p8p6.png)


# 5 Manipulating the data using aggfunc


```r
pd.pivot_table(happy, index= 'Region', values= "Happiness Score", aggfunc= [np.mean, np.median, np.min, np.max, np.std])
```

![](/post/2019-04-24-pivot-tables-with-python_files/p8p7.png)


# 6 Applying a custom function to remove outlier

Here we see how many countries exist in a region

```r
happy[['Region', 'Country']].groupby(['Region']).nunique().drop(columns=['Region']).reset_index()
```

![](/post/2019-04-24-pivot-tables-with-python_files/p8p8.png)


Let’s create a function that only calculates the values that are between the 0.25th and 0.75th quantiles.

```r
def remove_outliers(values):
    mid_quantiles = values.quantile([.25, .75])
    return np.mean(mid_quantiles)


pd.pivot_table(happy, index = 'Region', values="Happiness Score", aggfunc= [np.mean, remove_outliers])
```

![](/post/2019-04-24-pivot-tables-with-python_files/p8p9.png)


# 7 Categorizing using string manipulation

Here for Asia:

```r
table = pd.pivot_table(happy, index = 'Region', values="Happiness Score", aggfunc= [np.mean, remove_outliers])
table[table.index.str.contains('Asia')]
```

![](/post/2019-04-24-pivot-tables-with-python_files/p8p10.png)



Here for Europe:

```r
table[table.index.str.contains('Europe')]
```

![](/post/2019-04-24-pivot-tables-with-python_files/p8p11.png)



Now for certain years and regions:

```r
table = pd.pivot_table(happy, index = ['Region', 'Year'], values='Happiness Score',aggfunc= [np.mean, remove_outliers])

table.query('Year == [2015, 2017] and Region == ["Sub-Saharan Africa", "Middle East and Northern Africa"]')
```

![](/post/2019-04-24-pivot-tables-with-python_files/p8p12.png)




# 8 Conclusion

As you can see in Python you do not have to do without pivot tables if you like working with them. In my opinion, pivot tables are a great way to get a quick overview of the data and make comparisons between variables.






