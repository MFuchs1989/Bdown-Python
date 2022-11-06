---
title: Marketing - A/B Testing
author: Michael Fuchs
date: '2020-09-29'
slug: marketing-a-b-testing
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

At this point we have already covered some topics from the field of marketing:

+ ["Customer Lifetime Value"](https://michael-fuchs-python.netlify.app/2020/09/22/marketing-customer-lifetime-value/)
+ ["Market Basket Analysis"](https://michael-fuchs-python.netlify.app/2020/09/15/marketing-market-basket-analysis/)
+ ["Product Analytics and Recommendations"](https://michael-fuchs-python.netlify.app/2020/09/08/marketing-product-analytics-and-recommendations/)
+ ["Conversion Rate Analytics"](https://michael-fuchs-python.netlify.app/2020/09/01/marketing-conversion-rate-analytics/)

Now we turn to a smaller but equally important area: A/B Testing.
The decisions that are made in the marketing area can be very far-reaching. 
New product designs are developed, a new layout of the customer flyer is created and so on.
Such efforts are usually associated with high costs and should therefore achieve the desired effect. 
Whether this is the case can be determined in advance using A/B tests. 
This way, business decisions are not only made on the basis of gut feeling, but are supported by figures and facts. 

For this post the dataset *WA_Fn-UseC_-Marketing-Campaign-Eff-UseC_-FastF* from the statistic platform ["Kaggle"](https://www.kaggle.com) was used. You can download it from my ["GitHub Repository"](https://github.com/MFuchs1989/Datasets-and-Miscellaneous/tree/main/datasets).



# 2 Import the libraries and the data


```r
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

from scipy import stats
```



```r
df = pd.read_csv('WA_Fn-UseC_-Marketing-Campaign-Eff-UseC_-FastF.csv')
df
```

![](/post/2020-09-29-marketing-a-b-testing_files/p84p1.png)


# 3 Descriptive Analytics

As usual we start with descriptive statistics.
What interests us in this analysis is which campaign was most successful in terms of sales figures.

**Total Sales**


```r
ax = df.groupby(
    'Promotion'
).sum()[
    'SalesInThousands'
].plot.pie(
    figsize=(7, 7),
    autopct='%1.0f%%'
)


prom = ['Promotion 1', 'Promotion 2', 'Promotion 3']


ax.set_ylabel('')
ax.set_title('Sales distribution across different promotions')
plt.legend(prom, loc="upper right")

plt.show()
```

![](/post/2020-09-29-marketing-a-b-testing_files/p84p2.png)

**Market Size**


```r
df.groupby('MarketSize').count()['MarketID']
```

![](/post/2020-09-29-marketing-a-b-testing_files/p84p3.png)



```r
ax = df.groupby([
    'Promotion', 'MarketSize'
]).sum()[
    'SalesInThousands'
].unstack(
    'MarketSize'
).plot(
    kind='bar',
    figsize=(12,10),
    grid=True,
    stacked=True
)

ax.set_ylabel('Sales (in Thousands)')
ax.set_title('breakdowns of market sizes across different promotions')

plt.show()
```

![](/post/2020-09-29-marketing-a-b-testing_files/p84p4.png)

**Store Age**


```r
df['AgeOfStore'].describe()
```

![](/post/2020-09-29-marketing-a-b-testing_files/p84p5.png)



```r
ax = df.groupby(
    'AgeOfStore'
).count()[
    'MarketID'
].plot(
    kind='bar', 
    color='skyblue',
    figsize=(10,7),
    grid=True
)

ax.set_xlabel('age')
ax.set_ylabel('count')
ax.set_title('overall distributions of age of store')

plt.show()
```

![](/post/2020-09-29-marketing-a-b-testing_files/p84p6.png)



```r
ax = df.groupby(
    ['AgeOfStore', 'Promotion']
).count()[
    'MarketID'
].unstack(
    'Promotion'
).iloc[::-1].plot(
    kind='barh', 
    figsize=(12,15),
    grid=True
)

ax.set_ylabel('age')
ax.set_xlabel('count')
ax.set_title('overall distributions of age of store')

plt.show()
```

![](/post/2020-09-29-marketing-a-b-testing_files/p84p7.png)



```r
df.groupby('Promotion').describe()['AgeOfStore']
```

![](/post/2020-09-29-marketing-a-b-testing_files/p84p8.png)


# 4 Significance Tests

Now it is time for significance tests.
To investigate which campaign was most successful we use the t-test. 
What exactly does a t-test do?
Simply put, it looks to see if the mean values of two groups differ significantly. 
Let's take a look at the average sales figures broken down by campaign.



```r
means = df.groupby('Promotion').mean()['SalesInThousands']
means
```

![](/post/2020-09-29-marketing-a-b-testing_files/p84p9.png)

The mean values are different. But are these differences also significant? This question can be answered with a t-test.  For this we use the t-test function from the library scipy.

There are two important statistics in a t-Test, the *t-value* and the *p-value*.

The t-value measures the degree of difference relative to the variation in the data. The larger the t-value is, the more difference there is between two groups. 

On the other hand the p-value measures the probability that the results would occur by chance. The smaller the p-value is, the more statistically significant difference there will be between the two groups.


**Promotion 1 vs. 2**


```r
t, p = stats.ttest_ind(
    df.loc[df['Promotion'] == 1, 'SalesInThousands'].values, 
    df.loc[df['Promotion'] == 2, 'SalesInThousands'].values, 
    equal_var=False
)
```


```r
print("t-Value: " + str('{:.7f}'.format(t)))
print("p-Value: " + str('{:.7f}'.format(p)))
```

![](/post/2020-09-29-marketing-a-b-testing_files/p84p10.png)


**Promotion 1 vs. 3**


```r
t, p = stats.ttest_ind(
    df.loc[df['Promotion'] == 1, 'SalesInThousands'].values, 
    df.loc[df['Promotion'] == 3, 'SalesInThousands'].values, 
    equal_var=False
)
```


```r
print("t-Value: " + str('{:.7f}'.format(t)))
print("p-Value: " + str('{:.7f}'.format(p)))
```

![](/post/2020-09-29-marketing-a-b-testing_files/p84p11.png)


**Promotion 2 vs. 3**


```r
t, p = stats.ttest_ind(
    df.loc[df['Promotion'] == 2, 'SalesInThousands'].values, 
    df.loc[df['Promotion'] == 3, 'SalesInThousands'].values, 
    equal_var=False
)
```


```r
print("t-Value: " + str('{:.7f}'.format(t)))
print("p-Value: " + str('{:.7f}'.format(p)))
```

![](/post/2020-09-29-marketing-a-b-testing_files/p84p12.png)

**Interpretation**

As we can see from the p-value, the average value of the sales figures in promotion 1 and 3 do not differ significantly.
But the difference between promotion 1 and 2 does as well as promotion 2 vs. 3.
If you look at the corresponding t-value, you can say that promotion 1 and 3 were better than promotion 2.


# 5 Conclusion

In this post I discussed the performance and interpretation of A/B tests. 
Since a lot depends on the decisions of the marketing department, it is worthwhile, especially for far-reaching decisions, to carry out extensive A/B tests to get the desired results. 


**References**

The content of the entire post was created using the following sources:

Hwang, Y. (2019). Hands-On Data Science for Marketing: Improve your marketing strategies with machine learning using Python and R. Packt Publishing Ltd.






