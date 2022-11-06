---
title: How to use Pandas set_option()
author: Michael Fuchs
date: '2019-05-02'
slug: how-to-use-pandas-set-option
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

In my previous post ["How to suppress scientific notation in Pandas"](https://michael-fuchs-python.netlify.app/2019/04/28/how-to-suppress-scientific-notation-in-pandas/) I have shown how to use the set_option-function of pandas to convert scientifically written numbers into more readable ones.
I have taken this as an opportunity to introduce further possibilities of the set_options-function here.
As already mentioned ["at chapter 5.3"](https://michael-fuchs-python.netlify.app/2019/04/28/how-to-suppress-scientific-notation-in-pandas/), set_option() changes behavior globaly in Jupyter Notebooks. 
Therefore we have to reset them again!



# 2 The use of pandas set_option()


```r
import pandas as pd
import numpy as np
```


For the following examples I create a simple dataset with 100 rows and 4 columns.


```r
df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
df
```

![](/post/2019-05-02-how-to-use-pandas-set-option_files/p67p1.png)


## 2.1 to determine max_rows


```r
pd.set_option('display.max_rows', 2)
```


```r
df
```

![](/post/2019-05-02-how-to-use-pandas-set-option_files/p67p2.png)

We can also display all lines of the complete data set. 


```r
pd.set_option('display.max_rows', df.shape[0]+1)
```


```r
df
```

![](/post/2019-05-02-how-to-use-pandas-set-option_files/p67p3.png)

As already mentioned at the beginning we reset the previous setting every time we use it.



```r
pd.reset_option('display.max_rows')
```


## 2.2 to determine max_columns

Like the display of the rows, you can also schedule the output columns. 


```r
pd.set_option('display.max_columns',2)
```


```r
df
```

![](/post/2019-05-02-how-to-use-pandas-set-option_files/p67p4.png)

Now let's set them to max:


```r
pd.set_option('display.max_columns', df.shape[1]+1)
```


```r
df
```

![](/post/2019-05-02-how-to-use-pandas-set-option_files/p67p5.png)





```r
pd.reset_option('display.max_columns')
```



## 2.3 to determine text length


To show this I'll create a further dataframe:



```r
df_text = pd.DataFrame({'Novel': [1, 2, '...'],
                        'Text': ['This is a very long text to show how well the set_option function works with "display.max_colwidth"', 
                                 'This is also a very long text to show how well the set_option function works with "display.max_colwidth". I am also a much longer string than that of Novel 1', 
                                 '...']})
df_text
```

![](/post/2019-05-02-how-to-use-pandas-set-option_files/p67p6.png)

Let's see how long the string of the first row of the column text is.


```r
len(df_text['Text'][0])
```

![](/post/2019-05-02-how-to-use-pandas-set-option_files/p67p7.png)

99 characters. Ok but we don't know if this is also the longest string. We find out this as follows:



```r
longest_text = df_text.Text.map(len).max()
longest_text
```

![](/post/2019-05-02-how-to-use-pandas-set-option_files/p67p8.png)


Let's take this as input for our next set_option-function.



```r
pd.set_option('display.max_colwidth', int(longest_text+1))
```


```r
df_text
```

![](/post/2019-05-02-how-to-use-pandas-set-option_files/p67p9.png)


```r
pd.reset_option('display.max_colwidth')
```



## 2.4 to determine float_format

I have already introduced this part in my Post ["How to suppress scientific notation in Pandas"](https://michael-fuchs-python.netlify.app/2019/04/28/how-to-suppress-scientific-notation-in-pandas/). If you want to learn more about this function of set_option please see chapter 5.3.



# 3 Conclusion

The set_option function of Pandas has many more functions besides those presented here. 
Check out the official ["Pandas Homepage"](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.set_option.html) for this.



