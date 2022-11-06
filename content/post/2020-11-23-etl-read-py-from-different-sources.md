---
title: ETL - Read .py from different sources
author: Michael Fuchs
date: '2020-11-23'
slug: etl-read-py-from-different-sources
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

Looking back, we have already covered an incredible number of data science topics.

We have dealt with a wide range of ["topics in the field of machine learning"](https://michael-fuchs-python.netlify.app/2020/08/20/roadmap-for-the-machine-learning-fields/).
And furthermore, how these algorithms can be applied in practice. 
A rule of thumb says that a data scientist spends 80% of his time on data preparation. The same amount of code is generated at this point. 
If you are working on a customer project that is only interested in the results, the notebook in which you are working, for example, is quickly overcrowded with syntax that only refers to the preparation of the data. 
For such a case it is a good idea to write an ETL-script.
ETL stands for extract, transform and load. 
In python you have the possibility (I prefer Microsoft Visual Studio) to create a python file (.py). 
In this post I want to introduce how to call such python files from different sources and get their different functions. 
In the following publications I will present different types of ETL variations.


# 2 The Setup

My actual setup looks a little different, but we will come back later. 
Here I created a project folder where I put one python script in folder_1, another one in folder_2 and a third one under notebooks.

![](/post/2020-11-23-etl-read-py-from-different-sources_files/p86p1.png)

+ Step 1: Navigate to the notebooks folder
+ Step 2: Start the jupyter notebook from this point

From here I can call the python scripts as follows:


```r
import sys

# Specifies the file path where the first .py file is located.
sys.path.insert(1, '../folder1')
import py_script_source_1 as source1

# Specifies the file path where the second .py file is located.
sys.path.insert(1, '../folder2')
import py_script_source_2 as source2
```


# 3 Run the python scripts

Run script 1:


```r
# Run function from py_script_source_1 file
source1.happyBirthdayDaniel()
```

![](/post/2020-11-23-etl-read-py-from-different-sources_files/p86p2.png)


Run script 2:


```r
# Run function from py_script_source_2 file
source2.greetingsDaniel()
```

![](/post/2020-11-23-etl-read-py-from-different-sources_files/p86p3.png)

'Normal' libraries can still be imported as usual.



```r
import pandas 
import numpy
```


Even .py files which are located in the same folder as the .jpynb script can be called without specifying another path.


```r
import py_script_source_3 as source3
```

Let's try this script, too.


```r
# Run function from py_script_source_3 file
source3.thanks_for_reading()
```

![](/post/2020-11-23-etl-read-py-from-different-sources_files/p86p4.png)



# 4 Content of the python scripts

As we could see, the methods we used were not breathtaking. They were only used for illustrative purposes at this point.

But here is an overview of their exact contents:

**Script 1:**

```r
def happyBirthdayDaniel(): #program does nothing as written
    print("Happy Birthday to you!")
    print("Happy Birthday to you!")
    print("Happy Birthday, dear Daniel.")
    print("Happy Birthday to you!")
```

![](/post/2020-11-23-etl-read-py-from-different-sources_files/p86p5.png)


**Script 2:**

```r
def greetingsDaniel(): #program does nothing as written
    print("Thanks for your visit.")
    print("Thanks for being there.")
    print("Nice to have seen you again.")
    print("See you soon!")
```

![](/post/2020-11-23-etl-read-py-from-different-sources_files/p86p6.png)


**Script 3:**

```r
def thanks_for_reading(): #program does nothing as written
    print("Thank you for reading this article.!")
```

![](/post/2020-11-23-etl-read-py-from-different-sources_files/p86p7.png)


# 5 Conclusion

In this article I have shown exemplary how to call python scripts from different locations. 
As already announced I will talk about different variaions of ETLs in the following posts.
Keep reading.

