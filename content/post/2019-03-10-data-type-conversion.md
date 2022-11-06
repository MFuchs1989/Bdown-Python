---
title: Data type conversion
author: Michael Fuchs
date: '2019-03-10'
slug: data-type-conversion
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


It will always happen that you have an incorrect or unsuitable data type and you have to change it. It is therefore worthwhile to familiarize yourself with the conversion methods that exist in python.


For this post the dataset *Auto-mpg* from the statistic platform ["Kaggle"](https://www.kaggle.com) was used. You can download it from my [GitHub Repository](https://github.com/MFuchs1989/Datasets-and-Miscellaneous/tree/main/datasets).



# 2 Loading the libraries and the data



```r
import pandas as pd
import numpy as np

#for chapter 6
from datetime import datetime
```



```r
cars = pd.read_csv("path/to/file/auto-mpg.csv")
```


# 3 Overview of the existing data types


**Numeric:**

+ *integer*: Positive or negative whole numbers (without a fractional part)
+ *float*: Any real number with a floating point representation in which a fractional component is denoted by a decimal symbol or scientific notation
+ *complex number*: A number with a real and imaginary component represented as x+zj. x and z are floats and j is -1(square root of -1 called an imaginary number)


**Boolean**

Data with one of two built-in values 'True' or 'False.' 


**Sequence Type**

+ *string*: A string value is a collection of one or more characters put in single, double or triple quotes.
+ *list*: A list object is an ordered collection of one or more data items, not necessarily of the same type, put in square brackets.
+ *tuple*: A Tuple object is an ordered collection of one or more data items, not necessarily of the same type, put in parentheses.


**Dictionary**

A dictionary object is an unordered collection of data in a key:value pair form. A collection of such pairs is enclosed in curly brackets. For example: {1:"Sven", 2:"Tom", 3:"Eva", 4: "Will"}




You can check data types in python like this:



```r
type(1234)
```

![](/post/2019-03-10-data-type-conversion_files/p25p1.png)





```r
type(55.50)
```

![](/post/2019-03-10-data-type-conversion_files/p25p2.png)



```r
type(6+4j)
```

![](/post/2019-03-10-data-type-conversion_files/p25p3.png)



```r
type("hello")
```

![](/post/2019-03-10-data-type-conversion_files/p25p4.png)



```r
type([1,2,3,4])
```

![](/post/2019-03-10-data-type-conversion_files/p25p5.png)



```r
type((1,2,3,4))
```

![](/post/2019-03-10-data-type-conversion_files/p25p6.png)



```r
type({1:"one", 2:"two", 3:"three"})
```

![](/post/2019-03-10-data-type-conversion_files/p25p7.png)


# 4 Type Conversion

First of all let's have a look at the data types of our dataframe *cars*:



```r
cars.dtypes
```

![](/post/2019-03-10-data-type-conversion_files/p25p8.png)

## 4.1 Conversion of a single variable

### 4.1.1 float64 to float32


Conversions can be done within the same typ (here from float 64 to 32):


```r
cars['mpg'].dtypes
```

![](/post/2019-03-10-data-type-conversion_files/p25p9.png)



```r
cars['mpg'] = cars['mpg'].astype('float32')
cars['mpg'].dtypes
```

![](/post/2019-03-10-data-type-conversion_files/p25p10.png)


### 4.1.2 float to int

Conversions can also be made into any other data types:


```r
cars['mpg'].head()
```

![](/post/2019-03-10-data-type-conversion_files/p25p11.png)


```r
cars['mpg'] = cars['mpg'].astype('int64')

cars['mpg'].head()
```

![](/post/2019-03-10-data-type-conversion_files/p25p12.png)



### 4.1.3 object to numeric (float and int)


As you can see in the overview of the data types of the dataframe, the variable horsepower was loaded as an object. This should actually be an int. From this we now convert them into a numerical variable. astype () does not always work if, for example, there are stings under the objects. Here you can use the pandas function .to_numeric.




```r
cars["horsepower"] = pd.to_numeric(cars.horsepower, errors='coerce')

cars['horsepower'].dtypes
```

![](/post/2019-03-10-data-type-conversion_files/p25p13.png)




```r
cars['horsepower'].head()
```

![](/post/2019-03-10-data-type-conversion_files/p25p14.png)


As previously mentioned, horsepower is actually an int.
If we tried to convert it with the conventional syntax ("cars ['horsepower'] = cars ['horsepower']. astype ('int64')") we would get the following error message: "ValueError: Cannot convert non-finite values (NA or inf) to integer". This is because the variable horsepower contains NA or inf ...
Since Python version 0.24 pandas has gained the ability to hold integer dtypes with missing values. Just write the first letter of int as capital letter:



```r
cars['horsepower'] = cars['horsepower'].astype('Int64')

cars['horsepower'].dtypes
```

![](/post/2019-03-10-data-type-conversion_files/p25p15.png)




```r
cars['horsepower'].head()
```

![](/post/2019-03-10-data-type-conversion_files/p25p16.png)

# 5 Conversion of multiple variables


```r
cars[['cylinders', 'weight']].dtypes
```

![](/post/2019-03-10-data-type-conversion_files/p25p17.png)


```r
cars[['cylinders', 'weight']] = cars[['cylinders', 'weight']].astype('int32')
cars[['cylinders', 'weight']].dtypes
```

![](/post/2019-03-10-data-type-conversion_files/p25p18.png)


# 6 Conversion of date and time variables




```r
df = pd.DataFrame({'year': [2015, 2016],
                   'month': [2, 3],
                   'day': [4, 5],
                   'hour': [9, 11],
                   'minutes': [22, 50],
                   'seconds': [12, 8]})

df
```

![](/post/2019-03-10-data-type-conversion_files/p25p19.png)



```r
pd.to_datetime(df)
```

![](/post/2019-03-10-data-type-conversion_files/p25p20.png)


At least the year, the month and the day must be given here. One of the other three variables can be omitted at will. But month and day are not always numerical. The datetime library is ideal for these cases. Here are three examples of how differently formatted dates can be brought into a uniform format:


```r
date_string1 = 'Wednesday, June 6, 2018'
date_string2 = '6/6/18'
date_string3 = '06-06-2018'


date_date1 = datetime.strptime(date_str1, '%A, %B %d, %Y')
date_date2 = datetime.strptime(date_str2, '%m/%d/%y')
date_date3 = datetime.strptime(date_str3, '%m-%d-%Y')


print(date_date1)
print(date_date2)
print(date_date3)
```

![](/post/2019-03-10-data-type-conversion_files/p25p21.png)

Here is a short list of the most common directives: 

![](/post/2019-03-10-data-type-conversion_files/p25s1.png)

You can find the full list of directives ["here"](https://strftime.org/).


# 7 Conclusion

In this post we saw the different types of data and how to convert them to any other.





