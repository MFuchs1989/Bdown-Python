---
title: Modified Print Statements
author: Michael Fuchs
date: '2021-04-20'
slug: modified-print-statements
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

We often use print statements to get feedback on certain process steps or to present findigs. 
In this post, I want to show how to use print statements cleverly and make them more descriptive. 



# 2 Loading the libraries and classes




```r
import pandas as pd
import os
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



# 3 Modified Print Statements

## 3.1 Print Statements with Variables

As we all know for sure, beyond the simple text output like 

`print('My Text')` 

you can also print the contents of variables. 


### 3.1.1 String Variables

This is still relatively easy and requires no modification:



```r
today = pd.to_datetime('today').strftime('%Y-%m-%d')

print('Current Date: ' + today)
```

![](/post/2021-04-20-modified-print-statements_files/p119p1.png)

### 3.1.2 Nummeric Variables


Numeric variables cannot be used quite so easily in the print statement. We have to format them as string first. 


```r
my_calculation = 10 + 5 

print('My Calculation: ' + str(my_calculation))
```

![](/post/2021-04-20-modified-print-statements_files/p119p2.png)


## 3.2 Print Statements with compound Paths

Here we request the current working directory we are on:


```r
root_directory = os.getcwd()
root_directory
```

![](/post/2021-04-20-modified-print-statements_files/p119p3.png)

Now we connect this to our destination folder:


```r
new_path = root_directory + '\\' + 'Target_Folder'
print(new_path)
```

![](/post/2021-04-20-modified-print-statements_files/p119p4.png)

Even simpler, this is how it works:


```r
new_path = os.path.join(root_directory, 'Target_Folder')
print(new_path)
```

![](/post/2021-04-20-modified-print-statements_files/p119p4.png)


## 3.3 Color Print Statements

To make print statements even more beautiful, we can have parts printed in color or bold.
For this we use the Color-class created above. 



```r
print('Current Date: ' + Color.RED + today)
```

![](/post/2021-04-20-modified-print-statements_files/p119p5.png)



```r
print(Color.BLUE + 'Current Date: ' + 
      Color.BOLD + Color.RED + today)
```

![](/post/2021-04-20-modified-print-statements_files/p119p6.png)



```r
print(Color.RED + 'My ' 
      + Color.END + 
      'Calculation: ' + str(my_calculation))
```

![](/post/2021-04-20-modified-print-statements_files/p119p7.png)



```r
print(Color.BLUE + 'My ' + Color.END + 
      Color.UNDERLINE + Color.GREEN + 'Calculation:' + Color.END +
      ' ' + Color.BOLD + Color.RED + str(my_calculation))
```

![](/post/2021-04-20-modified-print-statements_files/p119p8.png)


## 3.4 Print Statements with if else




```r
num_list = (1, 2, 3, 4, 5, 6, 7, 8, 9)
threshold = 5

for i in num_list:
    if i < threshold:
        print(Color.GREEN + Color.BOLD + 'Below Threshold ' + str([i]))
    else:
        print(Color.RED + Color.BOLD + 'Above Threshold ' + str([i]))
```

![](/post/2021-04-20-modified-print-statements_files/p119p9.png)



# 4 Conclusion

In this short post I showed how to modify Print Statements and have them output in color. 




