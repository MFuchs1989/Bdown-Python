---
title: NLP - Text Manipulation
author: Michael Fuchs
date: '2021-05-18'
slug: nlp-text-manipulation
categories: []
tags: []
output:
  blogdown::html_page:
    toc: true
    toc_depth: 5
---


# 1 Introduction

![](/post/2021-05-18-nlp-text-manipulation_files/p122s1.png)

Let's now move on to another large but very interesting topic area from the field of Data Science: **Natural Language Processing**

I already covered the topic of [String Manipulation](https://michael-fuchs-python.netlify.app/2019/03/27/string-manipulation-an-intuition/) once at the beginning of my blog series on Data Science with Python. That was more about handling text columns with functions like: 

+ [Separate](https://michael-fuchs-python.netlify.app/2019/03/27/string-manipulation-an-intuition/#separate)
+ [Unite](https://michael-fuchs-python.netlify.app/2019/03/27/string-manipulation-an-intuition/#unite) and
+ [Prefixes](https://michael-fuchs-python.netlify.app/2019/03/27/string-manipulation-an-intuition/#add_prefix)

In the following, we will delve deeper into the topic of text processing in order to be able to extract valuable insights from text variables using machine learning. 


## 1.1 What is NLP?

[Natural Language Processing (NLP)](https://becominghuman.ai/a-simple-introduction-to-natural-language-processing-ea66a1747b32), is a branch of artificial intelligence and generally defined as the automatic manipulation of natural language, such as speech and text, by software. Natural Language Processing interfaces with many disciplines, including computer science and computational linguistics, to bridge the gap between human communication and computer understanding.


NLP is not a new science in this sense and has been around for a very long time. In recent years, the need for and interest in human-machine communication has increased dramatically, so that with the availability of big data and increasingly powerful computers, the technology for NLP has also developed rapidly and is now accessible to a wide range of interested parties. 


## 1.2 Future Perspectives for NLP

The global Natural Language Processing (NLP) market size is expected to grow from [USD 11.6 billion in 2020 to USD 35.1 billion by 2026](https://www.marketsandmarkets.com/Market-Reports/natural-language-processing-nlp-825.html), according to statistics from [MarketsandMarkets](https://www.marketsandmarkets.com/). The increasing adoption of NLP-based applications in various industries is expected to provide tremendous opportunities for NLP providers. 



## 1.3 Application Areas of NLP


Let’s take a look at [11 of the most interesting applications](https://monkeylearn.com/blog/natural-language-processing-applications/) of natural language processing  in business:

+ Sentiment Analysis
+ Text Classification
+ Chatbots & Virtual Assistants
+ Text Extraction
+ Machine Translation
+ Text Summarization
+ Market Intelligence
+ Auto-Correct
+ Intent Classification
+ Urgency Detection
+ Speech Recognition

I will start here with the basics and then from post to post cover more and deeper topics as described above. 



# 2 Text Manipulation


Before we can take the first steps towards NLP, we should know the basics of text manipulation. 
They are basics but they are essential to take further steps towards model training. 



## 2.1 String Variables

First of all, we assign a sample string to an object. 


```r
word = "Hello World!"

print(word)
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p1.png)


## 2.2 Use of Quotation Marks

If you want to use quotation marks within a string, you should choose one of the following two options and stick to the chosen variant for the sake of consistency to avoid problems later on. 



```r
quotation_marks_var1 = 'Hi, my name is "Alice"'
print(quotation_marks_var1)
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p2.png)



```r
quotation_marks_var2 = "Hi, my name is 'Alice'"
print(quotation_marks_var2)
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p3.png)


## 2.3 Obtaining specific Information from a String


Access the first character of a string:


```r
word[0]
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p4.png)


Access specific characters of a string via slicing:


```r
word[6:12]
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p5.png)



```r
word[:5]
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p6.png)


Obtaining the length of a string:


```r
len(word)
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p7.png)

Counting the number of specific letters (here 'l') within a string:



```r
word.count('l')
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p8.png)

Find the index of a specific letter:


```r
word.find('W')
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p9.png)

That's right, the letter W of the word World is at index position 6 of our string. We do not only have to search for certain letters (there can be several identical letters in a string, in this case the index value of the first letter found would be output) but we can also output the index at which a certain word starts:



```r
word.index('World')
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p10.png)



## 2.4 String Manipulation

For the following examples, let's take a look at this kind of example string:



```r
word2 = "tHiS Is aN uNstRucTured sEnTencE"

print(word2)
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p11.png)


Convert all characters to uppercase:


```r
word2.upper()
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p12.png)


Convert all characters to lowercase:



```r
word2.lower()
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p13.png)


Capitalize the first letter of each word:



```r
word2.title()
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p14.png)


Capitalize only the first letter of a sentence:


```r
word2.capitalize()
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p15.png)


You also have the possibility to reverse the upper and lower case of a string. Let's take this example sentence for this:



```r
word3 = "another FUNNY sentence"

print(word3)
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p16.png)


```r
word3.swapcase()
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p17.png)


## 2.5 Arithmetic Operations

Mathematical operations are just as well possible with strings. See the following examples:

Addition of another string part:


```r
print(word)
print()
print(word + ' What a sunny day!')
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p18.png)


Have a string played back multiple times:


```r
print(word * 5)
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p19.png)

Or so a little prettier:


```r
print((word + ' ')* 5)
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p20.png)


With join we can insert a space between the individual letters


```r
print(' '.join(word))
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p21.png)

or reverse the order of the sting:


```r
print(''.join(reversed(word)))
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p22.png)


## 2.6 Check String Properties

In the following I will check some properties of the string.


Here again the string:


```r
print(word)
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p23.png)


Check if all characters of the sting are alphanumeric:


```r
word.isalnum()
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p24.png)

Check if all characters of the sting are alphabetic:


```r
word.isalpha()
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p25.png)

Check if string contains digits:


```r
word.isdigit()
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p26.png)

Check if string contains title words:



```r
word.istitle()
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p27.png)

Check if the complete string is in upper case:



```r
word.isupper()
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p28.png)


Check if the complete string is in lower case:


```r
word.islower()
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p29.png)

Check if the string consists of spaces:


```r
word.isspace()
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p30.png)


Check whether the string ends with a !:


```r
word.endswith('!')
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p31.png)


Check whether the string starts with an 'H':


```r
word.startswith('H')
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p32.png)


## 2.7 Replace certain Characters in Strings

Very often used in practice in the replacement of string parts:



```r
print(word)
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p33.png)


```r
word.replace('World', 'Germany')
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p34.png)



```r
word.replace(word[:5], 'Good Morning')
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p35.png)


## 2.8 For Loops with Strings

Finally, two examples of how to use for loops in combination with strings: 



```r
for char in word:
    print(char)
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p36.png)




```r
for char in word:
    print(str(word.index(char)) + ': ' + char)
```

![](/post/2021-05-18-nlp-text-manipulation_files/p122p37.png)



# 3 Conclusion

In this post I introduced the topic of NLP and showed the basics of text manipulation. In the following I will start with the topic of text pre-processing. 






