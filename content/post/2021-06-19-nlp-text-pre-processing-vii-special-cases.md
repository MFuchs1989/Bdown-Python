---
title: NLP - Text Pre-Processing VII (Special Cases)
author: Michael Fuchs
date: '2021-06-19'
slug: nlp-text-pre-processing-vii-special-cases
categories: []
tags: []
output:
  blogdown::html_page:
    toc: true
    toc_depth: 5
---


# 1 Introduction


I have already described the most common and most used Text Cleaning Steps in this post here: [Text Pre-Processing I (Text Cleaning)](https://michael-fuchs-python.netlify.app/2021/05/22/nlp-text-pre-processing-i-text-cleaning/#text-cleaning)

However, it may happen that the analysis request requires other processing of the text, for example, if numbers are not to be removed but converted into text and analyzed as well. 

This post is about such special cases.

Feel free to download the files I used from my [GitHub Repository](https://github.com/MFuchs1989/Datasets-and-Miscellaneous/tree/main/datasets/NLP/Text%20Pre-Processing%20VII%20(Special%20Cases)).



# 2 Import the Libraries



```r
import pandas as pd
from num2words import num2words
import re
```




# 3 Definition of required Functions

All functions are summarized here. I will show them again where they are used during this post if they are new and have not been explained yet.



```r
def remove_punctuation_func(text):
    '''
    Removes all punctuation from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without punctuations
    '''
    return re.sub(r'[^a-zA-Z0-9]', ' ', text)
```


```r
from emo_dictonary import EMOTICONS

def emoticons_to_words_func(text):
    '''
    Convert Emoticons to Words
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string with converted Emoticons to Words
    ''' 
    for emot in EMOTICONS:
        emoticon_pattern = r'('+emot+')'
        # replace
        emoticon_words = EMOTICONS[emot]
        replace_text = emoticon_words.replace(",","")
        replace_text = replace_text.replace(":","")
        replace_text_list = replace_text.split()
        emoticon_name = '_'.join(replace_text_list)
        text = re.sub(emoticon_pattern, emoticon_name, text)
    return text
```


```r
def chat_words_to_norm_words_func(text):
    '''
    Replaces common chat expressions with their spelled out form
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string with replaced chat expressions
    ''' 
    return re.sub(r'\S+', lambda m: chat_expressions_dict.get(m.group().upper(), m.group()) , text)
```


```r
def sep_num_words_func(text):
    '''
    Separates numbers from words or other characters
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string with separated numbers from words or other characters
    ''' 
    return re.sub(r"([0-9]+(\.[0-9]+)?)",r" \1 ", text).strip() 
```


```r
def num_to_words(text):
    '''
    Convert Numbers to Words
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string with converted Numbers to Words
    ''' 
    after_spliting = text.split()

    for index in range(len(after_spliting)):
        if after_spliting[index].isdigit():
            after_spliting[index] = num2words(after_spliting[index])
    numbers_to_words = ' '.join(after_spliting)
    return numbers_to_words
```



# 4 Text Pre-Processing - Special Cases

As you may recall, in my first [Text Pre-Processing post](https://michael-fuchs-python.netlify.app/2021/05/22/nlp-text-pre-processing-i-text-cleaning/), I created a guideline for [Text Cleaning](https://michael-fuchs-python.netlify.app/2021/05/22/nlp-text-pre-processing-i-text-cleaning/#text-cleaning). The following three sections are about special cases that can be applied in text cleaning (for example, because the analysis requirements demand it). 

**However, it is of essential importance at which point the operations are performed!**

Of course I have listed this in each section and summarized it in the last [chapter 'Application to a DataFrame'](https://michael-fuchs-python.netlify.app/2021/06/19/nlp-text-pre-processing-vii-special-cases/#application-to-a-dataframe).


## 4.1  Converting Emoticons to Words


This operation should be performed in any case before [Removing Punctuation](https://michael-fuchs-python.netlify.app/2021/05/22/nlp-text-pre-processing-i-text-cleaning/#removing-punctuation). I will explain why in [chapter 5 Application to a DataFrame](https://michael-fuchs-python.netlify.app/2021/06/19/nlp-text-pre-processing-vii-special-cases/#application-to-a-dataframe).


I found the dictionary of emoticons and emojis (emo_dictonary.py) as well as the following function in this [GitHub](https://github.com/NeelShah18/emot) Repository.


```r
from emo_dictonary import EMOTICONS

def emoticons_to_words_func(text):
    '''
    Convert Emoticons to Words
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string with converted Emoticons to Words
    ''' 
    for emot in EMOTICONS:
        emoticon_pattern = r'('+emot+')'
        # replace
        emoticon_words = EMOTICONS[emot]
        replace_text = emoticon_words.replace(",","")
        replace_text = replace_text.replace(":","")
        replace_text_list = replace_text.split()
        emoticon_name = '_'.join(replace_text_list)
        text = re.sub(emoticon_pattern, emoticon_name, text)
    return text
```



```r
messy_text_conv_emo = \
"You really did a great job :)!\
"
messy_text_conv_emo
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p1.png)




```r
emoticons_to_words_func(messy_text_conv_emo)
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p2.png)


As we can see, the function does not interfere with punctuation marks, even if they directly follow emoticons.


```r
messy_text_conv_emo2 = \
"You really did a great job :( !\
"
messy_text_conv_emo2
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p3.png)




```r
emoticons_to_words_func(messy_text_conv_emo2)
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p4.png)




```r
messy_text_conv_emo3 = \
"Great! 8‑D\
"
messy_text_conv_emo3
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p5.png)




```r
emoticons_to_words_func(messy_text_conv_emo3)
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p6.png)


## 4.2 Converting Chat Conversion Words to normal Words


This operation should be performed before we [remove irrelevant characters (Numbers and Punctuation)](https://michael-fuchs-python.netlify.app/2021/05/22/nlp-text-pre-processing-i-text-cleaning/#removing-irrelevant-characters-numbers-and-punctuation) from a text.

First we need to load the following data set. 

This can be done in two ways. Either you download the file from my [GitHub Repository](https://github.com/MFuchs1989/Datasets-and-Miscellaneous/tree/main/datasets/NLP/Text%20Pre-Processing%20VII%20(Special%20Cases)) and load it in the usual way in the Jupyter Notebook.


```r
chat_expressions = pd.read_csv('chat_expressions.csv', sep=',')
chat_expressions
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p7.png)


Or you can download it directly from my [GitHub Repository](https://github.com/MFuchs1989/Datasets-and-Miscellaneous/tree/main/datasets/NLP/Text%20Pre-Processing%20VII%20(Special%20Cases)):



```r
# use raw view for this
url = "https://raw.githubusercontent.com/MFuchs1989/Datasets-and-Miscellaneous/main/datasets/NLP/Text%20Pre-Processing%20VII%20(Special%20Cases)/chat_expressions.csv" 

chat_expressions = pd.read_csv(url, error_bad_lines=False)
chat_expressions
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p7.png)



Now we create a dictionary from it:



```r
chat_expressions_dict = dict(zip(chat_expressions.Chat_Words, chat_expressions.Chat_Words_Extended))
```

Here is the function we are about to use:


```r
def chat_words_to_norm_words_func(text):
    '''
    Replaces common chat expressions with their spelled out form
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string with replaced chat expressions
    ''' 
    return re.sub(r'\S+', lambda m: chat_expressions_dict.get(m.group().upper(), m.group()) , text)
```



```r
messy_text_chat_words = \
"I'm afk for a moment\
"
messy_text_chat_words 
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p8.png)




```r
chat_words_to_norm_words_func(messy_text_chat_words)
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p9.png)


This has worked well before. But **be careful**, **punctuation interferes** with the function, see here:


```r
messy_text_chat_words2 = \
"OMG, that's great news.\
"
messy_text_chat_words2
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p10.png)



```r
chat_words_to_norm_words_func(messy_text_chat_words2)
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p11.png)


It is therefore **recommended to remove punctuation marks at this point**. We have already learned about this function in the post about [text cleaning](https://michael-fuchs-python.netlify.app/2021/05/22/nlp-text-pre-processing-i-text-cleaning/#removing-punctuation).




```r
# Remove Punctuation from text
messy_text_chat_words2_wo_punct = remove_punctuation_func(messy_text_chat_words2)

# Convert chat words to normal words
chat_words_to_norm_words_func(messy_text_chat_words2_wo_punct)
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p12.png)



## 4.3 Converting Numbers to Words

The following function can be used after the emoticons and chat words have been converted (since they may contain numbers and thus lose their meaning) but before [irrelevant characters (Numbers and Punctuation)](https://michael-fuchs-python.netlify.app/2021/05/22/nlp-text-pre-processing-i-text-cleaning/#removing-irrelevant-characters-numbers-and-punctuation) have been removed from the text.



```r
def num_to_words(text):
    '''
    Convert Numbers to Words
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string with converted Numbers to Words
    ''' 
    after_spliting = text.split()

    for index in range(len(after_spliting)):
        if after_spliting[index].isdigit():
            after_spliting[index] = num2words(after_spliting[index])
    numbers_to_words = ' '.join(after_spliting)
    return numbers_to_words
```


### 4.3.1 Small Numbers



```r
messy_text_numbers_to_words = \
"I paid 6 dollars for it.\
"
messy_text_numbers_to_words
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p13.png)



```r
num_to_words(messy_text_numbers_to_words)
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p14.png)



### 4.3.2 Larger Numbers


```r
messy_text_numbers_to_words2 = \
"I give 42 points which results in a 4 star rating. \
"
messy_text_numbers_to_words2
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p15.png)



```r
num_to_words(messy_text_numbers_to_words2)
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p16.png)



### 4.3.3 Numbers combined with Words and Punctuation

Often we have the case that numbers appear combined with other words or special characters in the text.



```r
messy_text_numbers_to_words3 = \
"Over 50% of today's smartphones have a 6inch screen.\
"
messy_text_numbers_to_words3
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p17.png)

Our function **will not work** on this:



```r
num_to_words(messy_text_numbers_to_words3)
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p18.png)

Ok the problem with the special character we can solve with the [remove_punctuation function](https://michael-fuchs-python.netlify.app/2021/05/22/nlp-text-pre-processing-i-text-cleaning/#removing-punctuation).


```r
messy_text_numbers_to_words3_wo_punct = remove_punctuation_func(messy_text_numbers_to_words3)
messy_text_numbers_to_words3_wo_punct
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p19.png)


But now we still have the problem that the 6inch are written together. 
To solve such problems I have written the following function:



```r
def sep_num_words_func(text):
    '''
    Separates numbers from words or other characters
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string with separated numbers from words or other characters
    ''' 
    return re.sub(r"([0-9]+(\.[0-9]+)?)",r" \1 ", text).strip()
```



```r
messy_text_numbers_to_words3_separated = sep_num_words_func(messy_text_numbers_to_words3_wo_punct)
messy_text_numbers_to_words3_separated
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p20.png)


Now I can use the num_to_words function:



```r
num_to_words(messy_text_numbers_to_words3_separated)
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p21.png)


### 4.3.4 Limitations

Of course the function I wrote is not omnipotent and also has its limitations e.g. this one with floats:



```r
messy_text_numbers_to_words4 = \
"I paid 4.50 for this.\
"
messy_text_numbers_to_words4
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p22.png)



```r
num_to_words(messy_text_numbers_to_words4)
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p23.png)



```r
messy_text_numbers_to_words4_wo_punct = remove_punctuation_func(messy_text_numbers_to_words4)
messy_text_numbers_to_words4_wo_punct
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p24.png)



```r
num_to_words(messy_text_numbers_to_words4_wo_punct)
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p24z1.png)



# 5 **Application** to a DataFrame

Note the order in which I will perform the operations. 
If I would bypass this, for example emoticons could no longer be recognized, because they no longer exist due to the removal of special characters. 
However, in order to extract the maximum information content from a text, one must consider exactly in which order the functions should be applied. 

+ Step 1: Converting emoticons into words
+ Intermediate Step: Removal of punctuation marks
+ Step 2: Converting chat words into real words
+ Step 3: Converting numbers into words

The function for emoticons is not affected by punctuation marks. 
The conversion of chat words (for example if they are placed just before a punctuation mark without a space) is. The function would not work then. In some chat words there are also numbers. Converting them beforehand with our function would therefore make some chat words unusable. 
Therefore always follow the order as I described it above.



## 5.1  Loading the Data Set

This dataset is an artificially created dataset by me, which fits well to the just described topics from this post. 



```r
pd.set_option('display.max_colwidth', 1000)

df = pd.read_csv('df_text_pre_processing_special_cases.csv')
df.T
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p25.png)

For text pre-processing, I'll stick to the guideline I created in this post: [NLP - Text Pre-Processing I (Text Cleaning)](https://michael-fuchs-python.netlify.app/2021/05/22/nlp-text-pre-processing-i-text-cleaning/#text-cleaning)



```r
df['Comments_lower'] = df['Comments'].str.lower()
df.T
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p26.png)


At this point, we skip the following points as they are not relevant to the problem at hand:

+ Removing HTML tags
+ Removing URLs
+ Removing Accented Characters


## 5.2  Step 1: Converting emoticons into words

However, before we get to removing special characters or numbers (or converting them to words) we should still convert emoticons to words. 




```r
df['Comments_emos_converted'] = df['Comments_lower'].apply(emoticons_to_words_func)
df.T
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p27.png)


As we can see on line 2 and 4 (here visually represented as a column) this worked. The emoticons were converted even if punctuation marks were directly before or after them. 



## 5.3 Intermediate Step: Removal of punctuation marks

So that all chat words can now be recognized correctly, we remove all punctuation marks at this point.




```r
df['Comments_wo_punct'] = df['Comments_emos_converted'].apply(remove_punctuation_func)
df.T
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p28.png)


## 5.4 Step 2: Converting chat words into real words



```r
df['Comments_chat_words_converted'] = df['Comments_wo_punct'].apply(chat_words_to_norm_words_func)
df.T
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p29.png)



## 5.5  Step 3: Converting numbers into words

### 5.5.1  Separation of numbers and words

So that now also all numbers can be converted into words, I separate all numbers from words, if these are written together.


```r
df['Comments_separated'] = df['Comments_chat_words_converted'].apply(sep_num_words_func)
df.T
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p30.png)


### 5.5.2  Numbers2Words



```r
df['Comments_num_to_words'] = df['Comments_separated'].apply(num_to_words)
df.T
```

![](/post/2021-06-19-nlp-text-pre-processing-vii-special-cases_files/p130p31.png)


Wonderful, the maximum of information has now been extracted. 

Now the [function to remove unnecessary spaces](https://michael-fuchs-python.netlify.app/2021/05/22/nlp-text-pre-processing-i-text-cleaning/#removing-extra-whitespaces) could be used.




# 6 Conclusion¶

In this post I showed how to translate emoticons and chat words into real language as well as how to convert numbers into words.

Concluding my post series on text pre-processing, in the [next post](https://michael-fuchs-python.netlify.app/2021/06/23/nlp-text-pre-processing-all-in-one/) I would like to give a summary of all the pre-processing steps that fit and can be applied to the dataset. 











