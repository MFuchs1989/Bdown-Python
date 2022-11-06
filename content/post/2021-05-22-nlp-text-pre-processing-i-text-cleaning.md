---
title: NLP - Text Pre-Processing I (Text Cleaning)
author: Michael Fuchs
date: '2021-05-22'
slug: nlp-text-pre-processing-i-text-cleaning
categories: []
tags: []
output:
  blogdown::html_page:
    toc: true
    toc_depth: 5
---


# 1 Introduction


In my last post (NLP - Text Manipulation) I got into the topic of Natural Language Processing. 

However, before we can start with Machine Learning algorithms some preprocessing steps are needed.
I will introduce these in this and the following posts. Since this is a coherent post series and will build on each other I recommend to start with reading this post. 

For this publication the dataset *Amazon Unlocked Mobile* from the statistic platform ["Kaggle"](https://www.kaggle.com) was used. You can download it from my ["GitHub Repository"](https://github.com/MFuchs1989/Datasets-and-Miscellaneous/tree/main/datasets/NLP/Text%20Pre-Processing%20I%20(Text%20Cleaning)).



# 2 Import the Libraries and the Data


If you are using the nltk library for the first time, you should import and download the following:



```r
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
```



```r
import pandas as pd
import numpy as np

import pickle as pk

import warnings
warnings.filterwarnings("ignore")


from bs4 import BeautifulSoup
import unicodedata
import re

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

from nltk.corpus import stopwords


from nltk.corpus import wordnet
from nltk import pos_tag
from nltk import ne_chunk

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud
```



```r
df = pd.read_csv('Amazon_Unlocked_Mobile_small.csv')
df.head()
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p1.png)


However, we will only work with the following part of the data set:


```r
df = df[['Rating', 'Reviews']]
df.head()
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p2.png)

Let's take a closer look at the first set of reviews:


```r
df['Reviews'].iloc[0]
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p3.png)


```r
df.dtypes
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p4.png)

To be on the safe side, I convert the reviews as strings to be able to work with them correctly. 


```r
df['Reviews'] = df['Reviews'].astype(str)
```



# 3 Definition of required Functions


All functions are summarized here. I will show them again in the course of this post at the place where they are used. 



```r
def remove_html_tags_func(text):
    '''
    Removes HTML-Tags from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without HTML-Tags
    ''' 
    return BeautifulSoup(text, 'html.parser').get_text()
```


```r
def remove_url_func(text):
    '''
    Removes URL addresses from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without URL addresses
    ''' 
    return re.sub(r'https?://\S+|www\.\S+', '', text)
```


```r
def remove_accented_chars_func(text):
    '''
    Removes all accented characters from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without accented characters
    '''
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
```


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
def remove_irr_char_func(text):
    '''
    Removes all irrelevant characters (numbers and punctuation) from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without irrelevant characters
    '''
    return re.sub(r'[^a-zA-Z]', ' ', text)
```


```r
def remove_extra_whitespaces_func(text):
    '''
    Removes extra whitespaces from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without extra whitespaces
    ''' 
    return re.sub(r'^\s*|\s\s*', ' ', text).strip()
```


```r
def word_count_func(text):
    '''
    Counts words within a string
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Number of words within a string, integer
    ''' 
    return len(text.split())
```


# 4  Text Pre-Processing

There are some text pre-processing steps to consider and a few more you can do. In this post I will talk about text cleaning.

## 4.1  Text Cleaning

Here I have created an example string, where you can understand the following steps very well.


```r
messy_text = \
"Hi e-v-e-r-y-o-n-e !!!@@@!!! I gave a 5-star rating. \
Bought this special product here: https://www.amazon.com/. Another link: www.amazon.com/ \
Here the HTML-Tag as well:  <a href='https://www.amazon.com/'> …</a>. \
I HIGHLY RECOMMEND THIS PRDUCT !! \
I @ (love) [it] <for> {all} ~it's* |/ #special / ^^characters;! \
I am currently investigating the special device and am excited about the features. Love it! \
Furthermore, I found the support really great. Paid about 180$ for it (5.7inch version, 4.8'' screen). \
Sómě special Áccěntěd těxt and words like résumé, café or exposé.\
"
messy_text
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p5.png)


In the following I will perform the individual steps for text cleaning and always use parts of the messy_text string. 


### 4.1.1 Conversion to Lower Case

In general, it is advisable to format the text completely in lower case.




```r
messy_text_lower_case = \
"I HIGHLY RECOMMEND THIS PRDUCT !!\
"
messy_text_lower_case
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p6.png)



```r
messy_text_lower_case.lower()
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p7.png)


### 4.1.2 Removing HTML-Tags



```r
messy_text_html = \
"Here the HTML-Tag as well:  <a href='https://www.amazon.com/'> …</a>.\
"
messy_text_html
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p8.png)



```r
def remove_html_tags_func(text):
    '''
    Removes HTML-Tags from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without HTML-Tags
    ''' 
    return BeautifulSoup(text, 'html.parser').get_text()
```



```r
remove_html_tags_func(messy_text_html)
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p9.png)


### 4.1.3 Removing URLs



```r
messy_text_url = \
"Bought this product here: https://www.amazon.com/. Another link: www.amazon.com/\
"
messy_text_url
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p10.png)



```r
def remove_url_func(text):
    '''
    Removes URL addresses from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without URL addresses
    ''' 
    return re.sub(r'https?://\S+|www\.\S+', '', text)
```


```r
remove_url_func(messy_text_url)
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p11.png)


### 4.1.4 Removing Accented Characters



```r
messy_text_accented_chars = \
"Sómě Áccěntěd těxt and words like résumé, café or exposé.\
"
messy_text_accented_chars
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p12.png)



```r
def remove_accented_chars_func(text):
    '''
    Removes all accented characters from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without accented characters
    '''
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
```


```r
remove_accented_chars_func(messy_text_accented_chars)
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p13.png)


### 4.1.5 Removing Punctuation

Punctuation is essentially the following set of symbols: [!”#$%&’()*+,-./:;<=>?@[]^_`{|}~]



```r
messy_text_remove_punctuation = \
"Furthermore, I found the support really great. Paid about 180$ for it (5.7inch version, 4.8'' screen).\
"
messy_text_remove_punctuation
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p14.png)



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
remove_punctuation_func(messy_text_remove_punctuation)
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p15.png)


### 4.1.6 Removing irrelevant Characters (Numbers and Punctuation)


```r
messy_text_irr_char = \
"Furthermore, I found the support really great. Paid about 180$ for it (5.7inch version, 4.8'' screen).\
"
messy_text_irr_char
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p16.png)


I am aware that this is the same example sentence as in the previous example, but here the difference between this and the previous function is made clear.


```r
def remove_irr_char_func(text):
    '''
    Removes all irrelevant characters (numbers and punctuation) from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without irrelevant characters
    '''
    return re.sub(r'[^a-zA-Z]', ' ', text)
```


```r
remove_irr_char_func(messy_text_irr_char)
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p17.png)


### 4.1.7 Removing extra Whitespaces



```r
messy_text_extra_whitespaces = \
"I  am   a  text    with  many   whitespaces.\
"
messy_text_extra_whitespaces
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p18.png)



```r
def remove_extra_whitespaces_func(text):
    '''
    Removes extra whitespaces from a string, if present
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Clean string without extra whitespaces
    ''' 
    return re.sub(r'^\s*|\s\s*', ' ', text).strip()
```


```r
remove_extra_whitespaces_func(messy_text_extra_whitespaces)
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p19.png)

I always like to use this function in between, for example, when you have removed stop words, certain words or individual characters from the string(s). From time to time, this creates new whitespaces that I always like to remove for the sake of order.


### 4.1.8 Extra: Count Words

It is worthwhile to display the number of existing words, especially for validation of the pre-proessing steps. We will use this function again and again in later steps.


```r
messy_text_word_count = \
"How many words do you think I will contain?\
"
messy_text_word_count
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p20.png)



```r
def word_count_func(text):
    '''
    Counts words within a string
    
    Args:
        text (str): String to which the function is to be applied, string
    
    Returns:
        Number of words within a string, integer
    ''' 
    return len(text.split())
```


```r
word_count_func(messy_text_word_count)
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p21.png)



### 4.1.9 Extra: Expanding Contractions

You can do expanding contractions but you don't have to. For the sake of completeness, I list the necessary functions, but do not use them in our following example with the Example String and DataFrame. I will give the reason for this in a later chapter.


```r
from contractions import CONTRACTION_MAP 
import re 

def expand_contractions(text, map=CONTRACTION_MAP):
    pattern = re.compile('({})'.format('|'.join(map.keys())), flags=re.IGNORECASE|re.DOTALL)
    def get_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded = map.get(match) if map.get(match) else map.get(match.lower())
        expanded = first_char+expanded[1:]
        return expanded     
    new_text = pattern.sub(get_match, text)
    new_text = re.sub("'", "", new_text)
    return new_text
```

With the help of this function, this sentence:

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p22.png)


becomes the following:

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p23.png)

This function should also work for this:


```r
from pycontractions import Contractions
cont = Contractions(kv_model=model)
cont.load_models()# 

def expand_contractions(text):
    text = list(cont.expand_texts([text], precise=True))[0]
    return text
```


### 4.1.10 **Application** to the Example String

Before that, I used individual text modules to show how all the text cleaning steps work. Now it is time to apply these functions to Example String (and subsequently to the DataFrame) one after the other.



```r
messy_text
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p24.png)



```r
messy_text_lower = messy_text.lower()
messy_text_lower
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p25.png)



```r
messy_text_wo_html = remove_html_tags_func(messy_text_lower)
messy_text_wo_html
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p26.png)



```r
messy_text_wo_url = remove_url_func(messy_text_wo_html)
messy_text_wo_url
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p27.png)



```r
messy_text_wo_acc_chars = remove_accented_chars_func(messy_text_wo_url)
messy_text_wo_acc_chars
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p28.png)



```r
messy_text_wo_punct = remove_punctuation_func(messy_text_wo_acc_chars)
messy_text_wo_punct
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p29.png)



```r
messy_text_wo_irr_char = remove_irr_char_func(messy_text_wo_punct)
messy_text_wo_irr_char
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p30.png)


```r
clean_text = remove_extra_whitespaces_func(messy_text_wo_irr_char)
clean_text
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p31.png)



```r
print('Number of words: ' + str(word_count_func(clean_text)))
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p32.png)


### 4.1.11 **Application** to the DataFrame

Now we apply the Text Cleaning Steps shown above to the DataFrame:


```r
df.head()
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p33.png)



```r
df['Clean_Reviews'] = df['Reviews'].str.lower()
df['Clean_Reviews'] = df['Clean_Reviews'].apply(remove_html_tags_func)
df['Clean_Reviews'] = df['Clean_Reviews'].apply(remove_url_func)
df['Clean_Reviews'] = df['Clean_Reviews'].apply(remove_accented_chars_func)
df['Clean_Reviews'] = df['Clean_Reviews'].apply(remove_punctuation_func)
df['Clean_Reviews'] = df['Clean_Reviews'].apply(remove_irr_char_func)
df['Clean_Reviews'] = df['Clean_Reviews'].apply(remove_extra_whitespaces_func)

df.head()
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p34.png)

Let's now compare the sentences from line 1 with the ones we have now edited:


```r
df['Reviews'].iloc[0]
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p35.png)



```r
df['Clean_Reviews'].iloc[0]
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p36.png)

Finally, we output the number of words and store them in a separate column. In this way, we can see whether and to what extent the number of words has changed in further steps.


```r
df['Word_Count'] = df['Clean_Reviews'].apply(word_count_func)

df[['Clean_Reviews', 'Word_Count']].head()
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p37.png)

Here is the average number of words:


```r
print('Average of words counted: ' + str(df['Word_Count'].mean()))
```

![](/post/2021-05-22-nlp-text-pre-processing-i-text-cleaning_files/p123p38.png)


# 5 Conclusion

This was the first post in my series about text pre-processing. 
In it I have listed all the necessary steps that should always be followed (except in exceptional cases). 

To be able to proceed with the edited record in the next post, I save it and the Example String.


```r
pk.dump(clean_text, open('clean_text.pkl', 'wb'))

df.to_csv('Amazon_Unlocked_Mobile_small_Part_I.csv', index = False)
```


In the following post I will link where you can find these two files. 
Stay tuned to learn more about Text Pre-Processing.





