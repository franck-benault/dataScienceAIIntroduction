# Data Science, Machine Learning, AI Introduction
Simple introduction about Data Science, Machine Learning, AI

## Introduction

* AI
  * Machine Learning
    * Deep Learning
      * Generative IA (LLM) 

## Python
### Python Libraries
* Numpy
* Pandas
* Scikit-learn
  * https://scikit-learn.org
* matplotlib
  * basic plots
* seaborn
  * advanced plots
### Pandas
### Basic code
```
# Basic code to manage a data frame
myDataFrame.head()
myDataFrame.info()
myDataFrame.describe()
myDataFrame.hist(bins=4)
```

```
# see values for one features
myDataFrame['myFeature'].value_counts()
```

### Split train test data

```
import sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(myDataFrame, test_size=0.2, random_state=42)
```

```
# important to work on a copy of a dataframe if any issue you can go back to the original data
dfCopy = df.copy()
```

### Data visualization
#### using Pandas dataframe
The dataframe in Pandas has the basic functionalities to create the plots.
Actually it uses matplotlib.
##### Histogram
```
# Importing pandas library
import pandas as pd
import matplotlib.pyplot as plt

# Creating a Data frame
values = pd.DataFrame({
    'Foo1': [3.7, 8.7, 3.4, 2.4, 1.9,1.3],
    'Foo2': [4.24, 2.67, 7.6, 7.1, 4.9,5.4],
    'Foo3': [4.34, 3.67, 7.6, 7.1, 4.9,5.4],
    'Foo4': [4.34, 3.47, 3.6, 7.1, 4.9,5.4]
})

# Creating Histograms of all columns 
values.hist(bins=5,column=['Foo1','Foo2','Foo4'], grid=False)
plt.show()
```


## Generative IA
### Definition
A type of Artificial Intelligence that creates contents (not only text) based on what it has learned from existing content
* the input is called the prompt
* the output is the completion

### Usage of Generative IA
about text
* Classify text
* answering question
* summarizing content
* generating text
  * Translate to another language
  * reformutate for a specific audience
  * generate code 

## Resources
### Books
* Introduction to Machine Learning with Python (Andreas C Müller, Sarah Guido)
