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
