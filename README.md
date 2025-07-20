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
* matplotlib
  * basic plots
* seaborn
  * advanced plots
* Scikit-learn
  * https://scikit-learn.org

### Numpy
see basic example in 002-Numpy.ipynb
### Pandas
see basic example in 003-Pandas.ipynb


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
#### using seaborn
sns.pairplot(mydataFrame)

with random variables you are sure that there is no correlations at all
```
import seaborn as sns
import numpy as np

# Example dataset
df = pd.DataFrame({
    'Math_Score': np.random.randint(50, 100, 50),
    'Science_Score': np.random.randint(50, 100, 50),
    'English_Score': np.random.randint(50, 100, 50)
})

# Creating a pair plot
sns.pairplot(df)
plt.show()
```

A more complete and realistic example
```
import seaborn
import matplotlib.pyplot as plt
df = seaborn.load_dataset('tips')
seaborn.pairplot(df, hue ='day')
plt.show()
```

## Example of artificial neural networks
### RNN (Recurent neural networks)
#### Definition from wikipedia
In artificial neural networks, recurrent neural networks (RNNs) are designed for processing sequential data, such as text, speech, and time series, where the order of elements is important.
RNNs utilize recurrent connections, where the output of a neuron at one time step is fed back as input to the network at the next time step. 
This enables RNNs to capture temporal dependencies and patterns within sequences.

The fundamental building block of RNN is the recurrent unit, which maintains a hidden state—a form of memory that is updated at each time step based on the current input and the previous hidden state.

https://en.wikipedia.org/wiki/Recurrent_neural_network

## Generative IA
### Definition
A type of Artificial Intelligence that creates contents (not only text) based on what it has learned from existing content
* the input is called the prompt
* the output is the completion

### Usage of Generative IA
About text
* Classify text
* answering question
* summarizing content
* generating text
  * Translate to another language
  * reformutate for a specific audience
  * generate code
#### NLP Natural Language Processing
Main domains of NLP or Natural Language Processing are
* NLU
  * Natural Language Understanding
* NLG
  * Natural Language Generation

##### Timeline of NLP
NLP is quite an old subject in IT. Several approaches have been used following the time.
* 1950 1980 Syntastic and grammar based
* 1980 2000 Expert systems and statistical models
* 2000 2010 Neural models and dense representation
* 2010 2020 Deep learning
* 2020 - now LLM 
    
#### LLM
##### Assitant
An assistant enable you to perform more complex task than a prompt feature by combining prompts with plugins and documents.

The goal of an assistant is to help people with repeated tasks.

## Resources
### Books
* [B1]Introduction to Machine Learning with Python (Andreas C Müller, Sarah Guido)
* [B2] Large Language Models:A deep dive (Uday Kamath, Kevin Keenan, Garrett Somers, Sarah Sorenson) Springer
