# -*- coding: utf-8 -*-

# Natural language processing
# We will be given reviews and we will find whether this review is positive or negative

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading the dataset
dataset=pd.read_csv("Restaurant_Reviews.tsv",delimiter='\t',quoting=3) 
# delimiter='\t' indicates to pandas that our data is separated by tab and it also indicates that we are using
# tsv file not csv file. and quoting =3 will remove the double 
# quotes present in our dataset.


# Cleaning the text

# To read the sentiments and then predict the result we need to remove irrelevant word or irrelevant punctuation marks.
import re # this library containt all the tools that is used to remove punctuation sign and other things also.
import nltk # it contains the list of irrevalent which are not used to predict the result like as the,a,an,this,that,...
nltk.download('stopwords') # it will download the package nltk
from nltk.stem.porter import PorterStemmer # used to stemming the word
from nltk.corpus import stopwords # Stopwords contains list of irrelevant words
corpus=[] # user build list.Generally corpus means collection of text
from nltk.corpus import movie_reviews
for i in range(1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) # first argument is pattern(we only wants letter, no any puntuation sign)
    # Second argument is used to replace punctation sign with space
    #Third argument is the string from which we have to extract the words.
    review=review.lower() # will convert all the letters into lower case letter
    review=review.split()
    
    ps=PorterStemmer()
    #Since we know love,loved,loving all three are same. So whenever any word will come among of them 
    # we will use the root word i.e love here.PorterStemmmer is used for this.
    # l=stopwords.words('englsih') # returns a list
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')] #As our words are in english.
    # Now review contain all the relevant words.
    review=' '.join(review) # join all the words(string) present in list separated by a space
    corpus.append(review)
    
# Creating bag of words model

# Here we will create sparse matrix where column contains the words present in all the reviews and rows contain
# the review.
from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer() # Class used to create sparse matrix
# Sparse matrix contains nothing but the features or independent variable. Now using these feature we will predict our result
# Now this becomes a classification problem
X=cv.fit_transform(corpus).toarray() # Transform corpus into sparse matrix
Y=dataset.iloc[:,1].values

# Training the model using naive bayes algorithm.Generally we use naive bayes algorithm in natural language processing

#splitting the dataset into train set and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=1)

# Implementing Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

# predicting the result
y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
