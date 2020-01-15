# !/usr/bin/env/python3
# Copyright 2019 YueLiu liuyue2@bu.edu
# Program Goal: preparing the dataset for the next few steps

# import libraries
import numpy as np
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer # PorterStemmer is also ok
import re
from sklearn.feature_extraction.text import TfidfVectorizer

#task1: narrow down the scale of the dataset
# labeling set
# set seed for the reproducibility
np.random.seed(1)
# read the dataset
label_df = pd.read_csv("label.csv")
# obeserve the dataset
print(label_df.head(0)) # 7 columns: 0, Tweets, id, keyword, location, text, target
print(len(label_df)) # 200 rows
print(label_df.shape) # (200, 7)

# trainging set
# set seed for the reproducibility
np.random.seed(1)
# read the dataset
train_df = pd.read_csv("train.csv")
# obeserve the dataset
print(train_df.head(0)) # 5 columns: id, keyword, location, text, target
print(len(train_df)) # 7613 rows
print(train_df.shape) # (7613, 5)

# texting set
# set seed for the reproducibility
np.random.seed(2)
# read the dataset .tcv
test_df = pd.read_csv("test.csv")
# obeserve the dataset
print(test_df.head(0)) # 4 columns: id, keyword, location, text
print(len(test_df)) # 3263 rows
print(test_df.shape) # (3263, 4)

#task2: define a function to stem and tokenization
# tokenization into words
# stemming into the roots words
def tokenize_and_stem(text):
    tokens = [word for text in nltk.sent_tokenize(text) for word in nltk.word_tokenize(text)]
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]
    stems = [SnowballStemmer("english").stem(word) for word in filtered_tokens]
    return stems

#task3: fit transform into TfidfVectorizer
# convert text into numbers
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer= tokenize_and_stem,
                                 ngram_range=(1,3))
# fit and transform the vector with the Tweets
tfidf_matrix_train = tfidf_vectorizer.fit_transform([x for x in train_df["text"]])
print(tfidf_matrix_train.shape)
tfidf_matrix_test = tfidf_vectorizer.fit_transform([x for x in test_df["text"]])
print(tfidf_matrix_test.shape)
