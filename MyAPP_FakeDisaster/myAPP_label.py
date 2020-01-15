# !/usr/bin/env/python3
# Copyright 2019 YueLiu liuyue2@bu.edu
# Program Goal: labeling a small-scale dataset by Snorkel
# labeling doc: https://snorkel.readthedocs.io/en/v0.9.3/packages/labeling.html

# import libraries

import pandas as pd
import nltk
import numpy as np
from nltk.stem.snowball import SnowballStemmer # PorterStemmer is also ok
import re
import snorkel
from snorkel.labeling import labeling_function
from sklearn.feature_extraction.text import TfidfVectorizer

# Program 1: Preprocessing
#task1: narrow down the scale of the dataset
# trainging set
# set seed for the reproducibility
np.random.seed(1)
# read the dataset
train_df = pd.read_csv("train.csv")
# obeserve the dataset
print(train_df.head(0)) # 5 columns: id, keyword, location, text, target
print(len(train_df)) # 7613 rows
print(train_df.shape) # (7613, 5)

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


# Program 2: labeling API
# task 1: writing your own labeling functions
#word lists about real disasters
# ORGANIZATIONS = r"\bjew(PDA|ITDRC|UMCOR|RECON|Rubicon|VOAD|IRUSA|community|aid)"
# ACTIONS = r"\bjew(affect|help|pray|impact|assemble|re-establish|monitor|rebuild|safe|relief|emergency)"
# LOCATIONS = r"\bjew(PuertoRico|warehouse|Kincade|Getty|shelter|Atlantic|Bahamas|Florida|Los|California)"
# DISASTERS = r"\bjew(Earthquake|aftershock|Hurricane|wildfires|HillsideFire|burn|tides|storm|flood|lightning|Tropical)"

ORGANIZATIONS = ["PDA","ITDRC","UMCOR","RECON","Rubicon","VOAD","IRUSA","community","aid"]
ACTIONS = ["affect","help","pray","impact","assemble","re-establish","monitor","rebuild","safe","relief","emergency"]
LOCATIONS = ["PuertoRico","warehouse","Kincade","Getty","shelter","Atlantic","Bahamas","Florida","Los","California"]
DISASTERS = ["Earthquake","aftershock","Hurricane","wildfires","HillsideFire","burn","tides","storm","flood","lightning","Tropical"]

TRUTH = 1
FAKE = 0

from snorkel.labeling import labeling_function

@labeling_function()
def LF_national_org(text):
    return TRUTH if "LF_national_org" in ORGANIZATIONS else FAKE

@labeling_function()
def LF_action_help(text):
    return TRUTH if "LF_action_help" in ACTIONS else FAKE

@labeling_function()
def LF_disaster_location(text):
    return TRUTH if "LF_disaster_location" in LOCATIONS else FAKE

@labeling_function()
def LF_disaster_type(text):
    return TRUTH if "LF_disaster_type" in DISASTERS else FAKE

# task 2: applying on labeling functions
#gather together the label functions and begin annotating
from snorkel.labeling import PandasLFApplier
LFs = [LF_national_org, LF_action_help, LF_disaster_location, LF_disaster_type]
applier = PandasLFApplier(lfs=LFs)  # LF applier for a Pandas DataFrame
label_train = applier.apply(df=train_df)
#applier.apply(df=test_df)
# #generating a label matrix
# np.random.seed(0)
# L_train = labeler.apply(split=0, parallelism=1)
# print(L_train.shape)
# L_dev = labeler.apply_existing(split=1, parallelism=1)
# print(L_dev.shape)
#get an accuracy score

# task 3: iterating on labeling functions
#adding new labels
#get an coverage
from snorkel.labeling import LFAnalysis
coverage_score = snorkel.labeling.LFAnalysis(L=label_train, lfs=LFs).label_coverage()
print(coverage_score)
