# !/usr/bin/env/python3
# Copyright 2019 YueLiu liuyue2@bu.edu
# Application Goal: to find real disaster tweets by nltk and Snorkel API
# edition1: improve the quality of word lists by knowledge base(nltk.corpus.wordnet)
# edition2: label the test set at the same time
# edition3: add more functionality into the application
# pay attention: task 2 and task 3 are not for snorkel
# Programs include:
# preprocess(from line, very simple);
# labeling(from line, the core);
# clustering(comparison);
# storing(collect the labeled data, for explaining the rules);
# visualizing;

# import libraries
import json
import numpy as np
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import pymongo
import re
import snorkel
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer # PorterStemmer is also ok
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from sklearn.metrics.pairwise import cosine_similarity

# Program 1: preprocess, by nltk & sklearn
# DO NOT DO USE THE RESUILTS IN TAKS 2 & 3, IF YOU ONLY WANT THE SNORKEL PART
# please use the other one
#task1: get familiar with the training set and the testing set
# labeling set
# set seed for the reproducibility
np.random.seed(1)
# read the dataset
label_df = pd.read_csv("label.csv")
# observe the dataset
print(label_df.head(0)) # 7 columns: 0, Tweets, id, keyword, location, text, target
print(len(label_df)) # 200 rows
print(label_df.shape) # (200, 7)

# training set
# set seed for the reproducibility
np.random.seed(1)
# read the dataset
train_df = pd.read_csv("train.csv")
# observe the dataset
print(train_df.head(0)) # 5 columns: id, keyword, location, text, target
print(len(train_df)) # 7613 rows
print(train_df.shape) # (7613, 5)

# testing set
# set seed for the reproducibility
np.random.seed(2)
# read the dataset .tcv
test_df = pd.read_csv("test.csv")
# observe the dataset
print(test_df.head(0)) # 4 columns: id, keyword, location, text
print(len(test_df)) # 3263 rows
print(test_df.shape) # (3263, 4)

#task2: define a function to stem and tokenization
# print & read this part: https://scikit-learn.org/stable/modules/feature_extraction.html?highlight=tfidfvectorizer
# tokenization into words
# stemming into the roots words: we need phrase-level at least
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
# concat the two data frames --> we are using unsupervised learning with visuliazation in the following
frames = [train_df, test_df]
cluster_df = pd.concat(frames)
print(cluster_df.shape) # (10876, 5)
tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in cluster_df["text"]])
print(tfidf_matrix.shape)

# Program 2: labeling, by Snorkel API & nltk
# task 1: get the labels by hand first, and expand it by nlp library
#word lists about real disasters by hand
#data comes from top 100 tweets
ORGANIZATIONS = ["PDA","ITDRC","UMCOR","RECON","Rubicon","VOAD","IRUSA","community","aid"]
ACTIONS = ["affect","help","pray","impact","assemble","re-establish","monitor","rebuild","safe","relief","emergency"]
LOCATIONS = ["PuertoRico","warehouse","Kincade","Getty","shelter","Atlantic","Bahamas","Florida","Los","California"]
DISASTERS = ["Earthquake","aftershock","Hurricane","wildfires","HillsideFire","burn","tides","storm","flood","lightning","Tropical"]

ORGANIZATIONS_syn = []
for word in ORGANIZATIONS:
    syns = wordnet.synsets(word)
    for syn in syns:
        for l in syn.lemmas():
            ORGANIZATIONS_syn.append(l.name())
ACTIONS_syn = []
for word in ACTIONS:
    syns = wordnet.synsets(word)
    for syn in syns:
        for l in syn.lemmas():
            ACTIONS_syn.append(l.name())
LOCATIONS_syn = []
for word in LOCATIONS:
    syns = wordnet.synsets(word)
    for syn in syns:
        for l in syn.lemmas():
            LOCATIONS_syn.append(l.name())
DISASTERS_syn = []
for word in DISASTERS:
    syns = wordnet.synsets(word)
    for syn in syns:
        for l in syn.lemmas():
            DISASTERS_syn.append(l.name())
print(len(ORGANIZATIONS), len(ORGANIZATIONS_syn)) # expand from 9 to 38
print(len(ACTIONS), len(ACTIONS_syn)) # expand from 11 to 157
print(len(LOCATIONS), len(LOCATIONS_syn)) # expand from 10 to 26
print(len(DISASTERS), len(DISASTERS_syn)) # expand from 11 to 106

# task 2: write my label functions
TRUTH = 1
FAKE = 0
@labeling_function()
def LF_national_org(text):
    return TRUTH if "LF_national_org" in ORGANIZATIONS_syn else FAKE
@labeling_function()
def LF_action_help(text):
    return TRUTH if "LF_action_help" in ACTIONS_syn else FAKE
@labeling_function()
def LF_disaster_location(text):
    return TRUTH if "LF_disaster_location" in LOCATIONS_syn else FAKE
@labeling_function()
def LF_disaster_type(text):
    return TRUTH if "LF_disaster_type" in DISASTERS_syn else FAKE

# task 3: applying on labeling functions
#gather together the label functions and begin annotating
LFs = [LF_national_org, LF_action_help, LF_disaster_location, LF_disaster_type]
applier = PandasLFApplier(lfs=LFs)  # LF applier for a Pandas DataFrame
label_train = applier.apply(df=train_df)
label_test = applier.apply(df=test_df)
coverage_score_train = snorkel.labeling.LFAnalysis(L=label_train, lfs=LFs).label_coverage()
coverage_score_test = snorkel.labeling.LFAnalysis(L=label_test, lfs=LFs).label_coverage()
print(coverage_score_train, coverage_score_test)

# Program 3: import the enlarged label result data into mongoDB, by pymongo & json
#task1: observe the data
print(ORGANIZATIONS_syn) # the data type is still a list
print(ACTIONS_syn) # we should store it into the database
print(LOCATIONS_syn)
print(DISASTERS_syn)
#task2:store the data into .json file
d = {'org': ORGANIZATIONS_syn, 'act': ACTIONS_syn, 'loc': LOCATIONS_syn, 'dist': DISASTERS_syn}
print(d)
with open('label_syn.json', 'w') as f:
    json.dump(d,f)
#task3:connect with mongoDB and store the data
client = pymongo.MongoClient('localhost', 27017)
db = client.streamingtweets
tweets = db.files
with open('/Users/yue/Desktop/601mini2/label_syn.json') as f:
    file_data = json.load(f)
tweets.insert_one(file_data)

# Program 4: clustering, by sklearn & nltk
# task1: create a k-means clustering
# k value: the mean, usually
km = KMeans(n_clusters=2) # we only have 0&1
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
cluster_df["cluster"] = clusters
cluster_df['cluster'].value_counts()
# task2: calculating the similarity distance
similarity_distance = 1 - cosine_similarity(tfidf_matrix)

# Program 5: get the statistics result, and visualize it, by matplotlib
## for now, we don't need it because the coverage is 100%
# plt.hist()
# plt.show()
# Program3: Visualization
# task1: merging
mergings = linkage(similarity_distance, method='complete')
print(mergings)
# we need to find another way to compare the similarity
# # task2: ploting the dendrogram
# dendrogram_ = dendrogram(mergings,
#                labels=[x for x in cluster_df["id"]],
#                leaf_rotation=90,
#                leaf_font_size=16,
# )
# fig = plt.gcf() # adjusting
# axis = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
# fig.set_size_inches(108, 21)
# plt.show()