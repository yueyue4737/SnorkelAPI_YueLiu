# !/usr/bin/env/python3
# Copyright 2019 YueLiu liuyue2@bu.edu
# Program Goal: Calculating the similarity by using clustering method

# import libraries
import matplotlib.pyplot as plt
import nltk
from nltk.stem.snowball import SnowballStemmer # PorterStemmer is also ok
import numpy as np
import pandas as pd
import re
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Program1: data pre-processing
#task1: narrow down the scale of the dataset
# trainging set
# read the dataset
train_df = pd.read_csv("train.csv")
# obeserve the dataset
print(train_df.head(0)) # 5 columns: id, keyword, location, text, target

# texting set
# read the dataset .tcv
test_df = pd.read_csv("test.csv")
# obeserve the dataset
print(test_df.head(0)) # 4 columns: id, keyword, location, text

# concat the two data frames --> we are using unsupervised learning with visuliazation in the following
frames = [train_df, test_df]
cluster_df = pd.concat(frames)
print(cluster_df.shape) # (10876, 5)


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
tfidf_matrix = tfidf_vectorizer.fit_transform([x for x in cluster_df["text"]])
print(tfidf_matrix.shape)

# Program2: k-means clustering
# task1: create a k-means clustering
# k value: the mean, usually
km = KMeans(n_clusters=2)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
cluster_df["cluster"] = clusters
cluster_df['cluster'].value_counts()
# task2: calculating the similarity distance
similarity_distance = 1 - cosine_similarity(tfidf_matrix)

# Program3: Visualization
# task1: merging
mergings = linkage(similarity_distance, method='complete')
# task2: ploting the dendrogram
dendrogram_ = dendrogram(mergings,
               labels=[x for x in cluster_df["id"]],
               leaf_rotation=90,
               leaf_font_size=16,
)
fig = plt.gcf() # adjusting
axis = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
fig.set_size_inches(108, 21)
plt.show()
# task3: read the similarity
