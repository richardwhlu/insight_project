# =========================================================================== #
# Date: 06/05/19
# Author: Richard Lu
# Description:
#   clean the raw review text data and run unsupervised topic models
# Runtime: ~ 210s for 10000
# =========================================================================== #

import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import pickle
import string
import sys
import time

from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from sklearn.decomposition import (NMF,
    LatentDirichletAllocation as LDA, TruncatedSVD as LSA)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# stopwords.words('english')

# =========================================================================== #
# READ IN THE DATA
# =========================================================================== #

data_folder = "../../0_data/3_robustness_check_data_for_unsupervised_models"

data_filename = sys.argv[1]
num_topics = int(sys.argv[2])
# data_filename = "10191994_1000_influential.csv"
# num_topics = 20
try:
    num_words = int(sys.argv[3])
except:
    num_words = 10

data = pd.read_csv(os.path.join(
    data_folder, data_filename)).drop("Unnamed: 0", axis=1)

# =========================================================================== #
# DO SOME QUICK TEXT CLEANING
# =========================================================================== #

def process_reviews(text):
    """Clean up the raw review text a little.

    Args:
        text - raw review text from csv

    Returns:
        cleaned_sentence - tokenized and stemmed sentence
    """
    stemmer = PorterStemmer()
    tmp = [stemmer.stem(x.lower()) 
           for x in word_tokenize(text)
           if x.lower() not in stopwords.words("english")
           and x.lower() not in string.punctuation]
    cleaned_sentence = " ".join(tmp)
    return cleaned_sentence

start0 = time.time()
text = data["text"].apply(process_reviews)
end0 = time.time()

print(end0 - start0)

# =========================================================================== #
# GET MOST COMMON STEMS FOR GRAPHS
# =========================================================================== #

def get_most_common_words_across_sample_reviews(num_words, tmp_df):
    """Get most common words in a set of reviews

    Args:
        num_words - int number of top words to get
        tmp_df - review data to process

    Returns:
        most_common_words - top words with count
    """
    tf_vectorizer = CountVectorizer(max_df=0.95,
                                    min_df=2,
                                    max_features=1000)
    tf = tf_vectorizer.fit(tmp_df)
    tf2 = tf.transform(tmp_df)
    term_sums = tf2.sum(axis=0)
    words_freq = [(word, term_sums[0, index])
                  for word, index in tf.vocabulary_.items()]
    most_common_words = sorted(words_freq, key=lambda x: x[1], reverse=True
        )[:num_words]
    return most_common_words


most_common_words_list = get_most_common_words_across_sample_reviews(20, text)

words = [x[0] for x in most_common_words_list]
freq = [x[1] for x in most_common_words_list]
x_pos = np.arange(len(words)) 
  
plt.bar(x_pos, freq, align='center')
plt.xticks(x_pos, words) 
plt.ylabel('Frequency Count')
plt.xticks(rotation=90)
plt.savefig("../../3_reports/figures/{}_most_common_words".format(
    data_filename.split(".csv")[0]))

# =========================================================================== #
# CREATE MODELS
# =========================================================================== #

tf_vectorizer = CountVectorizer(max_df=0.95,
                                min_df=2,
                                max_features=1000)

tfidf_vectorizer = TfidfVectorizer(max_df=0.95,
                                   min_df=2,
                                   max_features=1000)

# add in truncated SVD here

# =========================================================================== #
# TRANSFORM DATA AND GET RESULTS
# =========================================================================== #

tf = tf_vectorizer.fit_transform(text)
tf_feature_names = tf_vectorizer.get_feature_names()

tfidf = tfidf_vectorizer.fit_transform(text)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

start1 = time.time()
lda = LDA(n_components=num_topics,
          max_iter=5,
          learning_method="online",
          learning_offset=50,
          random_state=10191994).fit(tf)
# save the LDA model
with open("../../4_models/lda_50K_influential_reviews_10191994.pickle", "wb") as f:
    pickle.dump(lda, f)


end1 = time.time()
print("LDA: {}".format(end1 - start1))

start2 = time.time()
nmf = NMF(n_components=num_topics,
          random_state=10191994,
          alpha=.1,
          l1_ratio=.5,
          init="nndsvd").fit(tfidf)
end2 = time.time()
print("NMF: {}".format(end2 - start2))

start3 = time.time()
lsa_tf = LSA(n_components=num_topics,
             random_state=10191994).fit(tf)
end3 = time.time()
print("LSA TF {}".format(end3 - start3))

start4 = time.time()
lsa_tfidf = LSA(n_components=num_topics,
                random_state=10191994).fit(tfidf)
end4 = time.time()
print("LSA TF {}".format(end4 - start4))

def display_topics(model, feature_names, num_top_words, model_string):
    """Display topics from the topic models

    Args:
        model - topic model
        feature_names - names of the words in each topic
        num_top_words - number of top words to display per topic
        model_string - name of model
    """
    with open("../../3_reports/0_unsupervised_models/{}.txt".format(
        data_filename.split(".csv")[0]), "a") as f:
        f.write("Model {}:\n".format(model_string))
        for topic_index, topic in enumerate(model.components_):
            f.write("Topic {}:".format(topic_index))
            f.write(" ".join([feature_names[i]
                       for i in topic.argsort()[:-num_top_words - 1:-1]]))

            f.write("\n\n")


# =========================================================================== #
# PRESENT RESULTS
# =========================================================================== #

display_topics(lda, tf_feature_names, num_words, "LDA")
display_topics(nmf, tfidf_feature_names, num_words, "NMF")
display_topics(lsa_tf, tf_feature_names, num_words, "LSA_TF")
display_topics(lsa_tfidf, tfidf_feature_names, num_words, "LSA_TFIDF")
