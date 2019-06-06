# =========================================================================== #
# Date: 06/05/19
# Author: Richard Lu
# Description:
#   clean the raw review text data and run unsupervised topic models
# Runtime: 
# =========================================================================== #

import nltk
import os
import pandas as pd
import string
import sys

from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from sklearn.decomposition import NMF, LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# stopwords.words('english')

# =========================================================================== #
# READ IN THE DATA
# =========================================================================== #

data_folder = "../../0_data/3_robustness_check_data_for_unsupervised_models"

data_filename = sys.argv[1]
num_topics = int(sys.argv[2])
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


text = data["text"].apply(process_reviews)

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

lda = LDA(n_components=num_topics,
          max_iter=5,
          learning_method="online",
          learning_offset=50,
          random_state=10191994).fit(tf)

nmf = NMF(n_components=num_topics,
          random_state=10191994,
          alpha=.1,
          l1_ratio=.5,
          init="nndsvd").fit(tfidf)

def display_topics(model, feature_names, num_top_words, model_string):
    """Display topics from the topic models

    Args:
        model - topic model
        feature_names - names of the words in each topic
        num_top_words - number of top words to display per topic
        model_string - name of model
    """
    print("Model {}:".format(model_string))
    for topic_index, topic in enumerate(model.components_):
        print("Topic {}:".format(topic_index))
        print(" ".join([feature_names[i]
                       for i in topic.argsort()[:-num_top_words - 1:-1]]))


# =========================================================================== #
# PRESENT RESULTS
# =========================================================================== #

display_topics(nmf, tfidf_feature_names, 10, "NMF")
display_topics(lda, tf_feature_names, 10, "LDA")