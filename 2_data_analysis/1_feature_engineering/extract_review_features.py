# =========================================================================== #
# Date: 06/10/19
# Author: Richard Lu
# Description:
#   extract text features from yelp reviews for ML
# Runtime: 
# =========================================================================== #

import ast
import itertools as it
import nltk
import numpy as np
import os
import pandas as pd
import pickle
import string
import time

from collections import defaultdict
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# =========================================================================== #
# READ DATA
# =========================================================================== #

input_data_folder = "../../0_data/4_handlabeled_data"

data = pd.read_csv(os.path.join(input_data_folder,
    "handlabeled_reviews.csv"))

data["text_formatted"] = data["text"].apply(ast.literal_eval)

data2 = data[["text_formatted"]]
test_review = data2.iloc[0]

model = Word2Vec.load("../../4_models/word2vec_embeddings_100K_business_elite_reviews.model")

# =========================================================================== #
# FEATURE EXTRACTION FUNCTIONS
# =========================================================================== #

def preprocess_reviews(review_text, stem, join=0):
    """Clean up the raw review text a little.

    Args:
        review_text - raw review text from csv
        stem - stem tokens or not
        join - join the stems into sentence or not

    Returns:
        cleaned_sentence - tokenized and stemmed sentence
    """
    stemmer = PorterStemmer()
    if stem:
        cleaned_sentence = [stemmer.stem(x.lower()) 
               for x in word_tokenize(review_text)
               if x.lower() not in stopwords.words("english")
               and x.lower() not in string.punctuation]
        if join:
            cleaned_sentence = " ".join(cleaned_sentence)
    else:
        cleaned_sentence = [x.lower() 
               for x in word_tokenize(review_text)
               if x.lower() not in stopwords.words("english")
               and x.lower() not in string.punctuation]
    return cleaned_sentence


# CATEGORY 1: Metadata
def process_metadata(review_text):
    """Process for metadata elements of reviews [3 features]

    Args:
        review_text - raw review text from csv

    Returns:
        metadata_list - list of metadata attributes from original reviews
    """
    cleaned_sentence = preprocess_reviews(review_text, 0)
    
    wordlength_sentence = [len(x) for x in cleaned_sentence]
    sentence_length = np.sum(wordlength_sentence)
    sentence_avg_length = np.mean(wordlength_sentence)
    sentence_med_length = np.median(wordlength_sentence)

    metadata_list = [sentence_length,
                     sentence_avg_length,
                     sentence_med_length]

    return metadata_list


# CATEGORY 2: Syntactic Structure
def process_syntax(review_text):
    """Process for syntactic structure [144 features]

    Args:
        review_text - raw review_text from csv

    Returns:
        pos_list - list of parts of speech counts
    """
    cleaned_sentence = preprocess_reviews(review_text, 0)

    pos_sentence = nltk.pos_tag(cleaned_sentence, tagset="universal")

    # {'NOUN': 23286, 'VERB': 12851, 'ADJ': 12356, 'ADV': 5937, '.': 676, 'ADP': 1068, 'NUM': 1526, 'PRON': 245, 'PRT': 772, 'DET': 206, 'CONJ': 70, 'X': 50})
    pos_dict = defaultdict(int)

    for word_pos1, word_pos2 in zip(pos_sentence,
                                    pos_sentence[1:] + [("", "")]):
        pos_dict[word_pos1[1]] += 1
        pos_dict[(word_pos1[1], word_pos2[1])] += 1
    

    uni_pos_list = ["NOUN", "VERB", "ADJ", "ADV", ".", "ADP", "NUM",
                    "PRON", "PRT", "DET", "CONJ", "X"]

    bi_pos_list = []

    for perm in it.permutations(uni_pos_list, 2):
        bi_pos_list.append(perm)


    pos_list = uni_pos_list + bi_pos_list 
    # ['NOUN', 'VERB', 'ADJ', 'ADV', '.', 'ADP', 'NUM', 'PRON', 'PRT', 'DET', 'CONJ', 'X', ('NOUN', 'VERB'), ('NOUN', 'ADJ'), ('NOUN', 'ADV'), ('NOUN', '.'), ('NOUN', 'ADP'), ('NOUN', 'NUM'), ('NOUN', 'PRON'), ('NOUN', 'PRT'), ('NOUN', 'DET'), ('NOUN', 'CONJ'), ('NOUN', 'X'), ('VERB', 'NOUN'), ('VERB', 'ADJ'), ('VERB', 'ADV'), ('VERB', '.'), ('VERB', 'ADP'), ('VERB', 'NUM'), ('VERB', 'PRON'), ('VERB', 'PRT'), ('VERB', 'DET'), ('VERB', 'CONJ'), ('VERB', 'X'), ('ADJ', 'NOUN'), ('ADJ', 'VERB'), ('ADJ', 'ADV'), ('ADJ', '.'), ('ADJ', 'ADP'), ('ADJ', 'NUM'), ('ADJ', 'PRON'), ('ADJ', 'PRT'), ('ADJ', 'DET'), ('ADJ', 'CONJ'), ('ADJ', 'X'), ('ADV', 'NOUN'), ('ADV', 'VERB'), ('ADV', 'ADJ'), ('ADV', '.'), ('ADV', 'ADP'), ('ADV', 'NUM'), ('ADV', 'PRON'), ('ADV', 'PRT'), ('ADV', 'DET'), ('ADV', 'CONJ'), ('ADV', 'X'), ('.', 'NOUN'), ('.', 'VERB'), ('.', 'ADJ'), ('.', 'ADV'), ('.', 'ADP'), ('.', 'NUM'), ('.', 'PRON'), ('.', 'PRT'), ('.', 'DET'), ('.', 'CONJ'), ('.', 'X'), ('ADP', 'NOUN'), ('ADP', 'VERB'), ('ADP', 'ADJ'), ('ADP', 'ADV'), ('ADP', '.'), ('ADP', 'NUM'), ('ADP', 'PRON'), ('ADP', 'PRT'), ('ADP', 'DET'), ('ADP', 'CONJ'), ('ADP', 'X'), ('NUM', 'NOUN'), ('NUM', 'VERB'), ('NUM', 'ADJ'), ('NUM', 'ADV'), ('NUM', '.'), ('NUM', 'ADP'), ('NUM', 'PRON'), ('NUM', 'PRT'), ('NUM', 'DET'), ('NUM', 'CONJ'), ('NUM', 'X'), ('PRON', 'NOUN'), ('PRON', 'VERB'), ('PRON', 'ADJ'), ('PRON', 'ADV'), ('PRON', '.'), ('PRON', 'ADP'), ('PRON', 'NUM'), ('PRON', 'PRT'), ('PRON', 'DET'), ('PRON', 'CONJ'), ('PRON', 'X'), ('PRT', 'NOUN'), ('PRT', 'VERB'), ('PRT', 'ADJ'), ('PRT', 'ADV'), ('PRT', '.'), ('PRT', 'ADP'), ('PRT', 'NUM'), ('PRT', 'PRON'), ('PRT', 'DET'), ('PRT', 'CONJ'), ('PRT', 'X'), ('DET', 'NOUN'), ('DET', 'VERB'), ('DET', 'ADJ'), ('DET', 'ADV'), ('DET', '.'), ('DET', 'ADP'), ('DET', 'NUM'), ('DET', 'PRON'), ('DET', 'PRT'), ('DET', 'CONJ'), ('DET', 'X'), ('CONJ', 'NOUN'), ('CONJ', 'VERB'), ('CONJ', 'ADJ'), ('CONJ', 'ADV'), ('CONJ', '.'), ('CONJ', 'ADP'), ('CONJ', 'NUM'), ('CONJ', 'PRON'), ('CONJ', 'PRT'), ('CONJ', 'DET'), ('CONJ', 'X'), ('X', 'NOUN'), ('X', 'VERB'), ('X', 'ADJ'), ('X', 'ADV'), ('X', '.'), ('X', 'ADP'), ('X', 'NUM'), ('X', 'PRON'), ('X', 'PRT'), ('X', 'DET'), ('X', 'CONJ')]

    pos_list = [pos_dict.get(x, 0) for x in pos_list]

    return pos_list


# CATEGORY 3: TF-IDF top words
def process_tfidf(review_df):
    """Process for most common tfidf tokens [100 features]

    Args:
        review_df - df of raw review_text from csv

    Returns:
        tfidf_df - df of tfidf tokens
    """
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95,
                                   min_df=2,
                                   max_features=100)

    tfidf = tfidf_vectorizer.fit_transform(review_df) # sparse matrix
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    tfidf_df = pd.DataFrame(tfidf.toarray())
    tfidf_df.columns = tfidf_feature_names

    return tfidf_df


# CATEGORY 4: Topic model outputs
def process_topic_models(review_df):
    """Process the top 100 unsupervised topics from 50K lda model
    [100 features]

    Args:
        review_df - df of raw review_text from csv

    Returns:
        topic_df - topic space df of texts
    """
    with open("../../4_models/lda_50K_influential_reviews_10191994.pickle",
        "rb") as f:
        lda = pickle.load(f)

        tf_vectorizer = CountVectorizer(max_df=0.95,
                                        min_df=2,
                                        max_features=1000)

        review_tf = tf_vectorizer.fit_transform(review_df)

        topic_df = pd.DataFrame(lda.transform(review_tf))

        topic_df.columns = ["topic_{}".format(x) for x in range(100)]

        return topic_df


# CATEGORY 5: Word embeddings
def process_word_embeddings(review_text):
    """Process similarity of word embeddings to key words [4 features]
    # can do a lot more here with embeddings space and related words

    Args:
        review_text - raw review_text from csv

    Returns:
        embeddings_list - list of avg cosine similarity to focal words
    """
    cleaned_sentence = preprocess_reviews(review_text, 0)

    food_similarity = []
    service_similarity = []
    price_similarity = []
    ambiance_similarity = []

    for word in cleaned_sentence:
        try:
            tmp = model.wv.similarity(word, "food")
            food_similarity.append(tmp)
        except:
            pass


        try:
            tmp = model.wv.similarity(word, "service")
            service_similarity.append(tmp)
        except:
            pass


        try:
            tmp = model.wv.similarity(word, "price")
            price_similarity.append(tmp)
        except:
            pass


        try:
            tmp = model.wv.similarity(word, "ambiance")
            ambiance_similarity.append(tmp)
        except:
            pass


    food_avg_similarity = np.mean(food_similarity)
    service_avg_similarity = np.mean(service_similarity)
    price_avg_similarity = np.mean(price_similarity)
    ambiance_avg_similarity = np.mean(ambiance_similarity)
    
    embeddings_list = [food_avg_similarity,
                       service_avg_similarity,
                       price_avg_similarity,
                       ambiance_avg_similarity]

    return embeddings_list


def normalize(col):
    """Normalize the col to gaussian

    Args:
        col - input column

    Returns:
        col_n - normalized column
    """
    tmp = []
    for row in col:
        tmp.append((row - np.mean(col)) / np.std(col))


    return tmp


# =========================================================================== #
# PRODUCE FEATURE MATRIX
# =========================================================================== #

# need to wrap this in a function to get single or multiple reviews
def produce_feature_matrix(data):
    """Produces the feature matrix from an input df of text reviews

    Args:
        data - df of a single column with text reviews

    Returns:
        feature_matrix - df of features extracted from text
    """
    column = data.columns

    if len(column) != 1:
        print("too many columns")
        return


    column_name = column[0]

    preprocessed_df = data[column_name].apply(lambda x:
        preprocess_reviews(x, 1, 1))

    # perhaps can parallelize functions
    tfidf_df = process_tfidf(preprocessed_df)
    topic_df = process_topic_models(preprocessed_df)

    (data["sen_len"],
     data["sen_avg_len"],
     data["sen_med_len"]) = zip(*data[column_name].apply(process_metadata))

    (data['NOUN'],
     data['VERB'],
     data['ADJ'],
     data['ADV'],
     data['.'],
     data['ADP'],
     data['NUM'],
     data['PRON'],
     data['PRT'],
     data['DET'],
     data['CONJ'],
     data['X'],
     data['NOUN_VERB'],
     data['NOUN_ADJ'],
     data['NOUN_ADV'],
     data['NOUN_.'],
     data['NOUN_ADP'],
     data['NOUN_NUM'],
     data['NOUN_PRON'],
     data['NOUN_PRT'],
     data['NOUN_DET'],
     data['NOUN_CONJ'],
     data['NOUN_X'],
     data['VERB_NOUN'],
     data['VERB_ADJ'],
     data['VERB_ADV'],
     data['VERB_.'],
     data['VERB_ADP'],
     data['VERB_NUM'],
     data['VERB_PRON'],
     data['VERB_PRT'],
     data['VERB_DET'],
     data['VERB_CONJ'],
     data['VERB_X'],
     data['ADJ_NOUN'],
     data['ADJ_VERB'],
     data['ADJ_ADV'],
     data['ADJ_.'],
     data['ADJ_ADP'],
     data['ADJ_NUM'],
     data['ADJ_PRON'],
     data['ADJ_PRT'],
     data['ADJ_DET'],
     data['ADJ_CONJ'],
     data['ADJ_X'],
     data['ADV_NOUN'],
     data['ADV_VERB'],
     data['ADV_ADJ'],
     data['ADV_.'],
     data['ADV_ADP'],
     data['ADV_NUM'],
     data['ADV_PRON'],
     data['ADV_PRT'],
     data['ADV_DET'],
     data['ADV_CONJ'],
     data['ADV_X'],
     data['._NOUN'],
     data['._VERB'],
     data['._ADJ'],
     data['._ADV'],
     data['._ADP'],
     data['._NUM'],
     data['._PRON'],
     data['._PRT'],
     data['._DET'],
     data['._CONJ'],
     data['._X'],
     data['ADP_NOUN'],
     data['ADP_VERB'],
     data['ADP_ADJ'],
     data['ADP_ADV'],
     data['ADP_.'],
     data['ADP_NUM'],
     data['ADP_PRON'],
     data['ADP_PRT'],
     data['ADP_DET'],
     data['ADP_CONJ'],
     data['ADP_X'],
     data['NUM_NOUN'],
     data['NUM_VERB'],
     data['NUM_ADJ'],
     data['NUM_ADV'],
     data['NUM_.'],
     data['NUM_ADP'],
     data['NUM_PRON'],
     data['NUM_PRT'],
     data['NUM_DET'],
     data['NUM_CONJ'],
     data['NUM_X'],
     data['PRON_NOUN'],
     data['PRON_VERB'],
     data['PRON_ADJ'],
     data['PRON_ADV'],
     data['PRON_.'],
     data['PRON_ADP'],
     data['PRON_NUM'],
     data['PRON_PRT'],
     data['PRON_DET'],
     data['PRON_CONJ'],
     data['PRON_X'],
     data['PRT_NOUN'],
     data['PRT_VERB'],
     data['PRT_ADJ'],
     data['PRT_ADV'],
     data['PRT_.'],
     data['PRT_ADP'],
     data['PRT_NUM'],
     data['PRT_PRON'],
     data['PRT_DET'],
     data['PRT_CONJ'],
     data['PRT_X'],
     data['DET_NOUN'],
     data['DET_VERB'],
     data['DET_ADJ'],
     data['DET_ADV'],
     data['DET_.'],
     data['DET_ADP'],
     data['DET_NUM'],
     data['DET_PRON'],
     data['DET_PRT'],
     data['DET_CONJ'],
     data['DET_X'],
     data['CONJ_NOUN'],
     data['CONJ_VERB'],
     data['CONJ_ADJ'],
     data['CONJ_ADV'],
     data['CONJ_.'],
     data['CONJ_ADP'],
     data['CONJ_NUM'],
     data['CONJ_PRON'],
     data['CONJ_PRT'],
     data['CONJ_DET'],
     data['CONJ_X'],
     data['X_NOUN'],
     data['X_VERB'],
     data['X_ADJ'],
     data['X_ADV'],
     data['X_.'],
     data['X_ADP'],
     data['X_NUM'],
     data['X_PRON'],
     data['X_PRT'],
     data['X_DET'],
     data['X_CONJ']) = zip(*data[column_name].apply(process_syntax))

    (data["food_avg_sim"],
     data["service_avg_sim"],
     data["price_avg_sim"],
     data["ambiance_avg_sim"]) = zip(*data[column_name].apply(
        process_word_embeddings))

    # scale the word embedding columns
    data["food_avg_sim"] = normalize(data["food_avg_sim"])
    data["service_avg_sim"] = normalize(data["service_avg_sim"])
    data["price_avg_sim"] = normalize(data["price_avg_sim"])
    data["ambiance_avg_sim"] = normalize(data["ambiance_avg_sim"])

    feature_matrix = pd.concat([data, tfidf_df, topic_df], axis=1)

    return feature_matrix