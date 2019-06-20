# =========================================================================== #
# Date: 06/13/19
# Author: Richard Lu
# Description:
#   run models on input data
# Runtime: 
# =========================================================================== #

import ast
import itertools as it
import multiprocessing
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
# LOAD MODELS
# =========================================================================== #

model = Word2Vec.load("flask_app/static/models/word2vec_embeddings_100K_business_elite_reviews.model")

tfidf_model = pickle.load(open("flask_app/static/models/tfidf_50K_influential_reviews_10191994.pickle", "rb"))

tfidf_bigram_model = pickle.load(
    open("flask_app/static/models/tfidf_bigram_50K_influential_reviews_10191994.pickle", "rb"))

# =========================================================================== #
# FEATURE EXTRACTION FUNCTIONS
# =========================================================================== #

def preprocess_reviews(review_text, stem, join=0, punct=1):
    """Clean up the raw review text a little.

    Args:
        review_text - raw review text from csv
        stem - stem tokens or not
        join - join the stems into sentence or not
        punct - 1 if remove punct, 0 otherwise

    Returns:
        cleaned_sentence - tokenized and stemmed sentence
    """
    stemmer = PorterStemmer()
    if stem:
        if punct:
            cleaned_sentence = [stemmer.stem(x.lower()) 
                   for x in word_tokenize(review_text)
                   if x.lower() not in stopwords.words("english")
                   and x.lower() not in string.punctuation]
        else:
            cleaned_sentence = [stemmer.stem(x.lower()) 
                   for x in word_tokenize(review_text)
                   if x.lower() not in stopwords.words("english")]

        if join:
            cleaned_sentence = " ".join(cleaned_sentence)
    else:
        if punct:
            cleaned_sentence = [x.lower() 
                   for x in word_tokenize(review_text)
                   if x.lower() not in stopwords.words("english")
                   and x.lower() not in string.punctuation]
        else:
            cleaned_sentence = [x.lower() 
                   for x in word_tokenize(review_text)
                   if x.lower() not in stopwords.words("english")] 


    return cleaned_sentence


# CATEGORY 1: Metadata
def process_metadata(review_text):
    """Process for metadata elements of reviews [3 features]

    Args:
        review_text - raw review text from csv

    Returns:
        metadata_list - list of metadata attributes from original reviews
    """
    cleaned_sentence = review_text
    # cleaned_sentence = preprocess_reviews(review_text, 0)
    
    wordlength_sentence = []
    money_punct_sentence = 0
    numbers_sentence = 0
    for word in cleaned_sentence:
        wordlength_sentence.append(len(word))
        if "$" in word or "%" in word or "+" in word or "-" in word:
            money_punct_sentence += 1

        numbers_sentence += len(re.findall("[0-9]", word))

    sentence_length = np.sum(wordlength_sentence)
    sentence_avg_length = np.mean(wordlength_sentence)
    sentence_med_length = np.median(wordlength_sentence)


    metadata_list = [sentence_length,
                     sentence_avg_length,
                     sentence_med_length,
                     money_punct_sentence,
                     numbers_sentence]

    return metadata_list


# CATEGORY 2: Syntactic Structure
def process_syntax(review_text):
    """Process for syntactic structure [144 features]

    Args:
        review_text - raw review_text from csv

    Returns:
        pos_list - list of parts of speech counts
    """
    cleaned_sentence = review_text
    # cleaned_sentence = preprocess_reviews(review_text, 0)

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
        tfidf_bigram_df - df of tfidf tokens for bigrams
    """
    tfidf = tfidf_model.transform(review_df) # sparse matrix
    tfidf_feature_names = tfidf_model.get_feature_names()

    tfidf_df = pd.DataFrame(tfidf.toarray())
    tfidf_df.columns = tfidf_feature_names

    tfidf_bigram = tfidf_bigram_model.transform(review_df)
    tfidf_bigram_feature_names = tfidf_bigram_model.get_feature_names()

    tfidf_bigram_df = pd.DataFrame(tfidf_bigram.toarray())
    tfidf_bigram_df.columns = tfidf_bigram_feature_names

    return tfidf_df, tfidf_bigram_df


# CATEGORY 4: Topic model outputs
def process_topic_models(review_df):
    """Process the top 100 unsupervised topics from 50K lda model
    [100 features]

    Args:
        review_df - df of raw review_text from csv

    Returns:
        topic_df - topic space df of texts
    """
    with open("flask_app/static/models/lda_50K_influential_reviews_10191994.pickle",
        "rb") as f:
        lda = pickle.load(f)

        with open("flask_app/static/models/tf_50K_influential_reviews_10191994_full.pickle",
            "rb") as f2:
            tf_vectorizer = pickle.load(f2)

            review_tf = tf_vectorizer.transform(review_df)

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
    cleaned_sentence = review_text
    # cleaned_sentence = preprocess_reviews(review_text, 0)
    stemmer = PorterStemmer()
    wordlist = ["food",
                "delicious",
                "service",
                "staff",
                "price",
                "cost",
                "ambiance",
                "patio",
                "loud"]

    similarity_dict = defaultdict(list)
    for word in cleaned_sentence:
        for target_word in wordlist:
            try:
                other_words = [stemmer.stem(x)
                               for x in wordlist if x is not target_word]
                if stemmer.stem(word) not in other_words:
                    similarity_dict[target_word].append(
                        (word, model.wv.similarity(word, target_word)))
            except:
                pass
        
    most_similar_dict = {}
    for target_word, similarity_list in similarity_dict.items():
        sorted_similarity_list = sorted(similarity_list, key=lambda x: -x[1])
        most_similar_dict[target_word] = np.mean(
            [x[1] for x in sorted_similarity_list[:3]])
    
    embeddings_list = []
    for target_word in wordlist:
        embeddings_list.append(most_similar_dict.get(target_word, 0))
        
    return embeddings_list


# Category 6: Count vector (maybe binary?) normalized
def process_counts(review_df):
    """Process for normalized counts of common words [20 features]
    EXCLUDED FOR NOW
    IF USING IN FUTURE, NEED TO INCORPORATE GLOBAL TF MODEL

    Args:
        review_df - df of raw review_text from csv

    Returns:
        count_df - df of counts
    """
    tf_vectorizer = CountVectorizer(max_df=0.8,
                                    min_df=10,
                                    max_features=20)
    tf = tf_vectorizer.fit(review_df)
    tf2 = tf.transform(review_df)
    words = [word for word, index in tf.vocabulary_.items()]

    tf_df = pd.DataFrame(tf2.toarray())
    tf_df.columns = words

    return tf_df


def normalize(col):
    """Normalize the col to gaussian

    Args:
        col - input column

    Returns:
        col_n - normalized column
    """
    tmp = []
    for row in col:
        try:
            tmp.append((row - np.mean(col)) / np.std(col))
        except:
            tmp.append(row - np.mean(col))

    return tmp


def process_rows(data_in):
    """Process rows function for parallelization

    Args:
        data_in - df of text

    Returns:
        parsed and formatted text df
    """
    columns = data_in.columns
    data_in["text_formatted"] = data_in[columns[1]].apply(
        lambda x: preprocess_reviews(x, 1, 1))
    return data_in[["index", "text_formatted"]]


def process_rows2(data_in):
    """Process rows function for parallelization

    Args:
        data_in - df of text

    Returns:
        parsed and formatted text df
    """
    columns = data_in.columns
    data_in["text_formatted"] = data_in[columns[1]].apply(
        lambda x: preprocess_reviews(x, 0))
    return data_in[["index", "text_formatted"]]


def process_rows3(data_in):
    """Process rows function for parallelization

    Args:
        data_in - df of text

    Returns:
        parsed and formatted text df
    """
    columns = data_in.columns
    data_in["text_formatted"] = data_in[columns[1]].apply(
        lambda x: preprocess_reviews(x, 0, punct=0))
    return data_in[["index", "text_formatted"]]


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

    tmp_data = data[[column_name]]
    tmp_data["index"] = range(0, tmp_data.shape[0])
    tmp_data = tmp_data[["index", column_name]]

    # df_split = np.array_split(tmp_data, 14)

    # start0 = time.time()
    # pool = multiprocessing.Pool(multiprocessing.cpu_count())
    # preprocessed_df = pd.concat(pool.map(process_rows, df_split))
    # pool.close()
    # pool.join()
    # end0 = time.time()

    # print("Preprocessing Stem & Join (P): {}".format(end0 - start0))

    # start00 = time.time()
    # pool = multiprocessing.Pool(multiprocessing.cpu_count())
    # preprocessed_df2 = pd.concat(pool.map(process_rows2, df_split))
    # pool.close()
    # pool.join()
    # end00 = time.time()

    # print("Preprocessing No Stem (P): {}".format(end00 - start00))

    # start000 = time.time()
    # pool = multiprocessing.Pool(multiprocessing.cpu_count())
    # preprocessed_df3 = pd.concat(pool.map(process_rows3, df_split))
    # pool.close()
    # pool.join()
    # end000 = time.time()

    # print("Preprocessing No Stem & Punct (P): {}".format(end000 - start000))

    start0 = time.time()
    preprocessed_df = process_rows(tmp_data)
    preprocessed_df2 = process_rows2(tmp_data)
    preprocessed_df3 = process_rows3(tmp_data)
    end0 = time.time()

    print("Preprocessing (P): {}".format(end0 - start0))

    if preprocessed_df["index"].tolist() == list(
        range(preprocessed_df.shape[0])) and (preprocessed_df2["index"
        ].tolist() == list(range(preprocessed_df2.shape[0]))) and (
        preprocessed_df3["index"].tolist() == list(
            range(preprocessed_df3.shape[0]))):
        preprocessed_df = preprocessed_df["text_formatted"]
        preprocessed_df2 = preprocessed_df2["text_formatted"]
        preprocessed_df3 = preprocessed_df3["text_formatted"]
    else:
        return

    # perhaps can parallelize functions
    start1 = time.time()
    tfidf_df, tfidf_bigram_df = process_tfidf(preprocessed_df)
    end1 = time.time()

    print("TFIDF: {}".format(end1 - start1))

    start2 = time.time()
    topic_df = process_topic_models(preprocessed_df)
    end2 = time.time()

    print("Topics: {}".format(end2 - start2))

    # just use tfidf for now
    # start3 = time.time()
    # tf_df = process_counts(preprocessed_df)
    # end3 = time.time()

    # print("TF: {}".format(end3 - start3))

    start4 = time.time()
    (data["sen_len"],
     data["sen_avg_len"],
     data["sen_med_len"],
     data["sen_money_len"],
     data["sen_num_len"]) = zip(*preprocessed_df3.apply(process_metadata))
    end4 = time.time()

    print("Metadata: {}".format(end4 - start4))

    start5 = time.time()
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
     data['X_CONJ']) = zip(*preprocessed_df2.apply(process_syntax))

    metadata_columns = ['NOUN', 'VERB', 'ADJ', 'ADV', '.', 'ADP', 'NUM', 'PRON', 'PRT', 'DET', 'CONJ', 'X', 'NOUN_VERB', 'NOUN_ADJ', 'NOUN_ADV', 'NOUN_.', 'NOUN_ADP', 'NOUN_NUM', 'NOUN_PRON', 'NOUN_PRT', 'NOUN_DET', 'NOUN_CONJ', 'NOUN_X', 'VERB_NOUN', 'VERB_ADJ', 'VERB_ADV', 'VERB_.', 'VERB_ADP', 'VERB_NUM', 'VERB_PRON', 'VERB_PRT', 'VERB_DET', 'VERB_CONJ', 'VERB_X', 'ADJ_NOUN', 'ADJ_VERB', 'ADJ_ADV', 'ADJ_.', 'ADJ_ADP', 'ADJ_NUM', 'ADJ_PRON', 'ADJ_PRT', 'ADJ_DET', 'ADJ_CONJ', 'ADJ_X', 'ADV_NOUN', 'ADV_VERB', 'ADV_ADJ', 'ADV_.', 'ADV_ADP', 'ADV_NUM', 'ADV_PRON', 'ADV_PRT', 'ADV_DET', 'ADV_CONJ', 'ADV_X', '._NOUN', '._VERB', '._ADJ', '._ADV', '._ADP', '._NUM', '._PRON', '._PRT', '._DET', '._CONJ', '._X', 'ADP_NOUN', 'ADP_VERB', 'ADP_ADJ', 'ADP_ADV', 'ADP_.', 'ADP_NUM', 'ADP_PRON', 'ADP_PRT', 'ADP_DET', 'ADP_CONJ', 'ADP_X', 'NUM_NOUN', 'NUM_VERB', 'NUM_ADJ', 'NUM_ADV', 'NUM_.', 'NUM_ADP', 'NUM_PRON', 'NUM_PRT', 'NUM_DET', 'NUM_CONJ', 'NUM_X', 'PRON_NOUN', 'PRON_VERB', 'PRON_ADJ', 'PRON_ADV', 'PRON_.', 'PRON_ADP', 'PRON_NUM', 'PRON_PRT', 'PRON_DET', 'PRON_CONJ', 'PRON_X', 'PRT_NOUN', 'PRT_VERB', 'PRT_ADJ', 'PRT_ADV', 'PRT_.', 'PRT_ADP', 'PRT_NUM', 'PRT_PRON', 'PRT_DET', 'PRT_CONJ', 'PRT_X', 'DET_NOUN', 'DET_VERB', 'DET_ADJ', 'DET_ADV', 'DET_.', 'DET_ADP', 'DET_NUM', 'DET_PRON', 'DET_PRT', 'DET_CONJ', 'DET_X', 'CONJ_NOUN', 'CONJ_VERB', 'CONJ_ADJ', 'CONJ_ADV', 'CONJ_.', 'CONJ_ADP', 'CONJ_NUM', 'CONJ_PRON', 'CONJ_PRT', 'CONJ_DET', 'CONJ_X', 'X_NOUN', 'X_VERB', 'X_ADJ', 'X_ADV', 'X_.', 'X_ADP', 'X_NUM', 'X_PRON', 'X_PRT', 'X_DET', 'X_CONJ']
    
    for column in metadata_columns:
        data[column] = normalize(data[column])
    end5 = time.time()

    print("Syntax: {}".format(end5 - start5))

    start6 = time.time()
    (data["food_3avg_sim"],
     data["delicious_3avg_sim"],
     data["service_3avg_sim"],
     data["staff_3avg_sim"],
     data["price_3avg_similarity"],
     data["cost_3avg_similarity"],
     data["ambiance_3avg_similarity"],
     data["patio_3avg_similarity"],
     data["loud_3avg_similarity"]) = zip(*preprocessed_df2.apply(
        process_word_embeddings))
    end6 = time.time()

    # scale the word embedding columns
    data["food_3avg_sim"] = normalize(data["food_3avg_sim"])
    data["delicious_3avg_sim"] = normalize(data["delicious_3avg_sim"])
    data["service_3avg_sim"] = normalize(data["service_3avg_sim"])
    data["staff_3avg_sim"] = normalize(data["staff_3avg_sim"])
    data["price_3avg_similarity"] = normalize(data["price_3avg_similarity"])
    data["cost_3avg_similarity"] = normalize(data["cost_3avg_similarity"])
    data["ambiance_3avg_similarity"] = normalize(data["ambiance_3avg_similarity"])
    data["patio_3avg_similarity"] = normalize(data["patio_3avg_similarity"])
    data["loud_3avg_similarity"] = normalize(data["loud_3avg_similarity"])

    print("Embeddings: {}".format(end6 - start6))


    feature_matrix = pd.concat([data, tfidf_df, tfidf_bigram_df, topic_df], axis=1)

    return feature_matrix