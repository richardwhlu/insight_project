# =========================================================================== #
# Date: 06/11/19
# Author: Richard Lu
# Description:
#   train word2vec word embeddings on business elite dataset
# Runtime (word2vec model building): 
#   0.0620 for 100 reviews
#   0.2591 for 1000 reviews
#   3.0689 for 10000 reviews
#   27.0248 for 100000 reviews
# Runtime (preprocessing):
#   12.9825 for 1000 reviews
#   157.5179 for 10000 reviews
#   1585.7034 for 100000 reviews
# =========================================================================== #

import matplotlib.pyplot as plt
import multiprocessing
import os
import pandas as pd
import string
import time

from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA

# =========================================================================== #
# READ DATA
# =========================================================================== #

input_data_folder = "../../0_data/2_processed_data"

data = pd.read_json(os.path.join(input_data_folder,
    "business_elite_subset_reviews.json")) # 265,308 reviews

# =========================================================================== #
# HELPER FUNCTIONS
# =========================================================================== #

# copied over
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


# =========================================================================== #
# PROCESS REVIEWS
# =========================================================================== #

# code help from https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

start0 = time.time()
train_sents_df = data["text"][:100000].apply(lambda x: preprocess_reviews(x,
    0, 0))
end0 = time.time()

print(end0 - start0)

print("starting model training")
start = time.time()
model = Word2Vec(train_sents_df.tolist(),
    size=100, window=5, min_count=2, workers=multiprocessing.cpu_count()-1,
    sg=0)
end = time.time()

print(end - start)

model.save("../../4_models/word2vec_embeddings_100K_business_elite_reviews.model")

# X = model[model.wv.vocab]
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)

# plt.scatter(result[:5, 0], result[:5, 1])
# words = list(model.wv.vocab)[:5]
# for i, word in enumerate(words):
#     plt.annotate(word, xy=(result[i, 0], result[i, 1]))

# plt.show()