# =========================================================================== #
# Date: 06/12/19
# Author: Richard Lu
# Description:
#   train classification model
# Runtime: 
# =========================================================================== #

import copy
import glob
import logging
import numpy as np
import os
import pickle
import random
import ujson
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score, cross_val_predict

# =========================================================================== #
# RUN DEPENDENCY SCRIPTS
# =========================================================================== #

warnings.filterwarnings("ignore")

os.chdir("../1_feature_engineering")
exec(open("extract_review_features.py").read())
exec(open("iterative_stratification.py").read())

handlabeled = ujson.load(open(
    "../../0_data/4_handlabeled_data/already_handlabeled_review_ids.json",
    "r"))

handlabeled_set = set(handlabeled["labeled_ids"])

static_path = "/home/richard/Documents/work/insight/insight_project/0_data/2_processed_data"

review_filename = "review_count_by_business_id.csv"

review_count_data = pd.read_csv(os.path.join(static_path, review_filename)
    ).drop("Unnamed: 0", axis=1)

review_count_data["cum_sum"] = review_count_data["review_count"].cumsum()

max_review = review_count_data["review_count"].sum()

logging.basicConfig(filename="train_model.log", filemode="a",
level=logging.INFO,
format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")

logging.info("Start script")

# =========================================================================== #
# TRAIN MODEL
# =========================================================================== #

feature_matrix = produce_feature_matrix(data2)
# all nas seem to be coming from syntax bigrams
# [('ADV_X', 1050), ('ADP_CONJ', 1050), ('NUM_X', 1050), ('PRON_CONJ', 1050), ('PRON_X', 1050), ('PRT_PRON', 1050), ('PRT_CONJ', 1050), ('PRT_X', 1050), ('DET_ADP', 1050), ('DET_PRON', 1050), ('DET_PRT', 1050), ('DET_CONJ', 1050), ('DET_X', 1050), ('CONJ_ADP', 1050), ('CONJ_PRON', 1050), ('CONJ_PRT', 1050), ('CONJ_DET', 1050), ('CONJ_X', 1050), ('X_ADP', 1050), ('X_PRON', 1050), ('X_PRT', 1050), ('X_DET', 1050), ('X_CONJ', 1050)]

# fill na with 0 just in case other reviews have those combinations 

# na_columns = [x[0] for x in zip(feature_matrix.columns,
#                                 feature_matrix.isnull().sum().tolist())
#               if x[1] > 0]
feature_matrix = feature_matrix.fillna(0)

target = data[["food", "service", "price", "ambiance"]]

def train_pseudolabel_model(feature_matrix, target, target_string, iterations=0,
    step=3000, num_to_add=100):
    """Train the classifier with iterations for pseudolabeling

    Args:
        feature_matrix - features
        target - target class data
        target_string - column name of intended target
        iterations - number of pseudolabeling iterations to run
        step - number of samples for each pseudolabel iteration
        num_to_add - number of most confident samples to add if threshold
                     is adaptive

    Returns:
        final classifier

    """
    X_tmp, y_tmp, X_test, y_test = iterative_train_test_split(
        np.array(feature_matrix),
        np.array(target[[target_string]]),
        test_size=0.2,
        random_state=10191994)

    X_train, y_train, X_validation, y_validation = iterative_train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.25, # 0.25 * 0.8 = 0.2
        random_state=10191994)

    logging.info("Train test split")

    score, conf_mat, roc, clf = train_model(X_train, y_train,
        X_validation, y_validation, target_string)

    with open("../../4_models/rf_{}_{}iterations_athreshold_{}_{}initial.pickle".format(
        target_string, iterations, step, num_to_add), "wb") as f:
        pickle.dump(clf, f)

    logging.info("Initial model trained")
    logging.info("Step size: {}; Samples selected: {}; Iterations: {}".format(
        step, num_to_add, iterations))
    logging.info("Size of train set: {}".format(X_train.shape[0]))
    logging.info("Number of 0: {}".format(
        y_train.shape[0] - y_train.sum()))
    logging.info("Number of 1: {}".format(y_train.sum()))
    logging.info("Initial score: {}".format(auc(roc[0], roc[1])))
    logging.info("Initial conf_matrix: {}".format(conf_mat))

    if iterations:
        add_counter = 0
        add_size = copy.deepcopy(num_to_add)

        for i in range(iterations):
            X_train_new, y_train_new = pseudolabel_data(
                X_train, y_train, target_string,
                clf, threshold="adaptive", random_seed=10191994+i,
                num_samples=step, num_to_add=num_to_add)
            score_new, conf_mat_new, roc_new, clf_new = train_model(
                X_train_new, y_train_new,
                X_validation, y_validation, target_string)

            if auc(roc_new[0], roc_new[1]) >= auc(roc[0], roc[1]): # try adding = for now
                add_counter = 0
                roc = auc(roc_new[0], roc_new[1])
                # score = score_new
                conf_mat = conf_mat_new
                clf = clf_new
                X_train = X_train_new
                y_train = y_train_new
                logging.info("Pseudolabel {} finished".format(i+1))
                logging.info("Size of train set: {}".format(X_train.shape[0]))
                logging.info("Number of 0: {}".format(
                    y_train.shape[0] - y_train.sum()))
                logging.info("Number of 1: {}".format(y_train.sum()))
                logging.info("Updated score: {}".format(score))
                logging.info("Updated conf_matrix: {}\n\n".format(conf_mat))
            else:
                add_counter += 1
                logging.info("Pseudolabel {} finished".format(i+1))
                logging.info("Not improved: {}\n\n".format(score_new))
                # if add_counter % 20 == 0:
                #     # every 20 iterations, increase num added
                #     num_to_add += add_size
                #     logging.info("Number to add increased to {}".format(
                #         num_to_add))

    else:
        return clf

    return clf


# =========================================================================== #
# HELPER FUNCTIONS
# =========================================================================== #

def train_model(feature_matrix, target, validation_x, validation_y, target_string):
    """Train a classifier on review data to get topic classes

    Args:
        feature_matrix - features
        target - target class data
        validation_x - validation features
        validation_y - validation target
        target_string - column name of intended target

    Returns:
        accuracy scores, conf matrices, clf
    """
    clf = RandomForestClassifier(n_estimators=1000,
                                 max_depth=3,
                                 min_samples_leaf=10,
                                 max_features=0.6,
                                 random_state=10191994,
                                 n_jobs=-1)

    try:
        score = clf.fit(feature_matrix[:, 1:], # because first column is raw text
                        target).score(
                        validation_x[:, 1:],
                        validation_y)

        conf_mat = confusion_matrix(validation_y,
            clf.predict(validation_x[:, 1:]))

        roc = roc_curve(validation_y,
                        clf.predict_proba(validation_x[:, 1:])[:,1])

        clf.fit(feature_matrix[:, 1:], target)
    except: # np array versus dataframe
        score = clf.fit(feature_matrix.iloc[:, 1:], # because first column is raw text
                        target).score(
                        validation_x[:, 1:],
                        validation_y)

        conf_mat = confusion_matrix(validation_y,
            clf.predict(validation_x[:, 1:]))

        roc = roc_curve(validation_y,
                        clf.predict_proba(validation_x[:, 1:])[:,1])

        clf.fit(feature_matrix.iloc[:, 1:], target)

    return score, conf_mat, roc, clf


def pseudolabel_data(feature_matrix_o, target_o, target_string_o, clf,
    random_seed=10191994,
    threshold=0.9, num_samples=1000, num_to_add=100):
    """Adds in pseudolabeled data to training data if predicted
    above the threshold

    Args:
        feature_matrix_o - original features
        target_o - original target class data
        target_string_o - original column name of intended target
        clf - classifier object to use for labeling data
        random_seed - random seed for review selection
        threshold - threshold of prediction probability
        num_samples - number of samples to mine
        num_to_add - number of samples to add if threshold is adaptive

    Returns:
        new_feature_matrix - appended feature matrix
        new_target - appended target
    """
    tmp_df = select_random_reviews(random_seed, num_samples)
    # ids = tmp_df["review_id"].tolist()
    # for tmp_id in ids:
    #     handlabeled_set.add(tmp_id)
    tmp_df = tmp_df[tmp_df["review_id"].apply(
        lambda x: x not in handlabeled_set)].reset_index(
        ).drop("index", axis=1)

    # I think without the drop here, when producing the feature
    # matrix, because it takes into account index when doing
    # parallel processing, it will create the full number of 
    # indices and replace missing ones with NA rows... or something...
    
    feature_matrix_n = produce_feature_matrix(tmp_df[["text"]])
    feature_matrix_n = feature_matrix_n.fillna(0)
    new_X_train_func = feature_matrix_n.iloc[:, 1:]
    pred_proba = clf.predict_proba(new_X_train_func)
    pred = clf.predict(new_X_train_func)

    # create new dataset
    confident_list_x = []
    confident_list_y = []

    # include code here to balance out set
    num_0 = target_o.shape[0] - target_o.sum()
    num_1 = target_o.sum()

    if threshold == "adaptive":
        # ranked_indices = np.argsort(np.apply_along_axis(min, 1, pred_proba))
        # the logic behind this is that if you choose the most confident ones
        # it will essentially reinforce the features and separation that exists
        # but if you look at the ones that are hard to classify,
        # then perhaps the model can "learn" from the boundaries
        # and if it performs better on the validation, then it can learn
        # new features that would better predict the outcome
        ranked_indices = np.argsort(np.apply_along_axis(lambda x: abs(x[0]-x[1]),
            1, pred_proba))
        ranked_pred_proba = pred_proba[ranked_indices]
        counter = 0

        for index, row in list(
            zip(ranked_indices, ranked_pred_proba)):
            if counter == 0:
                logging.info("First row added: {}".format(row))

            num_0_plus = len([x for x in confident_list_y if x == 0])
            num_1_plus = len([x for x in confident_list_y if x == 1])

            # can make this an or
            if (int(num_0) + num_0_plus) <= (int(num_1) + num_1_plus) and (
                row[0] > row[1]):

                tmp = pd.DataFrame(feature_matrix_n.iloc[index]).transpose()
                confident_list_x.append(tmp)
                confident_list_y.append(pred[index])
                handlabeled_set.add(tmp_df["review_id"].iloc[index])
                counter += 1

            if (int(num_0) + num_0_plus) >= (int(num_1) + num_1_plus) and (
                row[1] > row[0]):

                tmp = pd.DataFrame(feature_matrix_n.iloc[index]).transpose()
                confident_list_x.append(tmp)
                confident_list_y.append(pred[index])
                handlabeled_set.add(tmp_df["review_id"].iloc[index])
                counter += 1

            if counter >= num_to_add:
                logging.info("Last row added: {}".format(row))
                break

    else:
        for index, row in enumerate(pred_proba):
            num_0_plus = len([x for x in confident_list_y if x == 0])
            num_1_plus = len([x for x in confident_list_y if x == 1])

            if (int(num_0) + num_0_plus) <= (int(num_1) + num_1_plus) and (
                row[0] >= threshold):

                tmp = pd.DataFrame(feature_matrix_n.iloc[index]).transpose()
                confident_list_x.append(tmp)
                confident_list_y.append(pred[index]) 
                handlabeled_set.add(tmp_df["review_id"].iloc[index])

            if (int(num_0) + num_0_plus) >= (int(num_1) + num_1_plus) and (
                row[1] >= threshold):

                tmp = pd.DataFrame(feature_matrix_n.iloc[index]).transpose()
                confident_list_x.append(tmp)
                confident_list_y.append(pred[index]) 
                handlabeled_set.add(tmp_df["review_id"].iloc[index])
    

    try:
        new_x = pd.concat(confident_list_x, axis=0)
        new_x.columns = feature_matrix.columns
    except:
        new_x = pd.DataFrame()
    

    new_y = pd.DataFrame(confident_list_y)

    try: # they get transformed to numpy arrays after splitting
        feature_matrix_o_df = pd.DataFrame(feature_matrix_o)
        feature_matrix_o_df.columns = feature_matrix.columns

        new_X_train = pd.concat([feature_matrix_o_df,
                              new_x], axis=0)
        new_y_train = pd.concat([pd.DataFrame(target_o),
                          new_y])
    except:
        new_X_train = pd.concat([feature_matrix_o,
                              new_x], axis=0)
        new_y_train = pd.concat([target_o,
                          new_y])        


    return new_X_train, new_y_train


def select_influential_reviews(random_seed, num_reviews):
    """Select a random review from the influential reviews json

    Args:
        random_seed - int random seed for random library
        num_reviews - int number of reviews to retrieve

    Returns:
        review_df - dataframe of randomly selected reviews from influential
    """
    random.seed(random_seed)

    with open(os.path.join("../../0_data/2_processed_data",
        "business_elite_subset_reviews.json"), "r") as f:
        data = ujson.load(f)

        random_indexes = random.sample(range(0, len(data)), num_reviews)

        review_dict = {}

        counter = 0

        for random_index in random_indexes:
            if data[random_index]["review_id"] not in handlabeled_set:
                review_dict[counter] = data[random_index]
                counter += 1

        out_df = pd.DataFrame.from_dict(review_dict, orient="index")
        return out_df


def select_random_reviews(random_seed, num_reviews):
    """Select a random number of reviews from the whole set
    copied over from select_random_reviews.py
    ~ 45s for 10000

    Args:
        random_seed - int random seed for random library
        num_reviews - int number of reviews to retrieve

    Returns:
        review_df - dataframe of randomly selected reviews
    """
    random.seed(random_seed)

    random_indexes = random.sample(range(0, max_review), num_reviews)

    review_dict = {}

    counter = 0

    for random_index in random_indexes:
        # select the right file to open
        tmp_row = review_count_data[((review_count_data["cum_sum"] - 
            review_count_data["review_count"]) <=
            random_index + 1) & (random_index + 1 <=
            review_count_data["cum_sum"])]
        business_id = tmp_row[["business_id"]].iloc[0]["business_id"]
        file_index = (random_index
                      - (tmp_row["cum_sum"] - tmp_row["review_count"]) - 1)
        # open the file and retrieve the review
        with open(os.path.join(static_path,
            "reviews_by_business_id",
            "{}.json".format(business_id)), "r") as f:
            tmp_data = ujson.load(f)
            review_dict[counter] = tmp_data[file_index.iloc[0]]
            counter += 1


    out_df = pd.DataFrame.from_dict(review_dict, orient="index")
    return out_df


# =========================================================================== #
# RUN AND SAVE MODELS
# =========================================================================== #

# clf_ambiance = train_pseudolabel_model(feature_matrix, target, "ambiance",
#     iterations=500, step=500, num_to_add=5)
# with open("../../4_models/rf_ambiance_500iterations_athreshold_500_5.pickle", "wb") as f:
#     pickle.dump(clf_ambiance, f)


clf_price = train_pseudolabel_model(feature_matrix, target, "price",
    iterations=1000, step=100, num_to_add=1)
with open("../../4_models/rf_price_1000iterations_athreshold_100_1.pickle", "wb") as f:
    pickle.dump(clf_price, f)


clf_service = train_pseudolabel_model(feature_matrix, target, "service",
    iterations=1000, step=100, num_to_add=1)
with open("../../4_models/rf_service_1000iterations_athreshold_100_1.pickle", "wb") as f:
    pickle.dump(clf_service, f)


clf_food = train_pseudolabel_model(feature_matrix, target, "food",
    iterations=1000, step=100, num_to_add=1)
with open("../../4_models/rf_food_1000iterations_athreshold_100_1.pickle", "wb") as f:
    pickle.dump(clf_food, f)