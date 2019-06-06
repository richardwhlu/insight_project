# =========================================================================== #
# Date: 06/06/19
# Author: Richard Lu
# Description:
#   try to clean the reviews to get more informative unsupervised topics
# Runtime: ~2.6s after data is subset
# =========================================================================== #

import glob
import os
import pandas as pd
import random
import sys
import time
import ujson

# =========================================================================== #
# READ IN DATA
# =========================================================================== #

input_data_folder1 = "../../0_data/1_raw_data"
input_data_folder2 = "../../0_data/2_processed_data"

random_seed = int(sys.argv[1])
num_reviews = int(sys.argv[2])
create_influential_reviews = int(sys.argv[3])

business = pd.read_json(os.path.join(input_data_folder1,
    "business.json"), lines=True)

user = pd.read_json(os.path.join(input_data_folder2,
    "elite_users.json"))

# =========================================================================== #
# GET SUBSETS
# =========================================================================== #

condition1 = business["is_open"]==1
# 75% cutoff for review count is 25
condition2 = business["review_count"]>=1000
top_reviewed_businesses = business[condition1 & condition2]
top_reviewed_business_ids = set(
    top_reviewed_businesses["business_id"].tolist())

# perhaps even subset on business types?

# subset on users who have been yelp elite at some point in time
elite_users = set(user["user_id"].tolist())

# =========================================================================== #
# SELECT REVIEWS AND WRITE TO FILE
# =========================================================================== #

# a lot of code reproduced here; could  be cleaned up
# old code was too slow, produce new dataset with subset of reviews and 
# select randomly from there

if create_influential_reviews:
    review_files = glob.glob(os.path.join(input_data_folder2,
        "reviews_by_business_id", "*.json"))

    all_reviews = []

    counter = 0

    counter2 = 0

    for filename in review_files:
        with open(filename, "r") as f:
            tmp_review_data = ujson.load(f)
            for tmp_review in tmp_review_data:
                counter2 += 1
                if counter2 % 100000 == 0:
                    print("parsed: {}".format(counter2))
                if (tmp_review["user_id"] in elite_users
                    and tmp_review["business_id"] in top_reviewed_business_ids):
                    all_reviews.append(tmp_review)
                    counter += 1
                    if counter % 1000 == 0:
                        print("collected: {}".format(counter))


    with open(os.path.join(input_data_folder2,
        "business_elite_subset_reviews.json"), "w") as f:
        ujson.dump(all_reviews, f)


# =========================================================================== #
# SELECT INFLUENTIAL REVIEWS
# =========================================================================== #

def select_influential_reviews(random_seed, num_reviews):
    """Select a random review from the influential reviews json

    Args:
        random_seed - int random seed for random library
        num_reviews - int number of reviews to retrieve

    Returns:
        review_df - dataframe of randomly selected reviews from influential
    """
    random.seed(random_seed)

    with open(os.path.join(input_data_folder2,
        "business_elite_subset_reviews.json"), "r") as f:
        data = ujson.load(f)

        random_indexes = random.sample(range(0, len(data)), num_reviews)

        review_dict = {}

        counter = 0

        for random_index in random_indexes:
            review_dict[counter] = data[random_index]
            counter += 1

        out_df = pd.DataFrame.from_dict(review_dict, orient="index")
        return out_df


start = time.time()
influential_review_df = select_influential_reviews(random_seed, num_reviews)
end = time.time()

print(end - start)

# # =========================================================================== #
# # WRITE OUT
# # =========================================================================== #

influential_review_df[["text"]].to_csv(
    "../../0_data/3_robustness_check_data_for_unsupervised_models/{}_{}_influential.csv"
    .format(random_seed, num_reviews))