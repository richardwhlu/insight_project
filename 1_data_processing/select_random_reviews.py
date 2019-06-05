# =========================================================================== #
# Date: 06/05/19
# Author: Richard Lu
# Description:
#   quick script to select a random number of reviews
#   meant to be imported  
# Runtime: 
#   ~44.6s for 10000
#   ~457.8 for 100000
# =========================================================================== #

import glob
import os
import pandas as pd
import random
import ujson

# =========================================================================== #
# READ IN CSV
# =========================================================================== #

static_path = "/home/richard/Documents/work/insight/insight_project/0_data/2_processed_data"

review_filename = "review_count_by_business_id.csv"

review_count_data = pd.read_csv(os.path.join(static_path, review_filename)
    ).drop("Unnamed: 0", axis=1)

review_count_data["cum_sum"] = review_count_data["review_count"].cumsum()

max_review = review_count_data["review_count"].sum()

# =========================================================================== #
# HELPER FUNCTIONS
# =========================================================================== #

def select_reviews(random_seed, num_reviews):
    """Select a random number of reviews from the whole set

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
# MAIN
# =========================================================================== #

if __name__ == "__main__":
    pass