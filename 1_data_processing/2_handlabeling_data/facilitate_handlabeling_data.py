# =========================================================================== #
# Date: 06/07/19
# Author: Richard Lu
# Description:
#   create a script that automatically serves up reviews and allows for
#   easy handlabeling
# Runtime: 
# =========================================================================== #

import csv
import os
import pandas as pd
import sys
import ujson

# =========================================================================== #
# READ IN DATA
# =========================================================================== #

exec(open("../0_accessing_the_data/select_random_reviews.py").read())

data_folder0 = "../../0_data/1_raw_data"
data_folder = "../../0_data/4_handlabeled_data"

input_filename = "already_handlabeled_review_ids.json"

output_filename = "handlabeled_reviews.csv"

# need to make sure that the review is for a restaurant
business = pd.read_json(os.path.join(data_folder0,
    "business.json"), lines=True)

def contains_restaurants(text):
    try:
        return "restaurants" in text.lower()
    except:
        return False


condition = business["categories"].apply(contains_restaurants)

restaurants = business[condition]
restaurant_ids = set(restaurants["business_id"].tolist())

# random_seed = int(sys.argv[1])
random_seed = 10191994

# make sure to create the handlabeled_reviews.csv first with the
# appropriate headings
try:
    with open(os.path.join(data_folder,
        input_filename), "r") as f:
        already_labeled_review_ids = ujson.load(f)
        already_labeled_review_ids["labeled_ids"] = set(
            already_labeled_review_ids["labeled_ids"])

except:
    already_labeled_review_ids = {"labeled_ids": set()}



# =========================================================================== #
# SELECT RANDOM REVIEW TO SHOW
# =========================================================================== #

# since num unlabeled compared to num labeled is large, go with a slower
# randomization method (i.e. checking if it is already labeled then
# randomly selecting another)

random_reviews = select_reviews(random_seed, 2000)

counter = 0

for index, row in random_reviews.iterrows():
    if (row["review_id"] not in already_labeled_review_ids["labeled_ids"] and
        row["business_id"] in restaurant_ids):
        counter += 1
        print("Review #{}\n".format(counter))
        print(row["text"])
        print("\n\n")
        food = int(input("Food? "))
        service = int(input("Service? "))
        price = int(input("Price? "))
        ambiance = int(input("Ambiance? "))
        print("\n\n")
        with open(os.path.join(data_folder, output_filename), "a") as f:
            tmp_row = [row["review_id"], repr(row["text"])]
            if food:
                tmp_row.append(1)
            else:
                tmp_row.append(0)
            if service:
                tmp_row.append(1)
            else:
                tmp_row.append(0)
            if price:
                tmp_row.append(1)
            else:
                tmp_row.append(0)
            if ambiance:
                tmp_row.append(1)
            else:
                tmp_row.append(0)
            csvwriter = csv.writer(f)
            csvwriter.writerow(tmp_row)


        already_labeled_review_ids["labeled_ids"].add(row["review_id"])
        with open(os.path.join(data_folder, input_filename), "w") as f:
            ujson.dump(already_labeled_review_ids, f)


# =========================================================================== #
# REVIEWS LABELED
# =========================================================================== #

# 10191994  