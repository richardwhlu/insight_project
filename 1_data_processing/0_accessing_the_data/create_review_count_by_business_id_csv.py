# =========================================================================== #
# Date: 06/05/19
# Author: Richard Lu
# Description:
#   create a quick lookup dataset that contains number of reviews per
#   business_id (since individual data are too large to easily merge)
#   and for purposes of data validation with the business.json
# Runtime: ~53s
# =========================================================================== #

import glob
import os
import pandas as pd
import time
import ujson

from collections import defaultdict

# =========================================================================== #
# HELPER FUNCTION
# =========================================================================== #

def get_review_count_for_business(filename):
    """Opens the file and retrieves the number of reviews

    Args:
        filename - name of the input file [business_id].json

    Returns:
        review_count
    """
    with open(filename, "r") as f:
        data = ujson.load(f)
        review_count = len(data)
        return review_count


# =========================================================================== #
# READ DATA
# =========================================================================== #

start = time.time()

input_data_folder = "../../0_data/2_processed_data/reviews_by_business_id"
output_data_folder = "../../0_data/2_processed_data"

filenames = glob.glob(os.path.join(input_data_folder, "*.json"))

review_count_dict = defaultdict(dict)

counter = 0

for filename in filenames:
    tmp_review_count = get_review_count_for_business(filename)
    review_count_dict[counter]["review_count"] = tmp_review_count
    review_count_dict[counter]["business_id"] = filename.split("/")[-1][:-5]
    counter += 1

out_dataframe = pd.DataFrame.from_dict(review_count_dict, orient="index")
out_dataframe[["business_id", "review_count"]].to_csv(os.path.join(
    output_data_folder, "review_count_by_business_id.csv"))

end = time.time()

print(end - start)

