# =========================================================================== #
# Date: 06/05/19
# Author: Richard Lu
# Description:
#   review.json is too large (~6GB) to load directly using pd.read_json
#   break up the file into smaller files for reading
# Runtime: ~20s per 1M rows
# =========================================================================== #

import os
import sys
import time
import ujson

from collections import defaultdict

# =========================================================================== #
# READ AND WRITE DATA
# =========================================================================== #

input_data_folder = "../0_data/1_raw_data"
output_data_folder = "../0_data/2_processed_data/reviews_by_business_id/tmp"
filename = "review.json"

starting_index = int(sys.argv[1])
ending_index = int(sys.argv[2])
file_counter = sys.argv[3]

data_dict = defaultdict(list)

start = time.time()
counter = 0

with open(os.path.join(input_data_folder, filename), "r") as f:
    for line in f:
        if counter >= starting_index and counter < ending_index:
            try:
                print(counter)
                tmp_review = ujson.loads(line)
                tmp_business_id = tmp_review["business_id"]
                data_dict[tmp_business_id].append(tmp_review)
            except: # to catch when the data "runs" out
                continue
        counter += 1


end = time.time()

print(end - start)



# =========================================================================== #
# WRITE DATA OUT
# =========================================================================== #

for out_business_id, review_list in data_dict.items():
    with open(os.path.join(output_data_folder,
                           "{}_{}.json".format(
                           out_business_id, file_counter)), "w") as f:
        ujson.dump(review_list, f)
