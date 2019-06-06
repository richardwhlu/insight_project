# ========================================================================== #
# Date: 06/06/19
# Author: Richard Lu
# Description:
#   user.json is too large (~2GB) to load directly using pd.read_json
#   break up the file into smaller files based on review history
# Runtime: ~ 27s
# ========================================================================== #

import os
import sys
import time
import ujson

# =========================================================================== #
# READ DATA
# =========================================================================== #

input_data_folder = "../../0_data/1_raw_data"
output_data_folder = "../../0_data/2_processed_data"
filename = "user.json"

starting_index = int(sys.argv[1])
ending_index = int(sys.argv[2])
# file_counter = sys.argv[3]

data_list = []

start = time.time()
counter = 0
counter2 = 0

# ~ 1.6M users
with open(os.path.join(input_data_folder, filename), "r") as f:
    for line in f:
        if counter >= starting_index and counter < ending_index:
            try:
                print(counter)
                tmp_user = ujson.loads(line)
                elite_status = tmp_user["elite"]
                review_count = tmp_user["review_count"]
                if elite_status: # get cutoffs here
                    data_list.append(tmp_user)
                    counter2 += 1
                    print(counter2)
            except: # to catch when the data "runs" out
                continue
        counter += 1


end = time.time()

print(end - start)

# =========================================================================== #
# WRITE DATA OUT
# =========================================================================== #

with open(os.path.join(output_data_folder,
    "elite_users.json"), "w") as f:
    ujson.dump(data_list, f)