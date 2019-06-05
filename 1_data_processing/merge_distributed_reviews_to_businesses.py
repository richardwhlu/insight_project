# =========================================================================== #
# Date: 06/05/19
# Author: Richard Lu
# Description:
#   merge reviews split by business_id across different files
#   to a single business_id file with reviews
# Runtime: ~165s
# =========================================================================== #

import ast
import glob
import os
import time
import ujson

from collections import defaultdict

# =========================================================================== #
# HELPER FUNCTIONS
# =========================================================================== #

def merge_list_of_files_into_single_file(filetuple):
    """Takes a tuple of (business_id, filenames),
    merges the files into a single file, and writes the file out

    Args:
        filetuple - tuple of (business_id, filenames)

    Returns:
        None, but writes file
    """
    lines = []
    business_id = filetuple[0]
    filelist = filetuple[1]

    for filename in filelist:
        with open(filename, "r") as f:
            reviews = ujson.load(f)
            lines.extend(reviews)


    with open(os.path.join(output_data_folder,
        "{}.json".format(business_id)), "w") as f:
        ujson.dump(lines, f)


def retrieve_business_id_from_full_file_path(filepath):
    """Retrieve business id portion of the full file path and filename

    Args:
        filepath - string for the full path of the file

    Returns:
        business_id - only the business_id portion
    """
    tmp_filename = filepath.split("/")[-1]
    business_id = tmp_filename[:-7]

    return business_id


# =========================================================================== #
# READ AND WRITE DATA
# =========================================================================== #

start = time.time()

input_data_folder = "../0_data/2_processed_data/reviews_by_business_id/tmp"
output_data_folder = "../0_data/2_processed_data/reviews_by_business_id"

filenames = glob.glob(os.path.join(input_data_folder, "*"))

file_dict = defaultdict(list)

for filename in filenames:
    business_id = retrieve_business_id_from_full_file_path(filename)
    file_dict[business_id].append(filename)


for tmp_tuple in file_dict.items():
    merge_list_of_files_into_single_file(tmp_tuple)


end = time.time()
print(end - start)
