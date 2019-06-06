# =========================================================================== #
# Date: 06/05/19
# Author: Richard Lu
# Description:
#   starting to build out pipeline
#   this script creates random subsets of the reviews for feeding into
#   unsupervised learning scripts
# Runtime: 
# =========================================================================== #

import sys
import time

exec(open("../0_accessing_the_data/select_random_reviews.py").read())

# =========================================================================== #
# READ IN 
# =========================================================================== #

random_seed = int(sys.argv[1])
num_reviews = int(sys.argv[2])

start = time.time()
random_review_df = select_reviews(random_seed, num_reviews)
end = time.time()

print(end - start)

random_review_df[["text"]].to_csv(
    "../../0_data/3_robustness_check_data_for_unsupervised_models/{}_{}_completely_random.csv"
    .format(random_seed, num_reviews))