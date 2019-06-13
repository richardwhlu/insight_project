# =========================================================================== #
# Date: 06/13/19
# Author: Richard Lu
# Description:
#   scrape site for input data
# Runtime: 
# =========================================================================== #

import pandas as pd
import requests

from bs4 import BeautifulSoup as bs
from bs4 import SoupStrainer


def scrape_url(url):
    """Scrape the webpage input for yelp comments

    Args:
        url - base url of yelp page

    Returns:
        df of comments and ratings
    """
    full_ratings = []
    full_review_texts = []
    counter = 0

    while True:
        tmp_url = url + "?start={}".format(counter)

        html = requests.get(tmp_url)

        strainer = SoupStrainer("ul")

        soup = bs(html.text, "lxml", parse_only=strainer)

        # if there are "previous ratings" they will be included
        try:
            review_space = soup.find_all("ul", {"class": "ylist-bordered"})[1]
        except:
            # I think this is wrong - only if ?osq=
            review_space = soup.find_all("ul", {"class": "ylist-bordered"})[0]

        review_content = review_space.find_all("div", {"class": "review-content"})

        review_ratings = [x.find("div", {"class": "i-stars"})
            for x in review_content]
        int_review_ratings = [int(float(x["title"].replace(" star rating", "")))
            for x in review_ratings]

        review_texts = [x.find("p").text for x in review_content]

        print(counter)

        if review_ratings:
            full_ratings.extend(int_review_ratings)
            full_review_texts.extend(review_texts)
            counter += 20
        else:
            break


    review_texts = pd.DataFrame(list(
        zip(full_review_texts, full_ratings)), columns=["text", "rating"])

    return review_texts