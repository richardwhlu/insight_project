# =========================================================================== #
# Date: 06/13/19
# Author: Richard Lu
# Description:
#   scrape site for input data
# Runtime: 
# =========================================================================== #

import pandas as pd
import queue
import requests
import threading
import time
import urllib.request

from bs4 import BeautifulSoup as bs
from bs4 import SoupStrainer

def read_url(url, queue):
    """Function to read url

    Args:
        url - url to read
        queue - queue object

    Returns:
        None
    """
    html = requests.get(url).text
    queue.put(html)


def fetch_parallel(urls_to_load):
    """Fetch data from urls in multi-threading fashion

    Args:
        urls_to_load - list of urls to scrape from

    Returns:
        list of html data
    """
    result = queue.Queue()
    threads = [threading.Thread(target=read_url,
        args=(url, result)) for url in urls_to_load]
    for t in threads:
        t.start()

    for t in threads:
        t.join()

    return list(result.queue)


def extract_from_page(soup):
    """Get relevant details from page

    Args:
        soup - bs object of page

    Returns:
        list of ratings
        list of texts
    """
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

    return int_review_ratings, review_texts


def scrape_url(url, num_reviews):
    """Scrape the webpage input for yelp comments

    Args:
        url - base url of yelp page
        num_reviews - X most recent reviews to scrape

    Returns:
        df of comments and ratings
    """
    urls_to_load = []
    full_ratings = []
    full_review_texts = []

    for i in range(0, int(num_reviews), 20):
        tmp_url = url + "?start={}".format(i)
        urls_to_load.append(tmp_url) 

    response_list = fetch_parallel(urls_to_load)

    for html_page in response_list:

        strainer = SoupStrainer("ul")

        soup = bs(html_page, "lxml", parse_only=strainer)

        int_review_ratings, review_texts = extract_from_page(soup)

        full_ratings.extend(int_review_ratings)
        full_review_texts.extend(review_texts)

    review_texts = pd.DataFrame(list(
        zip(full_review_texts, full_ratings)), columns=["text", "rating"])

    return review_texts