# =========================================================================== #
# Date: 06/13/19
# Author: Richard Lu
# Description:
#   MVP web app
# Runtime: 
# =========================================================================== #

from gensim.models import Word2Vec
import os
import pandas as pd
import pickle

from flask import render_template, request
from flask_app import app, run_models
from flask_app.scrape import scrape_url

# =========================================================================== #
# VIEWS
# =========================================================================== #

@app.route("/", methods=["POST", "GET"])
@app.route("/index", methods=["POST", "GET"])
def index():
    return render_template("index.html")


@app.route("/demo", methods=["POST", "GET"])
def demo():
    """MVP for Demo
    """
    if request.method == "POST":
        tmp_url = request.form["url"]
        num_reviews = request.form["num_reviews"]

        data = scrape_url(tmp_url, num_reviews)

        feature_matrix = run_models.produce_feature_matrix(data[["text"]])

        food_model = pickle.load(
            open("flask_app/static/models/rf_food_700iterations_athreshold_100_1initial.pickle", "rb"))

        food_pred = food_model.predict_proba(feature_matrix.iloc[:, 1:].fillna(0))[:, 1]

        service_model = pickle.load(
            open("flask_app/static/models/rf_service_700iterations_athreshold_100_1initial.pickle", "rb"))

        service_pred = service_model.predict_proba(feature_matrix.iloc[:, 1:].fillna(0))[:, 1]

        price_model = pickle.load(
            open("flask_app/static/models/rf_price_700iterations_athreshold_100_1initial.pickle", "rb"))

        price_pred = price_model.predict_proba(feature_matrix.iloc[:, 1:].fillna(0))[:, 1]

        ambiance_model = pickle.load(
            open("flask_app/static/models/rf_ambiance_700iterations_athreshold_100_1initial.pickle", "rb"))

        ambiance_pred = ambiance_model.predict_proba(feature_matrix.iloc[:, 1:].fillna(0))[:, 1]

        data["food"] = [round(x) for x in food_pred]
        data["service"] = [round(x) for x in service_pred]
        data["price"] = [round(x) for x in price_pred]
        data["ambiance"] = [round(x) for x in ambiance_pred]

        data["food_pred"] = [round(x, 2) for x in food_pred]
        data["service_pred"] = [round(x, 2) for x in service_pred]
        data["price_pred"] = [round(x, 2) for x in price_pred]
        data["ambiance_pred"] = [round(x, 2) for x in ambiance_pred]

        data = data.sort_values(["food_pred",
                                 "service_pred",
                                 "price_pred",
                                 "ambiance_pred"], ascending=False)

        overall_rating = data["rating"].mean()
        food_rating = (data["rating"] * data["food"])[
            (data["rating"] * data["food"]).apply(
            lambda x: x > 0)].mean()
        service_rating = (data["rating"] * data["service"])[
            (data["rating"] * data["service"]).apply(
            lambda x: x > 0)].mean()
        price_rating = (data["rating"] * data["price"])[
            (data["rating"] * data["price"]).apply(
            lambda x: x > 0)].mean()
        ambiance_rating = (data["rating"] * data["ambiance"])[
            (data["rating"] * data["ambiance"]).apply(
            lambda x: x > 0)].mean()

        ind_rating_data = [overall_rating,
                           food_rating,
                           service_rating,
                           price_rating,
                           ambiance_rating]

    else:
        tmp_url = ""
        num_reviews = 0
        data = pd.DataFrame()
        ind_rating_data = []


    return render_template(
        "demo.html",
        title="Insight Demo 3 - MockUp",
        tmp_url=tmp_url,
        num_reviews=num_reviews,
        data=data,
        ind_rating_data=ind_rating_data)


