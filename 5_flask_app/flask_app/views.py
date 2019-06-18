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

        data = scrape_url(tmp_url)

        feature_matrix = run_models.produce_feature_matrix(data[["text"]])

        # food_model = pickle.load(
        #     open("flask_app/static/models/rf_food_50iterations_0.8threshold.pickle", "rb"))

        # food_pred = food_model.predict(feature_matrix.iloc[:, 1:].fillna(0))

        # service_model = pickle.load(
        #     open("flask_app/static/models/rf_service_50iterations_0.8threshold.pickle", "rb"))

        # service_pred = service_model.predict(feature_matrix.iloc[:, 1:].fillna(0))

        # price_model = pickle.load(
        #     open("flask_app/static/models/rf_price_50iterations_0.8threshold.pickle", "rb"))

        # price_pred = price_model.predict(feature_matrix.iloc[:, 1:].fillna(0))

        ambiance_model = pickle.load(
            open("flask_app/static/models/rf_ambiance_700iterations_athreshold_100_1initial.pickle", "rb"))

        ambiance_pred = ambiance_model.predict(feature_matrix.iloc[:, 1:].fillna(0))

        # data["food"] = food_pred
        # data["service"] = service_pred
        # data["price"] = price_pred
        data["ambiance"] = ambiance_pred

        overall_rating = data["rating"].mean()
        # food_rating = (data["rating"] * data["food"])[
        #     (data["rating"] * data["food"]).apply(
        #     lambda x: x > 0)].mean()
        # service_rating = (data["rating"] * data["service"])[
        #     (data["rating"] * data["service"]).apply(
        #     lambda x: x > 0)].mean()
        # price_rating = (data["rating"] * data["price"])[
        #     (data["rating"] * data["price"]).apply(
        #     lambda x: x > 0)].mean()
        ambiance_rating = (data["rating"] * data["ambiance"])[
            (data["rating"] * data["ambiance"]).apply(
            lambda x: x > 0)].mean()

        ind_rating_data = [overall_rating,
                           # food_rating,
                           # service_rating,
                           # price_rating,
                           ambiance_rating]

        # print(request.form["submit_button"])
        # if request.form["submit_button"] == "Food":
        #     print("food")
        #     data = data[data["food"].apply(lambda x: x == 1)]

        # if request.form["submit_button"] == "Service":
        #     data = data[data["service"].apply(lambda x: x == 1)]

        # if request.form["submit_button"] == "Price":
        #     data = data[data["price"].apply(lambda x: x == 1)]

        # if request.form["submit_button"] == "Ambiance":
        #     data = data[data["ambiance"].apply(lambda x: x == 1)]


    else:
        tmp_url = ""
        data = pd.DataFrame()
        ind_rating_data = []


    return render_template(
        "demo.html",
        title="Insight Demo 2 - MockUp",
        tmp_url=tmp_url,
        data=data,
        ind_rating_data=ind_rating_data)


