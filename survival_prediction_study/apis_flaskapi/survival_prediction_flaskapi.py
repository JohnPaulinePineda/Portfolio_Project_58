##################################
# Loading Python libraries
##################################
from flask import Flask, request, jsonify
import os
import joblib
import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from lifelines import KaplanMeierFitter
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
matplotlib.use('Agg')
from flasgger import Swagger

##################################
# Defining file paths
##################################
MODELS_PATH = r"models"
PARAMETERS_PATH = r"parameters"

##################################
# Loading the model
##################################
try:
    final_survival_prediction_model = joblib.load(os.path.join("..", MODELS_PATH, "coxph_best_model.pkl"))
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

##################################
# Loading the median parameter
##################################
try:
    numeric_feature_median = joblib.load(os.path.join("..", PARAMETERS_PATH, "numeric_feature_median_list.pkl"))
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

##################################
# Loading the threshold parameter
##################################
try:
    final_survival_prediction_model_risk_group_threshold = joblib.load(os.path.join("..", PARAMETERS_PATH, "coxph_best_model_risk_group_threshold.pkl"))
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

##################################
# Initializing the Flask app
##################################
app = Flask(__name__)
Swagger(app)

##################################
# Defining a GET endpoint for
# validating API service connection
##################################
@app.route("/", methods=["GET"])
def root():
    """
    Root endpoint to validate API service connection.
    ---
    responses:
      200:
        description: A welcome message
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                  example: "Welcome to the Survival Prediction API!"
    """
    return jsonify({"message": "Welcome to the Survival Prediction API!"})

##################################
# Defining a POST endpoint for
# generating the heart failure survival profile,
# estimating the heart failure survival probabilities,
# and predicting the risk category
# of an individual test case
##################################
@app.route("/compute-individual-coxph-survival-probability-class/", methods=["POST"])
def compute_individual_coxph_survival_probability_class():
    """
    Compute survival probabilities and risk category for an individual test case.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            features_individual:
              type: array
              items:
                type: number
              example: [65, 0, 38, 1, 1.2, 137]
    responses:
      200:
        description: Survival probabilities and risk category for an individual test case
        content:
          application/json:
            schema:
              type: object
              properties:
                survival_function:
                  type: array
                  items:
                    type: number
                  example: [0.95, 0.90, 0.85, 0.80, 0.75]
                survival_time:
                  type: array
                  items:
                    type: number
                  example: [50, 100, 150, 200, 250]
                survival_probabilities:
                  type: array
                  items:
                    type: number
                  example: [95.0, 90.0, 85.0, 80.0, 75.0]
                risk_category:
                  type: string
                  example: "Low-Risk"
      400:
        description: Bad request (e.g., missing 'features_individual' in request)
      500:
        description: Internal server error
    """
    try:
        # Getting JSON data from the request
        data = request.json
        if "features_individual" not in data:
            return jsonify({"error": "Missing 'features_individual' in request"}), 400

        # Defining the column names
        column_names = ["AGE", "ANAEMIA", "EJECTION_FRACTION", "HIGH_BLOOD_PRESSURE", "SERUM_CREATININE", "SERUM_SODIUM"]

        # Converting the data input to a pandas DataFrame with appropriate column names
        X_test_sample = pd.DataFrame([data["features_individual"]], columns=column_names)

        # Obtaining the survival function for an individual test case
        survival_function = final_survival_prediction_model.predict_survival_function(X_test_sample)

        # Predicting the risk category an individual test case
        risk_category = (
            "High-Risk"
            if (final_survival_prediction_model.predict(X_test_sample)[0] > final_survival_prediction_model_risk_group_threshold)
            else "Low-Risk"
        )

        # Defining survival times
        survival_time = np.array([50, 100, 150, 200, 250])

        # Predicting survival probabilities an individual test case
        survival_probability = np.interp(
            survival_time, survival_function[0].x, survival_function[0].y
        )
        survival_probabilities = survival_probability * 100

        # Returning the endpoint response
        return jsonify({
            "survival_function": survival_function[0].y.tolist(),
            "survival_time": survival_time.tolist(),
            "survival_probabilities": survival_probabilities.tolist(),
            "risk_category": risk_category,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

##################################
# Defining a POST endpoint for
# generating the heart failure survival profile and
# estimating the heart failure survival probabilities
# of a list of train cases
##################################
@app.route("/compute-list-coxph-survival-profile/", methods=["POST"])
def compute_list_coxph_survival_profile():
    """
    Compute survival profiles for a list of test cases.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            features_list:
              type: array
              items:
                type: array
                items:
                  type: number
              example: [[65, 0, 38, 1, 1.2, 137], [70, 1, 40, 0, 1.5, 135]]
    responses:
      200:
        description: Survival profiles for a list of test cases
        content:
          application/json:
            schema:
              type: object
              properties:
                survival_profiles:
                  type: array
                  items:
                    type: array
                    items:
                      type: number
                  example: [[0.95, 0.90, 0.85], [0.90, 0.85, 0.80]]
      400:
        description: Bad request (e.g., missing 'features_list' in request)
      500:
        description: Internal server error
    """
    try:
        # Getting JSON data from the request
        data = request.json
        if "features_list" not in data:
            return jsonify({"error": "Missing 'features_list' in request"}), 400

        # Defining the column names
        column_names = ["AGE", "ANAEMIA", "EJECTION_FRACTION", "HIGH_BLOOD_PRESSURE", "SERUM_CREATININE", "SERUM_SODIUM"]

        # Converting the data input to a pandas DataFrame with appropriate column names
        X_train_list = pd.DataFrame(data["features_list"], columns=column_names)

        # Obtaining the survival function for a batch of cases
        survival_function = final_survival_prediction_model.predict_survival_function(X_train_list)

        # Extracting survival profiles from the survival function output for a batch of cases
        survival_profiles = [sf.y.tolist() for sf in survival_function]

        # Returning the endpoint response
        return jsonify({"survival_profiles": survival_profiles})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

##################################
# Defining a POST endpoint for
# creating dichotomous bins for the numeric features
# of a list of train cases
##################################
@app.route("/bin-numeric-model-feature/", methods=["POST"])
def bin_numeric_model_feature():
    """
    Dichotomize numeric features based on the median.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            X_original_list:
              type: array
              items:
                type: object
              example: [{"AGE": 65, "EJECTION_FRACTION": 38}, {"AGE": 70, "EJECTION_FRACTION": 40}]
            numeric_feature:
              type: string
              example: "AGE"
    responses:
      200:
        description: Dichotomized numeric features
        content:
          application/json:
            schema:
              type: array
              items:
                type: object
              example: [{"AGE": "Low", "EJECTION_FRACTION": 38}, {"AGE": "High", "EJECTION_FRACTION": 40}]
      400:
        description: Bad request (e.g., missing 'X_original_list' or 'numeric_feature' in request)
      500:
        description: Internal server error
    """
    try:
        # Getting JSON data from the request
        data = request.json
        if "X_original_list" not in data or "numeric_feature" not in data:
            return jsonify({"error": "Missing 'X_original_list' or 'numeric_feature' in request"}), 400

        # Converting the data input to a pandas DataFrame with appropriate column names
        X_original_list = pd.DataFrame(data["X_original_list"])

        # Computing the median
        median = numeric_feature_median.loc[data["numeric_feature"]]

        # Dichotomizing the data input to categories based on the median
        X_original_list[data["numeric_feature"]] = np.where(
            X_original_list[data["numeric_feature"]] <= median, "Low", "High"
        )

        # Returning the endpoint response
        return jsonify(X_original_list.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

##################################
# Defining a POST endpoint for
# plotting the estimated survival profiles
# using Kaplan-Meier Plots
##################################
@app.route("/plot-kaplan-meier/", methods=["POST"])
def plot_kaplan_meier():
    """
    Plot Kaplan-Meier survival curves.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            df:
              type: array
              items:
                type: object
              example: [{"TIME": 100, "DEATH_EVENT": 1, "AGE": "Low"}, {"TIME": 200, "DEATH_EVENT": 0, "AGE": "High"}]
            cat_var:
              type: string
              example: "AGE"
            new_case_value:
              type: string
              example: "Low"
    responses:
      200:
        description: Base64-encoded Kaplan-Meier plot
        content:
          application/json:
            schema:
              type: object
              properties:
                plot:
                  type: string
                  example: "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      400:
        description: Bad request (e.g., missing 'df' or 'cat_var' in request)
      500:
        description: Internal server error
    """
    try:
        # Getting JSON data from the request
        data = request.json
        if "df" not in data or "cat_var" not in data:
            return jsonify({"error": "Missing 'df' or 'cat_var' in request"}), 400

        # Obtaining plot components
        df = pd.DataFrame(data["df"])
        cat_var = data["cat_var"]
        new_case_value = data.get("new_case_value")

        # Initializing a Kaplan-Meier plot object
        kmf = KaplanMeierFitter()

        # Creating the Kaplan-Meier plot
        fig, ax = plt.subplots(figsize=(8, 6))
        if cat_var in ['AGE', 'EJECTION_FRACTION', 'SERUM_CREATININE', 'SERUM_SODIUM']:
            categories = ['Low', 'High']
            colors = {'Low': 'blue', 'High': 'red'}
        else:
            categories = ['Absent', 'Present']
            colors = {'Absent': 'blue', 'Present': 'red'}

        for value in categories:
            mask = df[cat_var] == value
            kmf.fit(df['TIME'][mask], event_observed=df['DEATH_EVENT'][mask], label=f'{cat_var}={value} (Baseline Distribution)')
            kmf.plot_survival_function(ax=ax, ci_show=False, color=colors[value], linestyle='-', linewidth=6.0, alpha=0.30)

        if new_case_value is not None:
            mask_new_case = df[cat_var] == new_case_value
            kmf.fit(df['TIME'][mask_new_case], event_observed=df['DEATH_EVENT'][mask_new_case], label=f'{cat_var}={new_case_value} (Test Case)')
            kmf.plot_survival_function(ax=ax, ci_show=False, color='black', linestyle=':', linewidth=3.0)

        ax.set_title(f'DEATH_EVENT Survival Probabilities by {cat_var} Categories')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('TIME')
        ax.set_ylabel('DEATH_EVENT Survival Probability')
        ax.legend(loc='lower left')

        # Saving the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        base64_image = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        # Closing the plot to release resources
        plt.close(fig)

        # Returning the endpoint response
        return jsonify({"plot": base64_image})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

##################################
# Running the Flask app
##################################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
