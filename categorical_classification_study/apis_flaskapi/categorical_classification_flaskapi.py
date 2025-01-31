##################################
# Loading Python libraries
##################################
from flask import Flask, request, jsonify
import os
import joblib
import numpy as np
import pandas as pd
from flasgger import Swagger

##################################
# Defining file paths
##################################
MODELS_PATH = "models"

##################################
# Loading the model
##################################
try:
    final_classification_model = joblib.load(os.path.join("..", MODELS_PATH, "stacked_balanced_class_best_model_upsampled.pkl"))
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
                  example: "Welcome to the Categorical Classification API!"
    """
    return jsonify({"message": "Welcome to the Categorical Classification API!"})

##################################
# Defining a POST endpoint for
# computing the risk index,
# estimating the lung cancer probability,
# and predicting the risk category
# of an individual test case
##################################
@app.route("/predict-individual-logit-probability-class", methods=["POST"])
def predict_individual_logit_probability_class():
    """
    Predict logit, probability, and risk class for an individual test case.
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
              example: [1.0, 2.0, 3.0, 4.0, 5.0]
    responses:
      200:
        description: Prediction results for an individual test case
        content:
          application/json:
            schema:
              type: object
              properties:
                logit:
                  type: number
                  example: 0.123
                probability:
                  type: number
                  example: 0.456
                risk_class:
                  type: string
                  example: "Low-Risk"
      400:
        description: Bad request (e.g., invalid input)
    """
    try:
        input_data = request.json
        # Converting the data input to a DataFrame with proper feature names
        X_test_sample = pd.DataFrame([input_data["features_individual"]], columns=final_classification_model.feature_names_in_)

        # Obtaining the estimated logit and probability values for an individual test case
        logit, probability, risk_class = compute_individual_logit_probability_class(X_test_sample)

        # Returning the endpoint response
        return jsonify({
            "logit": logit,
            "probability": probability,
            "risk_class": risk_class,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

##################################
# Defining a POST endpoint for
# computing the risk index,
# estimating the lung cancer probability,
# and predicting the risk category
# of a list of train cases
##################################
@app.route("/predict-list-logit-probability-class", methods=["POST"])
def predict_list_logit_probability_class():
    """
    Predict logit, probability, and risk class for a list of test cases.
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
              example: [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]
    responses:
      200:
        description: Prediction results for a list of test cases
        content:
          application/json:
            schema:
              type: object
              properties:
                logit:
                  type: array
                  items:
                    type: number
                  example: [0.123, 0.456]
                probability:
                  type: array
                  items:
                    type: number
                  example: [0.456, 0.789]
                logit_sorted:
                  type: array
                  items:
                    type: number
                  example: [0.123, 0.456]
                probability_sorted:
                  type: array
                  items:
                    type: number
                  example: [0.456, 0.789]
      400:
        description: Bad request (e.g., invalid input)
    """
    try:
        input_data = request.json
        # Converting the data input to a DataFrame with proper feature names
        X_train_list = pd.DataFrame(input_data["features_list"], columns=final_classification_model.feature_names_in_)

        # Obtaining the estimated logit and probability values for a batch of cases
        logit, probability, logit_sorted, probability_sorted = compute_list_logit_probability_class(X_train_list)

        # Returning the endpoint response
        return jsonify({
            "logit": logit.tolist(),
            "probability": probability.tolist(),
            "logit_sorted": logit_sorted.tolist(),
            "probability_sorted": probability_sorted.tolist(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

##################################
# Formulating a function to
# compute the risk index,
# estimate the lung cancer probability,
# and predict the risk category
# of an individual test case
##################################
def compute_individual_logit_probability_class(X_test_sample):
    """Compute logit, probability, and risk class for an individual test case."""
    X_sample_logit = final_classification_model.decision_function(X_test_sample)[0]
    X_sample_probability = final_classification_model.predict_proba(X_test_sample)[0, 1]
    X_sample_class = "Low-Risk" if X_sample_probability < 0.50 else "High-Risk"
    return X_sample_logit, X_sample_probability, X_sample_class

##################################
# Formulating a function to
# compute the risk index,
# estimate the lung cancer probability,
# and predict the risk category
# of a list of train cases
##################################
def compute_list_logit_probability_class(X_train_list):
    """Compute and sort the logit and probability values for a batch of cases."""
    X_list_logit = final_classification_model.decision_function(X_train_list)
    X_list_probability = final_classification_model.predict_proba(X_train_list)[:, 1]
    X_list_logit_index_sorted = np.argsort(X_list_logit)
    X_list_logit_sorted = X_list_logit[X_list_logit_index_sorted]
    X_list_probability_sorted = X_list_probability[X_list_logit_index_sorted]
    return X_list_logit, X_list_probability, X_list_logit_sorted, X_list_probability_sorted

##################################
# Running the Flask app
##################################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
