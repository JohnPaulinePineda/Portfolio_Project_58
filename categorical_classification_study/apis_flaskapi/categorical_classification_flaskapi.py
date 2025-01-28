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
