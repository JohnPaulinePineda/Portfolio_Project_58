##################################
# Loading Python libraries
##################################
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import joblib
import numpy as np
import pandas as pd 

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
# Defining the input schema for the function that
# computes the risk index,
# estimates the lung cancer probability,
# and predicts the risk category
# of an individual test case
##################################
class TestSample(BaseModel):
    features_individual: list[float]

##################################
# Defining the input schema for the function that
# computes the risk index,
# estimates the lung cancer probability,
# and predicts the risk category
# of a list of train cases
##################################
class TestBatch(BaseModel):
    features_list: list[list[float]] 

##################################
# Formulating the API endpoints
##################################

##################################
# Initializing the FastAPI app
##################################
app = FastAPI()

##################################
# Defining a GET endpoint for
# validating API service connection
##################################
@app.get("/")
def root():
    return {"message": "Welcome to the Categorical Classification API!"}

##################################
# Defining a POST endpoint for
# computing the risk index,
# estimating the lung cancer probability,
# and predicting the risk category
# of an individual test case
##################################
@app.post("/predict-individual-logit-probability-class")
def predict_individual_logit_probability_class(input_data: TestSample):
    try:
        # Converting the data input to a DataFrame with proper feature names
        X_test_sample = pd.DataFrame([input_data.features_individual], columns=final_classification_model.feature_names_in_)

        # Obtaining the estimated logit and probability values for an individual test case
        logit, probability, risk_class = compute_individual_logit_probability_class(X_test_sample)

        # Returning the endpoint response
        return {
            "logit": logit,
            "probability": probability,
            "risk_class": risk_class,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")

##################################
# Defining a POST endpoint for
# computing the risk index,
# estimating the lung cancer probability,
# and predicting the risk category
# of a list of train cases
##################################
@app.post("/predict-list-logit-probability-class")
def predict_list_logit_probability_class(input_data: TestBatch):
    try:
        # Converting the data input to a DataFrame with proper feature names
        X_train_list = pd.DataFrame(input_data.features_list, columns=final_classification_model.feature_names_in_)

        # Obtaining the estimated logit and probability values for a batch of cases
        logit, probability, logit_sorted, probability_sorted = compute_list_logit_probability_class(X_train_list)

        # Returning the endpoint response
        return {
            "logit": logit.tolist(),
            "probability": probability.tolist(),
            "logit_sorted": logit_sorted.tolist(),
            "probability_sorted": probability_sorted.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")

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
