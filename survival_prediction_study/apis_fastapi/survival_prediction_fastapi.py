##################################
# Loading Python Libraries
##################################
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
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
# Defining the input schema for the function that
# generates the heart failure survival profile,
# estimates the heart failure survival probabilities,
# and predicts the risk category
# of an individual test case
##################################
class TestSample(BaseModel):
    features_individual: List[float]

##################################
# Defining the input schema for the function that
# generates the heart failure survival profile and
# estimates the heart failure survival probabilities
# of a list of train cases
##################################
class TrainList(BaseModel):
    features_list: List[List[float]]

##################################
# Defining the input schema for the function that
# creates dichotomous bins for the numeric features
# of a list of train cases
##################################
class BinningRequest(BaseModel):
    X_original_list: List[dict]
    numeric_feature: str

##################################
# Defining the input schema for the function that
# plots the estimated survival profiles
# using Kaplan-Meier Plots
##################################
class KaplanMeierRequest(BaseModel):
    df: List[dict]
    cat_var: str
    new_case_value: Optional[str] = None

##################################
# Formulating the API endpoints
##################################

##################################
# Initializing the FastAPI app
##################################
app = FastAPI()

##################################
# Defining a GET endpoint for
# for validating API service connection
##################################
@app.get("/")
def root():
    return {"message": "Welcome to the Survival Prediction API!"}

##################################
# Defining a POST endpoint for
# generating the heart failure survival profile,
# estimating the heart failure survival probabilities,
# and predicting the risk category
# of an individual test case
##################################
@app.post("/compute-individual-coxph-survival-probability-class/")
def compute_individual_coxph_survival_probability_class(sample: TestSample):
    try:
        # Defining the column names
        column_names = ["AGE", "ANAEMIA", "EJECTION_FRACTION", "HIGH_BLOOD_PRESSURE", "SERUM_CREATININE", "SERUM_SODIUM"]

        # Converting the data input to a pandas DataFrame with appropriate column names
        X_test_sample = pd.DataFrame([sample.features_individual], columns=column_names)

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
        return {
            "survival_function": survival_function[0].y.tolist(),
            "survival_time": survival_time.tolist(),
            "survival_probabilities": survival_probabilities.tolist(),
            "risk_category": risk_category,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

##################################
# Defining a POST endpoint for
# generating the heart failure survival profile and
# estimating the heart failure survival probabilities
# of a list of train cases
##################################
@app.post("/compute-list-coxph-survival-profile/")
def compute_list_coxph_survival_profile(train_list: TrainList):
    try:
        # Defining the column names
        column_names = ["AGE", "ANAEMIA", "EJECTION_FRACTION", "HIGH_BLOOD_PRESSURE", "SERUM_CREATININE", "SERUM_SODIUM"]

        # Converting the data input to a pandas DataFrame with appropriate column names
        X_train_list = pd.DataFrame(train_list.features_list, columns=column_names)

        # Obtaining the survival function for a batch of cases
        survival_function = final_survival_prediction_model.predict_survival_function(X_train_list)

        # Extracting survival profiles from the survival function output for a batch of cases
        survival_profiles = [sf.y.tolist() for sf in survival_function]

        # Returning the endpoint response
        return {"survival_profiles": survival_profiles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

##################################
# Defining a POST endpoint for
# creating dichotomous bins for the numeric features
# of a list of train cases
##################################
@app.post("/bin-numeric-model-feature/")
def bin_numeric_model_feature(request: BinningRequest):
    try:
        # Converting the data input to a pandas DataFrame with appropriate column names
        X_original_list = pd.DataFrame(request.X_original_list)

        # Computing the median
        median = numeric_feature_median.loc[request.numeric_feature]

        # Dichotomizing the data input to categories based on the median
        X_original_list[request.numeric_feature] = np.where(
            X_original_list[request.numeric_feature] <= median, "Low", "High"
        )

        # Returning the endpoint response
        return X_original_list.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

##################################
# Defining a POST endpoint for
# plotting the estimated survival profiles
# using Kaplan-Meier Plots
##################################
@app.post("/plot-kaplan-meier/")
def plot_kaplan_meier(request: KaplanMeierRequest):
    try:
        # Obtaining plot components
        df = pd.DataFrame(request.df)
        cat_var = request.cat_var
        new_case_value = request.new_case_value

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
        return {"plot": base64_image}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
