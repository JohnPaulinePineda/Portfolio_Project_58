***
# Model Deployment : Exploring Modular Application Programming Interface Frameworks For Serving Model Predictions

***
### [**John Pauline Pineda**](https://github.com/JohnPaulinePineda) <br> <br> *January 25, 2025*
***

* [**1. Table of Contents**](#TOC)
    * [1.1 Project Background](#1.1)
        * [1.1.1 Categorical Classfication](#1.1.1)
            * [1.1.1.1 Data Background](#1.1.1.1)
            * [1.1.1.2 Model Background](#1.1.2.1)
        * [1.1.2 Survival Prediction](#1.1.2)
            * [1.1.2.1 Data Background](#1.1.2.1)
            * [1.1.2.2 Model Background](#1.1.2.2)
        * [1.1.3 Survival Prediction](#1.1.3)
            * [1.1.3.1 Data Background](#1.1.2.1)
            * [1.1.3.2 Model Background](#1.1.3.2)
    * [1.2 Application Programming Interface (API) Development Using the FastAPI Framework](#1.2)
        * [1.2.1 Categorical Classfication](#1.2.1)
            * [1.2.1.1 API Building](#1.2.1.1)
            * [1.2.1.2 API Testing](#1.2.1.2)
        * [1.2.2 Categorical Classfication](#1.2.2)
            * [1.2.2.1 API Building](#1.2.2.1)
            * [1.2.2.2 API Testing](#1.2.2.2)
        * [1.2.3 Categorical Classfication](#1.2.3)
            * [1.2.3.1 API Building](#1.2.3.1)
            * [1.2.3.2 API Testing](#1.2.3.2)
    * [1.3 Application Programming Interface (API) Development Using the Flask Framework](#1.3)
        * [1.3.1 Categorical Classfication](#1.3.1)
            * [1.3.1.1 API Building](#1.3.1.1)
            * [1.3.1.2 API Testing](#1.3.1.2)
        * [1.3.2 Categorical Classfication](#1.3.2)
            * [1.3.2.1 API Building](#1.3.2.1)
            * [1.3.2.2 API Testing](#1.3.2.2)
        * [1.3.3 Categorical Classfication](#1.3.3)
            * [1.3.3.1 API Building](#1.3.3.1)
            * [1.3.3.2 API Testing](#1.3.3.2)
    * [1.4 Consolidated Findings](#1.1)
* [**2. Summary**](#Summary)   
* [**3. References**](#References)

***

# 1. Table of Contents <a class="anchor" id="TOC"></a>

## 1.1. Project Background <a class="anchor" id="1.1"></a>

### 1.1.1 Categorical Classification <a class="anchor" id="1.1.1"></a>

#### 1.1.1.1 Data Background <a class="anchor" id="1.1.1.1"></a>

#### 1.1.1.2 Model Background <a class="anchor" id="1.1.1.2"></a>

### 1.1.2 Survival Prediction <a class="anchor" id="1.1.2"></a>

#### 1.1.2.1 Data Background <a class="anchor" id="1.1.2.1"></a>

#### 1.1.2.2 Model Background <a class="anchor" id="1.1.2.2"></a>

### 1.1.3 Image Classfication <a class="anchor" id="1.1.3"></a>

#### 1.1.3.1 Data Background <a class="anchor" id="1.1.3.1"></a>

#### 1.1.3.2 Model Background <a class="anchor" id="1.1.3.2"></a>

## 1.2. Application Programming Interface (API) Development Using the FastAPI Framework <a class="anchor" id="1.2"></a>

### 1.2.1 Categorical Classification <a class="anchor" id="1.2.1"></a>

#### 1.2.1.1 API Building <a class="anchor" id="1.2.1.1"></a>

#### 1.2.1.2 API Testing <a class="anchor" id="1.2.1.2"></a>

![cc_fastapi_activation.png](1ef82129-c142-4aad-a439-7109bea39923.png)

![cc_fastapi_endpoints.png](e9dc12de-815b-4ef4-a347-042f7c45e40c.png)


```python
##################################
# Loading Python Libraries
##################################
import requests

```


```python
##################################
# Defining the base URL of the API
# for the categorical classification model
##################################
CC_BASE_URL = "http://127.0.0.1:8000"

```


```python
##################################
# Defining the input values for an individual test case
##################################
individual_test_case = {
    "features_individual": [1, 0, 0, 0, 0, 1, 0, 0, 1, 1] 
}

```


```python
##################################
# Defining the input values for a batch of cases
##################################
batch_test_case = {
    "features_list": [
        [1, 0, 0, 0, 0, 1, 0, 0, 1, 1], 
        [1, 0, 1, 0, 1, 1, 0, 1, 1, 1]
    ]
}

```


```python
##################################
# Generating a GET endpoint request for
# for validating API service connection
##################################
response = requests.get(f"{CC_BASE_URL}/")
if response.status_code == 200:
    print("Root Endpoint Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
    
```

    Root Endpoint Response: {'message': 'Welcome to the Categorical Classification API!'}
    


```python
##################################
# Generating a POST endpoint request for
# computing the risk index,
# estimating the lung cancer probability,
# and predicting the risk category
# of an individual test case
##################################
response = requests.post(f"{CC_BASE_URL}/predict-individual-logit-probability-class", json=individual_test_case)
if response.status_code == 200:
    display("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
    
```


    'Response:'



    {'logit': -1.2117837409390746,
     'probability': 0.22938559072691203,
     'risk_class': 'Low-Risk'}



```python
##################################
# Sending a POST endpoint request for
# computing the risk index,
# estimating the lung cancer probability,
# and predicting the risk category
# of a list of train cases
##################################
response = requests.post(f"{CC_BASE_URL}/predict-list-logit-probability-class", json=batch_test_case)
if response.status_code == 200:
    display("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
    
```


    'Response:'



    {'logit': [-1.2117837409390746, 3.4784950973590973],
     'probability': [0.22938559072691203, 0.9700696569701589],
     'logit_sorted': [-1.2117837409390746, 3.4784950973590973],
     'probability_sorted': [0.22938559072691203, 0.9700696569701589]}



```python
##################################
# Sending a POST endpoint request
# using malformed data to evaluate
# the API's error handling function
##################################
malformed_test_case = {"features": [1, 0, 1]}
response = requests.post(f"{CC_BASE_URL}/predict-individual-logit-probability-class", json=malformed_test_case)
if response.status_code == 200:
    display("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
    
```

    Error: 422 {"detail":[{"type":"missing","loc":["body","features_individual"],"msg":"Field required","input":{"features":[1,0,1]}}]}
    

### 1.2.2 Survival Prediction <a class="anchor" id="1.2.2"></a>

#### 1.2.2.1 API Building <a class="anchor" id="1.2.2.1"></a>

#### 1.2.2.2 API Testing <a class="anchor" id="1.2.2.2"></a>

![sp_fastapi_activation.png](2cc97028-ce4a-4cf0-948f-840b22cdbd0e.png)

![sp_fastapi_endpoints.png](459dc1e7-32ad-4854-98ac-f12992cce1ca.png)


```python
##################################
# Loading Python Libraries
##################################
import requests
import json
import pandas as pd
import base64
from IPython.display import Image, display

```


```python
##################################
# Defining the base URL of the API
# for the categorical classification model
##################################
SP_BASE_URL = "http://127.0.0.1:8001"

```


```python
##################################
# Defining the input values for an individual test case
##################################
single_test_case = {
    "features_individual": [43, 0, 75, 1, 0.75, 100]  
}

```


```python
##################################
# Defining the input values for a batch of cases
##################################
train_list = {
        "features_list": [
            [43, 0, 75, 1, 0.75, 100],
            [70, 1,	20,	1, 0.75, 100]
        ]
    }

```


```python
##################################
# Defining the input values for a batch of cases for binning request
##################################
bin_request = {
        "X_original_list": [
            {"AGE": -0.10, "EJECTION_FRACTION": -0.10, "SERUM_CREATININE ": -0.10, "SERUM_SODIUM": -0.10},
            {"AGE": 0.20, "EJECTION_FRACTION": 0.20, "SERUM_CREATININE ": 0.20, "SERUM_SODIUM": 0.20},
            {"AGE": 0.90, "EJECTION_FRACTION": 0.90, "SERUM_CREATININE ": 0.90, "SERUM_SODIUM": 0.90}
        ],
        "numeric_feature": "AGE"
    }

```


```python
##################################
# Defining the input values for a batch of cases for Kaplan-Meier plotting
##################################
km_request = {
        "df": [
            {"TIME": 0, "DEATH_EVENT": 0, "AGE": "Low"},
            {"TIME": 25, "DEATH_EVENT": 0, "AGE": "Low"},
            {"TIME": 50, "DEATH_EVENT": 0, "AGE": "Low"},
            {"TIME": 100, "DEATH_EVENT": 0, "AGE": "Low"},
            {"TIME": 125, "DEATH_EVENT": 0, "AGE": "Low"},
            {"TIME": 150, "DEATH_EVENT": 0, "AGE": "Low"},
            {"TIME": 175, "DEATH_EVENT": 0, "AGE": "Low"},
            {"TIME": 200, "DEATH_EVENT": 0, "AGE": "Low"},
            {"TIME": 225, "DEATH_EVENT": 1, "AGE": "Low"},
            {"TIME": 250, "DEATH_EVENT": 1, "AGE": "Low"},
            {"TIME": 0, "DEATH_EVENT": 0, "AGE": "High"},
            {"TIME": 25, "DEATH_EVENT": 0, "AGE": "High"},
            {"TIME": 50, "DEATH_EVENT": 0, "AGE": "High"},
            {"TIME": 100, "DEATH_EVENT": 1, "AGE": "High"},
            {"TIME": 125, "DEATH_EVENT": 0, "AGE": "High"},
            {"TIME": 150, "DEATH_EVENT": 0, "AGE": "High"},
            {"TIME": 175, "DEATH_EVENT": 1, "AGE": "High"},
            {"TIME": 200, "DEATH_EVENT": 1, "AGE": "High"},
            {"TIME": 225, "DEATH_EVENT": 1, "AGE": "High"},
            {"TIME": 250, "DEATH_EVENT": 1, "AGE": "High"},
        ],
        "cat_var": "AGE",
        "new_case_value": "Low"
    }

```


```python
##################################
# Generating a GET endpoint request for
# for validating API service connection
##################################
response = requests.get(f"{SP_BASE_URL}/")
if response.status_code == 200:
    display("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)

```


    'Response:'



    {'message': 'Welcome to the Survival Prediction API!'}



```python
##################################
# Sending a POST endpoint request for
# generating the heart failure survival profile,
# estimating the heart failure survival probabilities,
# and predicting the risk category
# of an individual test case
##################################
response = requests.post(f"{SP_BASE_URL}/compute-individual-coxph-survival-probability-class/", json=single_test_case)
if response.status_code == 200:
    display("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
    
```


    'Response:'



    {'survival_function': [0.9973812917524568,
      0.9920416812438736,
      0.9893236791425079,
      0.972381113071464,
      0.9693179903073035,
      0.9631930672135339,
      0.9631930672135339,
      0.9600469571766689,
      0.9600469571766689,
      0.9568596864927983,
      0.9536305709158891,
      0.9471625843882805,
      0.93729581350105,
      0.9338986486591409,
      0.93048646553474,
      0.9270645831787163,
      0.9202445006124622,
      0.9167715111530355,
      0.9132845175345189,
      0.9097550958520674,
      0.9097550958520674,
      0.9097550958520674,
      0.9060810720432387,
      0.9024157452999795,
      0.9024157452999795,
      0.9024157452999795,
      0.9024157452999795,
      0.9024157452999795,
      0.8985598696587259,
      0.8985598696587259,
      0.8985598696587259,
      0.8945287485160898,
      0.8945287485160898,
      0.8945287485160898,
      0.8945287485160898,
      0.8901959645503091,
      0.8812352215018253,
      0.8812352215018253,
      0.8812352215018253,
      0.8812352215018253,
      0.8764677174183527,
      0.8764677174183527,
      0.8764677174183527,
      0.8764677174183527,
      0.8709113650481243,
      0.8709113650481243,
      0.8652494086650531,
      0.8593884303802698,
      0.8593884303802698,
      0.8593884303802698,
      0.8593884303802698,
      0.8593884303802698,
      0.8528574859874233,
      0.8528574859874233,
      0.8528574859874233,
      0.8528574859874233,
      0.8528574859874233,
      0.8459534502216807,
      0.8389821875092403,
      0.8319419786276306,
      0.8246669811915435,
      0.8099879066057215,
      0.8099879066057215,
      0.7943979200335176,
      0.7943979200335176,
      0.7943979200335176,
      0.7943979200335176,
      0.7943979200335176,
      0.7848178617845467,
      0.7848178617845467,
      0.7848178617845467,
      0.7848178617845467,
      0.7741993572193384,
      0.7741993572193384,
      0.7741993572193384,
      0.7741993572193384,
      0.7741993572193384,
      0.7741993572193384,
      0.7741993572193384,
      0.7555469848652164,
      0.7555469848652164,
      0.7555469848652164,
      0.7555469848652164,
      0.7555469848652164,
      0.7337716342207724,
      0.7337716342207724,
      0.7337716342207724,
      0.7070184115456696,
      0.7070184115456696,
      0.7070184115456696,
      0.7070184115456696,
      0.7070184115456696,
      0.7070184115456696,
      0.7070184115456696,
      0.7070184115456696,
      0.7070184115456696],
     'survival_time': [50, 100, 150, 200, 250],
     'survival_probabilities': [90.97550958520674,
      87.64677174183527,
      84.59534502216806,
      78.48178617845467,
      70.70184115456696],
     'risk_category': 'Low-Risk'}



```python
##################################
# Sending a POST endpoint request for
# generating the heart failure survival profile and
# estimating the heart failure survival probabilities
# of a list of train cases
##################################
response = requests.post(f"{SP_BASE_URL}/compute-list-coxph-survival-profile/", json=train_list)
if response.status_code == 200:
    display("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
    
```


    'Response:'



    {'survival_profiles': [[0.9973812917524568,
       0.9920416812438736,
       0.9893236791425079,
       0.972381113071464,
       0.9693179903073035,
       0.9631930672135339,
       0.9631930672135339,
       0.9600469571766689,
       0.9600469571766689,
       0.9568596864927983,
       0.9536305709158891,
       0.9471625843882805,
       0.93729581350105,
       0.9338986486591409,
       0.93048646553474,
       0.9270645831787163,
       0.9202445006124622,
       0.9167715111530355,
       0.9132845175345189,
       0.9097550958520674,
       0.9097550958520674,
       0.9097550958520674,
       0.9060810720432387,
       0.9024157452999795,
       0.9024157452999795,
       0.9024157452999795,
       0.9024157452999795,
       0.9024157452999795,
       0.8985598696587259,
       0.8985598696587259,
       0.8985598696587259,
       0.8945287485160898,
       0.8945287485160898,
       0.8945287485160898,
       0.8945287485160898,
       0.8901959645503091,
       0.8812352215018253,
       0.8812352215018253,
       0.8812352215018253,
       0.8812352215018253,
       0.8764677174183526,
       0.8764677174183526,
       0.8764677174183526,
       0.8764677174183526,
       0.8709113650481243,
       0.8709113650481243,
       0.8652494086650531,
       0.8593884303802697,
       0.8593884303802697,
       0.8593884303802697,
       0.8593884303802697,
       0.8593884303802697,
       0.8528574859874233,
       0.8528574859874233,
       0.8528574859874233,
       0.8528574859874233,
       0.8528574859874233,
       0.8459534502216807,
       0.8389821875092403,
       0.8319419786276306,
       0.8246669811915435,
       0.8099879066057215,
       0.8099879066057215,
       0.7943979200335176,
       0.7943979200335176,
       0.7943979200335176,
       0.7943979200335176,
       0.7943979200335176,
       0.7848178617845467,
       0.7848178617845467,
       0.7848178617845467,
       0.7848178617845467,
       0.7741993572193384,
       0.7741993572193384,
       0.7741993572193384,
       0.7741993572193384,
       0.7741993572193384,
       0.7741993572193384,
       0.7741993572193384,
       0.7555469848652164,
       0.7555469848652164,
       0.7555469848652164,
       0.7555469848652164,
       0.7555469848652164,
       0.7337716342207724,
       0.7337716342207724,
       0.7337716342207724,
       0.7070184115456695,
       0.7070184115456695,
       0.7070184115456695,
       0.7070184115456695,
       0.7070184115456695,
       0.7070184115456695,
       0.7070184115456695,
       0.7070184115456695,
       0.7070184115456695],
      [0.9761144218801228,
       0.928980888267716,
       0.905777064852962,
       0.7724242339590301,
       0.7502787164583535,
       0.7076872741961866,
       0.7076872741961866,
       0.6866593185026403,
       0.6866593185026403,
       0.6659260634219393,
       0.6454915885099762,
       0.6062342264686207,
       0.5504405490863784,
       0.5323184765768243,
       0.5146536440658629,
       0.49746533888245986,
       0.464726413395405,
       0.4488047347163327,
       0.433309930422515,
       0.4181140392975694,
       0.4181140392975694,
       0.4181140392975694,
       0.4028020116306455,
       0.38802639234746467,
       0.38802639234746467,
       0.38802639234746467,
       0.38802639234746467,
       0.38802639234746467,
       0.373006008573048,
       0.373006008573048,
       0.373006008573048,
       0.35785929480690143,
       0.35785929480690143,
       0.35785929480690143,
       0.35785929480690143,
       0.3421927616040032,
       0.3117176431598899,
       0.3117176431598899,
       0.3117176431598899,
       0.3117176431598899,
       0.29651072362871467,
       0.29651072362871467,
       0.29651072362871467,
       0.29651072362871467,
       0.2796248763802668,
       0.2796248763802668,
       0.2633052706162029,
       0.24731169874453887,
       0.24731169874453887,
       0.24731169874453887,
       0.24731169874453887,
       0.24731169874453887,
       0.23051507888001282,
       0.23051507888001282,
       0.23051507888001282,
       0.23051507888001282,
       0.23051507888001282,
       0.213871875082776,
       0.19816204676152543,
       0.18334919195124752,
       0.16908728217620728,
       0.1432835564355214,
       0.1432835564355214,
       0.119778195679268,
       0.119778195679268,
       0.119778195679268,
       0.119778195679268,
       0.119778195679268,
       0.10710184631976834,
       0.10710184631976834,
       0.10710184631976834,
       0.10710184631976834,
       0.09446095671842468,
       0.09446095671842468,
       0.09446095671842468,
       0.09446095671842468,
       0.09446095671842468,
       0.09446095671842468,
       0.09446095671842468,
       0.07544024472233593,
       0.07544024472233593,
       0.07544024472233593,
       0.07544024472233593,
       0.07544024472233593,
       0.05761125190131533,
       0.05761125190131533,
       0.05761125190131533,
       0.040906392932898404,
       0.040906392932898404,
       0.040906392932898404,
       0.040906392932898404,
       0.040906392932898404,
       0.040906392932898404,
       0.040906392932898404,
       0.040906392932898404,
       0.040906392932898404]]}



```python
##################################
# Sending a POST endpoint request for
# creating dichotomous bins for the numeric features
# of a list of train cases
##################################
response = requests.post(f"{SP_BASE_URL}/bin-numeric-model-feature/", json=bin_request)
if response.status_code == 200:
    display("Response:", pd.DataFrame(response.json()))
else:
    print("Error:", response.status_code, response.text)
    
```


    'Response:'



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AGE</th>
      <th>EJECTION_FRACTION</th>
      <th>SERUM_CREATININE</th>
      <th>SERUM_SODIUM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Low</td>
      <td>-0.1</td>
      <td>-0.1</td>
      <td>-0.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>High</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>High</td>
      <td>0.9</td>
      <td>0.9</td>
      <td>0.9</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Sending a POST endpoint request for
# plotting the estimated survival profiles
# using Kaplan-Meier Plots
##################################
response = requests.post(f"{SP_BASE_URL}/plot-kaplan-meier/", json=km_request)
if response.status_code == 200:
    plot_data = response.json()["plot"]
    # Decoding and displaying the plot
    img = base64.b64decode(plot_data)
    with open("kaplan_meier_plot.png", "wb") as f:
        f.write(img)
        display(Image("kaplan_meier_plot.png"))
else:
    print("Error:", response.status_code, response.text)
    
```


    
![png](output_43_0.png)
    



```python
##################################
# Sending a POST endpoint request
# using malformed data to evaluate
# the API's error handling function
##################################
malformed_test_case = {"features": [43, 0, 75, 1, 0.75]}
response = requests.post(f"{SP_BASE_URL}/compute-individual-coxph-survival-probability-class", json=malformed_test_case)
if response.status_code == 200:
    display("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
    
```

    Error: 422 {"detail":[{"type":"missing","loc":["body","features_individual"],"msg":"Field required","input":{"features":[43,0,75,1,0.75]}}]}
    

### 1.2.3 Image Classfication <a class="anchor" id="1.2.3"></a>

#### 1.2.3.1 API Building <a class="anchor" id="1.2.3.1"></a>

#### 1.2.3.2 API Testing <a class="anchor" id="1.2.3.2"></a>

## 1.3. Application Programming Interface (API) Development Using the Flask Framework <a class="anchor" id="1.3"></a>

### 1.3.1 Categorical Classification <a class="anchor" id="1.3.1"></a>

#### 1.3.1.1 API Building <a class="anchor" id="1.3.1.1"></a>

#### 1.3.1.2 API Testing <a class="anchor" id="1.3.1.2"></a>

### 1.3.2 Survival Prediction <a class="anchor" id="1.3.2"></a>

#### 1.3.2.1 API Building <a class="anchor" id="1.3.2.1"></a>

#### 1.3.2.2 API Testing <a class="anchor" id="1.3.2.2"></a>

### 1.3.3 Image Classfication <a class="anchor" id="1.3.3"></a>

#### 1.3.3.1 API Building <a class="anchor" id="1.3.3.1"></a>

#### 1.3.3.2 API Testing <a class="anchor" id="1.3.3.2"></a>

## 1.4. Consolidated Findings <a class="anchor" id="1.4"></a>

# 2. Summary <a class="anchor" id="Summary"></a>

# 3. References <a class="anchor" id="References"></a>
