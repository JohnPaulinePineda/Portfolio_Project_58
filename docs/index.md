***
# Model Deployment : Exploring Modular Application Programming Interface Frameworks For Serving Model Predictions

***
### [**John Pauline Pineda**](https://github.com/JohnPaulinePineda) <br> <br> *February 1, 2025*
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

![cc_fastapi_code.png](4b7df75c-cef6-4438-a3d2-240bcabc7c92.png)

![cc_fastapi_documentation.png](4df4e089-9764-4883-9642-4fa4cf057de5.png)

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

![sp_fastapi_code.png](c97106a8-9270-4c6b-8513-4259e137e5ad.png)

![sp_fastapi_documentation.png](36d9ee14-88d1-4780-b05f-deaa5e4767a9.png)

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


    
![png](output_47_0.png)
    



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

* **[Book]** [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) by Chip Huyen
* **[Book]** [Machine Learning Bookcamp: Build a Portfolio of Real-Life Projects](https://www.manning.com/books/machine-learning-bookcamp) by Alexey Grigorev and Adam Newmark 
* **[Book]** [Building Machine Learning Pipelines: Automating Model Life Cycles with TensorFlow](https://www.oreilly.com/library/view/building-machine-learning/9781492053187/) by Hannes Hapke and Catherine Nelson
* **[Book]** [Hands-On APIs for AI and Data Science: Python Development with FastAPI](https://handsonapibook.com/index.html) by Ryan Day
* **[Book]** [Managing Machine Learning Projects: From Design to Deployment](https://www.manning.com/books/managing-machine-learning-projects) by Simon Thompson
* **[Book]** [Building Data Science Applications with FastAPI: Develop, Manage, and Deploy Efficient Machine Learning Applications with Python](https://www.oreilly.com/library/view/building-data-science/9781837632749/) by François Voron
* **[Book]** [Microservice APIs: Using Python, Flask, FastAPI, OpenAPI and More](https://www.manning.com/books/microservice-apis) by Jose Haro Peralta
* **[Book]** [Machine Learning Engineering with Python: Manage the Lifecycle of Machine Learning odels using MLOps with Practical Examples](https://www.oreilly.com/library/view/machine-learning-engineering/9781837631964/) by Andrew McMahon
* **[Book]** [Introducing MLOps: How to Scale Machine Learning in the Enterprise](https://www.oreilly.com/library/view/introducing-mlops/9781492083283/) by Mark Treveil, Nicolas Omont, Clément Stenac, Kenji Lefevre, Du Phan, Joachim Zentici, Adrien Lavoillotte, Makoto Miyazaki and Lynn Heidmann
* **[Book]** [Practical Python Backend Programming: Build Flask and FastAPI Applications, Asynchronous Programming, Containerization and Deploy Apps on Cloud](https://leanpub.com/practicalpythonbackendprogramming) by Tim Peters
* **[Python Library API]** [NumPy](https://numpy.org/doc/) by NumPy Team
* **[Python Library API]** [pandas](https://pandas.pydata.org/docs/) by Pandas Team
* **[Python Library API]** [seaborn](https://seaborn.pydata.org/) by Seaborn Team
* **[Python Library API]** [matplotlib.pyplot](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html) by MatPlotLib Team
* **[Python Library API]** [matplotlib.image](https://matplotlib.org/stable/api/image_api.html) by MatPlotLib Team
* **[Python Library API]** [matplotlib.offsetbox](https://matplotlib.org/stable/api/offsetbox_api.html) by MatPlotLib Team
* **[Python Library API]** [itertools](https://docs.python.org/3/library/itertools.html) by Python Team
* **[Python Library API]** [operator](https://docs.python.org/3/library/operator.html) by Python Team
* **[Python Library API]** [sklearn.experimental](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.experimental) by Scikit-Learn Team
* **[Python Library API]** [sklearn.impute](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute) by Scikit-Learn Team
* **[Python Library API]** [sklearn.linear_model](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model) by Scikit-Learn Team
* **[Python Library API]** [sklearn.preprocessing](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) by Scikit-Learn Team
* **[Python Library API]** [scipy](https://docs.scipy.org/doc/scipy/) by SciPy Team
* **[Python Library API]** [sklearn.tree](https://scikit-learn.org/stable/modules/tree.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.ensemble](https://scikit-learn.org/stable/modules/ensemble.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.svm](https://scikit-learn.org/stable/modules/svm.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.model_selection](https://scikit-learn.org/stable/model_selection.html) by Scikit-Learn Team
* **[Python Library API]** [imblearn.over_sampling](https://imbalanced-learn.org/stable/over_sampling.html) by Imbalanced-Learn Team
* **[Python Library API]** [imblearn.under_sampling](https://imbalanced-learn.org/stable/under_sampling.html) by Imbalanced-Learn Team
* **[Python Library API]** [SciKit-Survival](https://pypi.org/project/scikit-survival/) by SciKit-Survival Team
* **[Python Library API]** [SciKit-Learn](https://scikit-learn.org/stable/index.html) by SciKit-Learn Team
* **[Python Library API]** [StatsModels](https://www.statsmodels.org/stable/index.html) by StatsModels Team
* **[Python Library API]** [SciPy](https://scipy.org/) by SciPy Team
* **[Python Library API]** [Lifelines](https://lifelines.readthedocs.io/en/latest/) by Lifelines Team
* **[Python Library API]** [tensorflow](https://pypi.org/project/tensorflow/) by TensorFlow Team
* **[Python Library API]** [keras](https://pypi.org/project/keras/) by Keras Team
* **[Python Library API]** [pil](https://pypi.org/project/Pillow/) by Pillow Team
* **[Python Library API]** [glob](https://docs.python.org/3/library/glob.html) by glob Team
* **[Python Library API]** [cv2](https://pypi.org/project/opencv-python/) by OpenCV Team
* **[Python Library API]** [os](https://docs.python.org/3/library/os.html) by os Team
* **[Python Library API]** [random](https://docs.python.org/3/library/random.html) by random Team
* **[Python Library API]** [keras.models](https://www.tensorflow.org/api_docs/python/tf/keras/models) by TensorFlow Team
* **[Python Library API]** [keras.layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers) by TensorFlow Team
* **[Python Library API]** [keras.wrappers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Wrapper) by TensorFlow Team
* **[Python Library API]** [keras.utils](https://www.tensorflow.org/api_docs/python/tf/keras/utils) by TensorFlow Team
* **[Python Library API]** [keras.optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) by TensorFlow Team
* **[Python Library API]** [keras.preprocessing.image](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image) by TensorFlow Team
* **[Python Library API]** [keras.callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks) by TensorFlow Team
* **[Python Library API]** [keras.metrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics) by TensorFlow Team
* **[Python Library API]** [sklearn.metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) by Scikit-Learn Team
* **[Python Library API]** [Streamlit](https://streamlit.io/) by Streamlit Team
* **[Python Library API]** [Streamlit Community Cloud](https://streamlit.io/cloud) by Streamlit Team
* **[Article]** [ML - Deploy Machine Learning Models Using FastAPI](https://dorian599.medium.com/ml-deploy-machine-learning-models-using-fastapi-6ab6aef7e777) by Dorian Machado (Medium)
* **[Article]** [Deploying Machine Learning Models Using FastAPI](https://medium.com/@kevinnjagi83/deploying-machine-learning-models-using-fastapi-0389c576d8f1) by Kevin Njagi (Medium)
* **[Article]** [Deploy Machine Learning API with FastAPI for Free](https://lightning.ai/lightning-ai/studios/deploy-machine-learning-api-with-fastapi-for-free?section=featured) by Aniket Maurya (Lightning.AI)
* **[Article]** [How to Use FastAPI for Machine Learning](https://blog.jetbrains.com/pycharm/2024/09/how-to-use-fastapi-for-machine-learning/) by Cheuk Ting Ho (JetBrains.Com)
* **[Article]** [Deploying and Hosting a Machine Learning Model with FastAPI and Heroku](https://testdriven.io/blog/fastapi-machine-learning/) by Michael Herman (TestDriven.IO)
* **[Article]** [A Practical Guide to Deploying Machine Learning Models](https://machinelearningmastery.com/a-practical-guide-to-deploying-machine-learning-models/) by Bala Priya (MachineLearningMastery.Com)
* **[Article]** [Using FastAPI to Deploy Machine Learning Models](https://engineering.rappi.com/using-fastapi-to-deploy-machine-learning-models-cd5ed7219ea) by Carl Handlin (Medium)
* **[Article]** [How to Deploy a Machine Learning Model](https://www.maartengrootendorst.com/blog/deploy/) by Maarten Grootendorst (MaartenGrootendorst.Com)
* **[Article]** [Accelerating Machine Learning Deployment: Unleashing the Power of FastAPI and Docker](https://medium.datadriveninvestor.com/accelerating-machine-learning-deployment-unleashing-the-power-of-fastapi-and-docker-933865cb990a) by Pratyush Khare (Medium)
* **[Article]** [Containerize and Deploy ML Models with FastAPI & Docker](https://towardsdev.com/containerize-and-deploy-ml-models-with-fastapi-docker-d8c19cc8ef94) by Hemachandran Dhinakaran (Medium)
* **[Article]** [Quick Tutorial to Deploy Your ML models using FastAPI and Docker](https://shreyansh26.github.io/post/2020-11-30_fast_api_docker_ml_deploy/) by Shreyansh Singh (GitHub)
* **[Article]** [How to Deploying Machine Learning Models in Production](https://levelup.gitconnected.com/how-to-deploying-machine-learning-models-in-production-3009b90eadfa) by Umair Akram (Medium)
* **[Article]** [Deploying a Machine Learning Model with FastAPI: A Comprehensive Guide](https://ai.plainenglish.io/deploying-a-machine-learning-model-with-fastapi-a-comprehensive-guide-997ac747601d) by Muhammad Naveed Arshad (Medium)
* **[Article]** [Deploy Machine Learning Model with REST API using FastAPI](https://blog.yusufberki.net/deploy-machine-learning-model-with-rest-api-using-fastapi-288f229161b7) by Yusuf Berki Yazıcıoğlu (Medium)
* **[Article]** [Deploying An ML Model With FastAPI — A Succinct Guide](https://towardsdatascience.com/deploying-an-ml-model-with-fastapi-a-succinct-guide-69eceda27b21) by Yash Prakash (Medium)
* **[Article]** [How to Build a Machine Learning App with FastAPI: Dockerize and Deploy the FastAPI Application to Kubernetes](https://dev.to/bravinsimiyu/beginner-guide-on-how-to-build-a-machine-learning-app-with-fastapi-part-ii-deploying-the-fastapi-application-to-kubernetes-4j6g) by Bravin Wasike (Dev.TO)
* **[Article]** [Building a Machine Learning Model API with Flask: A Step-by-Step Guide](https://medium.com/@nileshshindeofficial/building-a-machine-learning-model-api-with-flask-a-step-by-step-guide-6f85e9bb9773) by Nilesh Shinde (Medium)
* **[Article]** [Deploying Your Machine Learning Model as a REST API Using Flask](https://medium.com/analytics-vidhya/deploying-your-machine-learning-model-as-a-rest-api-using-flask-c2e6a0b574f5) by Emmanuel Oludare (Medium)
* **[Article]** [Machine Learning Model Deployment on Heroku Using Flask](https://towardsdatascience.com/machine-learning-model-deployment-on-heroku-using-flask-467acb4a34da) by Charu Makhijani (Medium)
* **[Article]** [Model Deployment using Flask](https://towardsdatascience.com/model-deployment-using-flask-c5dcbb6499c9) by Ravindra Sharma (Medium)



```python
from IPython.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>"))
```


<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>

