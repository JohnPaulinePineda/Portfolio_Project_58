***
# Model Deployment : Exploring Modular Application Programming Interface Frameworks For Serving Model Predictions

***
### [**John Pauline Pineda**](https://github.com/JohnPaulinePineda) <br> <br> *February 8, 2025*
***

* [**1. Table of Contents**](#TOC)
    * [1.1 Project Background](#1.1)
        * [1.1.1 Categorical Classification](#1.1.1)
            * [1.1.1.1 Data Background](#1.1.1.1)
            * [1.1.1.2 Model Background](#1.1.1.2)
            * [1.1.1.3 Deployment Background](#1.1.1.3)
        * [1.1.2 Survival Prediction](#1.1.2)
            * [1.1.2.1 Data Background](#1.1.2.1)
            * [1.1.2.2 Model Background](#1.1.2.2)
            * [1.1.2.3 Deployment Background](#1.1.2.3)
        * [1.1.3 Image Classification](#1.1.3)
            * [1.1.3.1 Data Background](#1.1.3.1)
            * [1.1.3.2 Model Background](#1.1.3.2)
            * [1.1.3.3 Deployment Background](#1.1.3.3)
    * [1.2 Application Programming Interface (API) Development Using the FastAPI Framework](#1.2)
        * [1.2.1 Categorical Classification](#1.2.1)
            * [1.2.1.1 API Building](#1.2.1.1)
            * [1.2.1.2 API Testing](#1.2.1.2)
        * [1.2.2 Survival Prediction](#1.2.2)
            * [1.2.2.1 API Building](#1.2.2.1)
            * [1.2.2.2 API Testing](#1.2.2.2)
        * [1.2.3 Image Classification](#1.2.3)
            * [1.2.3.1 API Building](#1.2.3.1)
            * [1.2.3.2 API Testing](#1.2.3.2)
    * [1.3 Application Programming Interface (API) Development Using the Flask Framework](#1.3)
        * [1.3.1 Categorical Classification](#1.3.1)
            * [1.3.1.1 API Building](#1.3.1.1)
            * [1.3.1.2 API Testing](#1.3.1.2)
        * [1.3.2 Survival Prediction](#1.3.2)
            * [1.3.2.1 API Building](#1.3.2.1)
            * [1.3.2.2 API Testing](#1.3.2.2)
        * [1.3.3 Image Classification](#1.3.3)
            * [1.3.3.1 API Building](#1.3.3.1)
            * [1.3.3.2 API Testing](#1.3.3.2)
    * [1.4 Consolidated Findings](#1.1)
* [**2. Summary**](#Summary)   
* [**3. References**](#References)

***

# 1. Table of Contents <a class="anchor" id="TOC"></a>

This project explores the modular deployment of machine learning models using Representational State Transfer (RESTful) Application Programming Interfaces (APIs), specifically comparing **FastAPI** and **Flask** frameworks in <mark style="background-color: #CCECFF"><b>Python</b></mark>. Pre-trained models were loaded and integrated into the APIs, including a [**Stacked ensemble binary classification model for predicting lung cancer probabilities**](https://johnpaulinepineda.github.io/Portfolio_Project_54/), a [**Cox Proportional Hazards survival prediction model for estimating heart failure survival profiles**](https://johnpaulinepineda.github.io/Portfolio_Project_55/), and a [**Convolutional Neural Network-based image classification model for determining class categories for brain magnetic resonance images**](https://johnpaulinepineda.github.io/Portfolio_Project_56/). The study objectives included understanding the similarities and differences between FastAPI and Flask in implementating and documenting RESTful API endpoints to process data preprocessing and model prediction logic, handling a variety of input types (such as structured data for class and survival probability predictions and file uploads for image classification), enabling diverse output formats (including string, float, list and base64-encoded visualization objects, depending on the model), and ensuring robust error handling and validation. All results were consolidated in a [<span style="color: #FF0000"><b>Summary</b></span>](#Summary) presented at the end of the document. 

[RESTful APIs](https://www.oreilly.com/library/view/building-machine-learning/9781492045106/) are a standardized architectural style for designing networked applications, enabling communication between clients and servers over HTTP. They use HTTP methods like GET, POST, PUT, and DELETE to perform CRUD (Create, Read, Update, Delete) operations on resources, which are typically represented in JSON or XML format. RESTful APIs are stateless, meaning each request from a client to a server must contain all the information needed to process the request, ensuring scalability and reliability. For machine learning model deployment, RESTful APIs serve as a bridge between trained models and end-users or applications, allowing models to be accessed remotely via HTTP requests. This enables real-time predictions, batch processing, and integration with web or mobile applications. RESTful APIs are particularly significant for machine learning because they provide a standardized, platform-agnostic way to serve predictions, making models accessible to a wide range of clients. They also facilitate modularity, as models can be updated or replaced without affecting the client-side application. Additionally, RESTful APIs support scalability, as they can be deployed on cloud platforms and scaled horizontally to handle increased traffic. Error handling and validation mechanisms in RESTful APIs ensure robustness, which is critical for machine learning applications where malformed inputs can lead to incorrect predictions. By encapsulating machine learning logic behind APIs, developers can abstract away the complexity of model inference, making it easier for non-technical users to interact with the models. RESTful APIs also enable versioning, allowing multiple versions of a model to coexist and be accessed independently. This is particularly useful for A/B testing or gradual rollouts of updated models. Furthermore, RESTful APIs can be secured using authentication and authorization mechanisms, ensuring that only authorized users or applications can access the model. Overall, RESTful APIs are a cornerstone of modern machine learning deployment, providing a flexible, scalable, and secure way to serve predictions in production environments.

[FastAPI](https://fastapi.tiangolo.com/) is a modern, high-performance web framework for building APIs with Python, specifically designed for speed and ease of use. It is built on top of Starlette for web handling and Pydantic for data validation, making it one of the fastest Python frameworks available. FastAPI leverages Python type hints to automatically generate OpenAPI (Swagger) documentation, which simplifies API testing and debugging. Its asynchronous capabilities, powered by Python’s async and await keywords, make it ideal for handling high-concurrency workloads, such as serving machine learning models to multiple clients simultaneously. FastAPI’s built-in dependency injection system allows for modular and reusable code, which is particularly useful for complex machine learning pipelines. The framework also supports WebSocket communication, enabling real-time interactions, though this is less commonly used in machine learning deployments. One of FastAPI’s key strengths is its automatic data validation using Pydantic, which ensures that inputs to machine learning models are correctly formatted and reduces the risk of errors during inference. However, FastAPI’s reliance on asynchronous programming can be a double-edged sword; while it improves performance, it may require developers to have a deeper understanding of asynchronous programming concepts. Additionally, FastAPI’s ecosystem, while growing, is still smaller than that of more established frameworks like Flask, which can limit the availability of third-party plugins and extensions. Despite these limitations, FastAPI is widely regarded as an excellent choice for machine learning model deployment due to its speed, scalability, and developer-friendly features. Its ability to automatically generate API documentation and validate inputs makes it particularly well-suited for production environments where reliability and maintainability are critical.

[Flask](https://flask.palletsprojects.com/en/stable/) Flask is a lightweight and flexible web framework for Python, designed to be simple and easy to use while providing the essentials for building web applications and APIs. It follows the WSGI (Web Server Gateway Interface) standard and is often described as a "micro-framework" because it provides only the core components needed for web development, such as routing and request handling, while leaving other functionalities to extensions. Flask’s simplicity and minimalistic design make it highly customizable, allowing developers to tailor it to specific use cases, including machine learning model deployment. Flask’s synchronous nature makes it easier to understand and use for developers who are not familiar with asynchronous programming, though this can limit its performance in high-concurrency scenarios. For machine learning deployments, Flask’s simplicity is both a strength and a weakness; while it is easy to set up and deploy, it lacks built-in features like data validation and automatic documentation, which must be implemented manually or through extensions like Flasgger. Flask’s extensive ecosystem of extensions, such as Flask-RESTful for building RESTful APIs and Flask-SQLAlchemy for database integration, provides additional functionality but can also introduce complexity. One of Flask’s key strengths is its widespread adoption and community support, which makes it easier to find tutorials, documentation, and third-party tools. However, Flask’s synchronous architecture can become a bottleneck when serving machine learning models to multiple clients simultaneously, as it may struggle to handle high traffic efficiently. Despite these limitations, Flask remains a popular choice for machine learning model deployment, particularly for smaller-scale applications or prototypes where simplicity and ease of use are prioritized over performance. Its flexibility and extensive ecosystem make it a versatile tool for developers, though it may require more effort to achieve the same level of robustness and scalability as FastAPI.


## 1.1. Project Background <a class="anchor" id="1.1"></a>

### 1.1.1 Categorical Classification <a class="anchor" id="1.1.1"></a>

This project implements the **Logistic Regression Model** as an independent learner and as a meta-learner of a stacking ensemble model with **Decision Trees**, **Random Forest**, and **Support Vector Machine** classifier algorithms using various helpful packages in <mark style="background-color: #CCECFF"><b>Python</b></mark> to estimate probability of a dichotomous categorical response variable by modelling the relationship between one or more predictor variables and a binary outcome. The resulting predictions derived from the candidate models were evaluated using the **F1 Score** that ensures both false positives and false negatives are considered, providing a more balanced view of model classification performance. Resampling approaches including **Synthetic Minority Oversampling Technique** and **Condensed Nearest Neighbors** for imbalanced classification problems were applied by augmenting the dataset used for model training based on its inherent characteristics to achieve a more reasonably balanced distribution between the majority and minority classes. Additionally, **Class Weights** were also implemented by amplifying the loss contributed by the minority class and diminishing the loss from the majority class, forcing the model to focus more on correctly predicting the minority class. Penalties including **Least Absolute Shrinkage and Selection Operator** and **Ridge Regularization** were evaluated to impose constraints on the model coefficient updates. 

* The complete data preprocessing and model development process was consolidated in this [**Jupyter Notebook**](https://johnpaulinepineda.github.io/Portfolio_Project_54/).
* All associated datasets and code files were stored in this [**GitHub Project Repository**](https://github.com/JohnPaulinePineda/Portfolio_Project_54). 
* The final model was deployed as a prototype application with a web interface via [**Streamlit**](https://lung-cancer-diagnosis-probability-estimation.streamlit.app/).


#### 1.1.1.1 Data Background <a class="anchor" id="1.1.1.1"></a>

1. The original dataset comprised rows representing observations and columns representing variables.
2. The target variable is of dichotomous categorical data type:
    * <span style="color: #FF0000">LUNG_CANCER</span> (Categorical: YES, Lung Cancer Cases | NO, Non-Lung Cancer Cases)
3. The complete set of 15 predictor variables contain both numeric and categorical data types:   
    * <span style="color: #FF0000">AGE</span> (Numeric: Years)
    * <span style="color: #FF0000">GENDER</span> (Categorical: M, Male | F, Female)
    * <span style="color: #FF0000">SMOKING</span> (Categorical: 1, Absent | 2, Present)
    * <span style="color: #FF0000">YELLOW_FINGERS</span> (Categorical: 1, Absent | 2, Present)
    * <span style="color: #FF0000">ANXIETY</span> (Categorical: 1, Absent | 2, Present)
    * <span style="color: #FF0000">PEER_PRESSURE</span> (Categorical: 1, Absent | 2, Present)
    * <span style="color: #FF0000">CHRONIC_DISEASE</span> (Categorical: 1, Absent | 2, Present)
    * <span style="color: #FF0000">FATIGUE</span> (Categorical: 1, Absent | 2, Present)
    * <span style="color: #FF0000">ALLERGY</span> (Categorical: 1, Absent | 2, Present)
    * <span style="color: #FF0000">WHEEZING</span> (Categorical: 1, Absent | 2, Present)
    * <span style="color: #FF0000">ALCOHOL_CONSUMING </span> (Categorical: 1, Absent | 2, Present)
    * <span style="color: #FF0000">COUGHING</span> (Categorical: 1, Absent | 2, Present)
    * <span style="color: #FF0000">SHORTNESS_OF_BREATH</span> (Categorical: 1, Absent | 2, Present)
    * <span style="color: #FF0000">SWALLOWING_DIFFICULTY</span> (Categorical: 1, Absent | 2, Present)
    * <span style="color: #FF0000">CHEST_PAIN</span> (Categorical: 1, Absent | 2, Present)
4. Exploratory data analysis identified a subset of 10 predictor variables that was significantly associated with the target variable and subsequently used as the final model predictors:   
    * <span style="color: #FF0000">YELLOW_FINGERS</span> (Categorical: 1, Absent | 2, Present)
    * <span style="color: #FF0000">ANXIETY</span> (Categorical: 1, Absent | 2, Present)
    * <span style="color: #FF0000">PEER_PRESSURE</span> (Categorical: 1, Absent | 2, Present)
    * <span style="color: #FF0000">FATIGUE</span> (Categorical: 1, Absent | 2, Present)
    * <span style="color: #FF0000">ALLERGY</span> (Categorical: 1, Absent | 2, Present)
    * <span style="color: #FF0000">WHEEZING</span> (Categorical: 1, Absent | 2, Present)
    * <span style="color: #FF0000">ALCOHOL_CONSUMING</span> (Categorical: 1, Absent | 2, Present)
    * <span style="color: #FF0000">COUGHING</span> (Categorical: 1, Absent | 2, Present)
    * <span style="color: #FF0000">SWALLOWING_DIFFICULTY</span> (Categorical: 1, Absent | 2, Present)
    * <span style="color: #FF0000">CHEST_PAIN</span> (Categorical: 1, Absent | 2, Present)


![cc_data_background.png](c5ce3cb2-c166-43f1-8fb3-f50ef8f190f2.png)

#### 1.1.1.2 Model Background <a class="anchor" id="1.1.1.2"></a>

1. The model development process involved combining different **Dataset Versions** (with preprocessing actions applied to address class imbalance) and **Model Structures** (with ensembling strategy applied to improve generalization). Hyperparameter tuning was conducted using the 5-fold cross-validation method with optimal model performance determined using the **F1 score**.
    * Individual classifier using [logistic regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) developed from the original data.
    * Stacked classifier using [logistic regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) as a meta-learner with a [decision tree model](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html), [random forest model](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#), and [support vector machine model](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) as base learners developed from the original data.
    * Individual classifier using [logistic regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) developed from the SMOTE-upsampled data.
    * Stacked classifier using [logistic regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) as a meta-learner with a [decision tree model](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html), [random forest model](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#), and [support vector machine model](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) as base learners developed from the SMOTE-upsampled data.
    * Individual classifier using [logistic regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) developed from the CNN-downsampled data.
    * Stacked classifier using [logistic regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) as a meta-learner with a [decision tree model](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html), [random forest model](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#), and [support vector machine model](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) as base learners developed from the CNN-downsampled data.
2. The stacked classifier using [logistic regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) as a meta-learner with a [decision tree model](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html), [random forest model](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#), and [support vector machine model](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) as base learners developed from the SMOTE-upsampled data was selected as the final model by demonstrating the best validation **F1 score** with minimal overfitting. 
3. The final model configuration are described as follows:
    * **Base learner**: [decision tree model](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) with optimal hyperparameters:
        * <span style="color: #FF0000">max_depth</span> = 3
        * <span style="color: #FF0000">class_weight</span> = none
        * <span style="color: #FF0000">criterion</span> = entropy
        * <span style="color: #FF0000">min_samples_leaf</span> = 3
        * <span style="color: #FF0000">random_state</span> = 88888888
    * **Base learner**: [random forest model](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#) with optimal hyperparameters:
        * <span style="color: #FF0000">max_depth</span> = 5
        * <span style="color: #FF0000">class_weight</span> = none
        * <span style="color: #FF0000">criterion</span> = entropy
        * <span style="color: #FF0000">max_features</span> = sqrt
        * <span style="color: #FF0000">min_samples_leaf</span> = 3
        * <span style="color: #FF0000">random_state</span> = 88888888
    * **Base learner**: [support vector machine model](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) with optimal hyperparameters:
        * <span style="color: #FF0000">C</span> = 1.00
        * <span style="color: #FF0000">class_weight</span> = none
        * <span style="color: #FF0000">kernel</span> = linear
        * <span style="color: #FF0000">probability</span> = true
        * <span style="color: #FF0000">random_state</span> = 88888888  
    * **Meta-learner**: [logistic regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) with optimal hyperparameters:
        * <span style="color: #FF0000">penalty</span> = none
        * <span style="color: #FF0000">class_weight</span> = none
        * <span style="color: #FF0000">solver</span> = saga
        * <span style="color: #FF0000">max_iter</span> = 500
        * <span style="color: #FF0000">random_state</span> = 88888888
          

![cc_model_background.png](5d81838d-3fa5-41d0-9511-2456a0137a05.png)

#### 1.1.1.3 Deployment Background <a class="anchor" id="1.1.1.3"></a>

1. The prediction model was deployed using a web application hosted at [<mark style="background-color: #CCECFF"><b>Streamlit</b></mark>](https://lung-cancer-diagnosis-probability-estimation.streamlit.app).
2. The user interface input consists of the following:
    * radio buttons to enable binary category selection (Present | Absent) to identify the status of the test case for each of the ten clinical symptoms and behavioral indicators
        * <span style="color: #FF0000">YELLOW_FINGERS</span>
        * <span style="color: #FF0000">ANXIETY</span>
        * <span style="color: #FF0000">PEER_PRESSURE</span>
        * <span style="color: #FF0000">FATIGUE</span>
        * <span style="color: #FF0000">ALLERGY</span>
        * <span style="color: #FF0000">WHEEZING</span>
        * <span style="color: #FF0000">ALCOHOL_CONSUMING </span>
        * <span style="color: #FF0000">COUGHING</span>
        * <span style="color: #FF0000">SWALLOWING_DIFFICULTY</span>
        * <span style="color: #FF0000">CHEST_PAIN</span>
    * action button to:
        * process study population data as baseline
        * process user input as test case
        * render all entries into visualization charts
        * execute all computations, estimations and predictions
        * render test case prediction into logistic probability plot
3. The user interface ouput consists of the following:
    * count plots to:
        * provide a visualization of the proportion of lung cancer categories (Yes | No) by status (Present | Absent) as baseline
        * indicate the entries made from the user input to visually assess the test case characteristics against the study population 
    * logistic curve plot to:
        * provide a visualization of the baseline logistic regression probability curve using the study population with lung cancer categories (Yes | No)
        * indicate the estimated risk index and lung cancer probability of the test case into the baseline logistic regression probability curvee
    * summary table to:
        * present the computed risk index, estimated lung cancer probability and predicted risk category for the test case
          

![cc_deployment_background.png](d0d3486e-750b-46a1-aaf8-c6680d97af35.png)

### 1.1.2 Survival Prediction <a class="anchor" id="1.1.2"></a>

This project implements the **Cox Proportional Hazards Regression**, **Cox Net Survival**, **Survival Tree**, **Random Survival Forest**, and **Gradient Boosted Survival** models as independent base learners using various helpful packages in <mark style="background-color: #CCECFF"><b>Python</b></mark> to estimate the survival probabilities of right-censored survival time and status responses. The resulting predictions derived from the candidate models were evaluated in terms of their discrimination power using the **Harrel's Concordance Index** metric. Penalties including **Ridge Regularization** and **Elastic Net Regularization** were evaluated to impose constraints on the model coefficient updates, as applicable. Additionally, survival probability functions were estimated for model risk-groups and sampled individual cases. 

* The complete model development process was consolidated in this [**Jupyter Notebook**](https://johnpaulinepineda.github.io/Portfolio_Project_55/).
* All associated datasets and code files were stored in this [**GitHub Project Repository**](https://github.com/JohnPaulinePineda/Portfolio_Project_55). 
* The final model was deployed as a prototype application with a web interface via [**Streamlit**](https://heart-failure-survival-probability-estimation.streamlit.app/).


#### 1.1.2.1 Data Background <a class="anchor" id="1.1.2.1"></a>

1. The original dataset comprised rows representing observations and columns representing variables.
2. The target variables contain both numeric and dichotomous categorical data types:
    * <span style="color: #FF0000">DEATH_EVENT</span> (Categorical: 0, Censored | 1, Death)
    * <span style="color: #FF0000">TIME</span> (Numeric: Days)
3. The complete set of 11 predictor variables contain both numeric and categorical data types:   
    * <span style="color: #FF0000">AGE</span> (Numeric: Years)
    * <span style="color: #FF0000">ANAEMIA</span> (Categorical: 0, Absent | 1 Present)
    * <span style="color: #FF0000">CREATININE_PHOSPHOKINASE</span> (Numeric: Percent)
    * <span style="color: #FF0000">DIABETES</span> (Categorical: 0, Absent | 1 Present)
    * <span style="color: #FF0000">EJECTION_FRACTION</span> (Numeric: Percent)
    * <span style="color: #FF0000">HIGH_BLOOD_PRESSURE</span> (Categorical: 0, Absent | 1 Present)
    * <span style="color: #FF0000">PLATELETS</span> (Numeric: kiloplatelets/mL)
    * <span style="color: #FF0000">SERUM_CREATININE</span> (Numeric: mg/dL)
    * <span style="color: #FF0000">SERUM_SODIUM</span> (Numeric: mEq/L)
    * <span style="color: #FF0000">SEX</span> (Categorical: 0, Female | 1, Male)
    * <span style="color: #FF0000">SMOKING</span> (Categorical: 0, Absent | 1 Present)
4. Exploratory data analysis identified a subset of 6 predictor variables that was significantly associated with the target variables and subsequently used as the final model predictors:   
    * <span style="color: #FF0000">AGE</span> (Numeric: Years)
    * <span style="color: #FF0000">ANAEMIA</span> (Categorical: 0, Absent | 1 Present)
    * <span style="color: #FF0000">EJECTION_FRACTION</span> (Numeric: Percent)
    * <span style="color: #FF0000">HIGH_BLOOD_PRESSURE</span> (Categorical: 0, Absent | 1 Present)
    * <span style="color: #FF0000">SERUM_CREATININE</span> (Numeric: mg/dL)
    * <span style="color: #FF0000">SERUM_SODIUM</span> (Numeric: mEq/L)


![sp_data_background.png](e84513d4-bd1f-4a06-aaf3-cc0ebeae8a8d.png)

#### 1.1.2.2 Model Background <a class="anchor" id="1.1.2.2"></a>

1. The model development process involved evaluating different **Model Structures**. Hyperparameter tuning was conducted using the 5-fold cross-validation method with optimal model performance determined using the **Harrel's concordance index**.
    * [Cox proportional hazards regression model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxPHSurvivalAnalysis.html) developed from the original data.
    * [Cox net survival model ](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxnetSurvivalAnalysis.html) developed from the original data.
    * [Survival tree model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.tree.SurvivalTree.html) developed from the original data.
    * [Random survival forest model,](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.ensemble.RandomSurvivalForest.html) developed from the original data.
    * [Gradient boosted survival model ](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.ensemble.GradientBoostingSurvivalAnalysis.html) developed from the original data.
2. The [cox proportional hazards regression model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxPHSurvivalAnalysis.html) developed from the original data was selected as the final model by demonstrating the most stable **Harrel's concordance index** across the different internal and external validation sets. 
3. The final model configuration for the [cox proportional hazards regression model](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxPHSurvivalAnalysis.html) is described as follows:
    * <span style="color: #FF0000">alpha</span> = 10


![sp_model_background.png](bbf692c9-307f-4630-bfd5-8996a2590deb.png)

#### 1.1.2.3 Deployment Background <a class="anchor" id="1.1.2.3"></a>

1. The prediction model was deployed using a web application hosted at [<mark style="background-color: #CCECFF"><b>Streamlit</b></mark>](https://heart-failure-survival-probability-estimation.streamlit.app/).
2. The user interface input consists of the following:
    * range sliders to enable numerical input to measure the characteristics of the test case for certain cardiovascular, hematologic and metabolic markers:
        * <span style="color: #FF0000">AGE</span>
        * <span style="color: #FF0000">EJECTION_FRACTION</span>
        * <span style="color: #FF0000">SERUM_CREATININE</span>
        * <span style="color: #FF0000">SERUM_SODIUM</span>
    * radio buttons to enable binary category selection (Present | Absent) to identify the status of the test case for certain hematologic and cardiovascular markers:
        * <span style="color: #FF0000">ANAEMIA</span>
        * <span style="color: #FF0000">HIGH_BLOOD_PRESSURE</span>
    * action button to:
        * process study population data as baseline
        * process user input as test case
        * render all entries into visualization charts
        * execute all computations, estimations and predictions
        * render test case prediction into the survival probability plot
4. The user interface ouput consists of the following:
    * Kaplan-Meier plots to:
        * provide a baseline visualization of the survival profiles of the various feature categories (Yes | No or High | Low) estimated from the study population given the survival time and event status
        * Indicate the entries made from the user input to visually assess the survival probabilities of the test case characteristics against the study population across all time points
    * survival probability plot to:
        * provide a visualization of the baseline survival probability profile using each observation of the study population given the survival time and event status
        * indicate the heart failure survival probabilities of the test case at different time points
    * summary table to:
        * present the estimated heart failure survival probabilities and predicted risk category for the test case


![sp_deployment_background.png](e4759860-217e-4c31-8a4a-797844305b59.png)

### 1.1.3 Image Classification <a class="anchor" id="1.1.3"></a>

This project focuses on leveraging the **Convolutional Neural Network Model** using various helpful packages in <mark style="background-color: #CCECFF"><b>Python</b></mark> for multiclass image classification by directly learning hierarchical features from raw pixel data. The CNN models were designed to extract low- and high-level features for differentiating between image categories. Various hyperparameters, including the number of layers, filter size, and number of dense layer weights, were systematically evaluated to optimize the model architecture. **Image Augmentation** techniques were employed to increase the diversity of training images and improve the model's ability to generalize. To enhance model performance and robustness, various regularization techniques were explored, including **Dropout**, **Batch Normalization**, and their **Combinations**. These methods helped mitigate overfitting and ensured stable learning. Callback functions such as **Early Stopping**, **Learning Rate Reduction on Performance Plateaus**, and **Model Checkpointing** were implemented to fine-tune the training process, optimize convergence, and prevent overtraining. Model evaluation was conducted using **Precision**, **Recall**, and **F1 Score** metrics to ensure both false positives and false negatives are considered, providing a more balanced view of model classification performance. Post-training, interpretability was emphasized through an advanced visualization technique using **Gradient Class Activation Mapping (Grad-CAM)**, providing insights into the spatial and hierarchical features that influenced the model's predictions, offering a deeper understanding of the decision-making process. 

* The complete model development process was consolidated in this [**Jupyter Notebook**](https://johnpaulinepineda.github.io/Portfolio_Project_56/).
* All associated datasets and code files were stored in this [**GitHub Project Repository**](https://github.com/JohnPaulinePineda/Portfolio_Project_56). 
* The final model was deployed as a prototype application with a web interface via [**Streamlit**](https://brain-mri-image-classification.streamlit.app/).


#### 1.1.3.1 Data Background <a class="anchor" id="1.1.3.1"></a>

![ic_data_background.png](174e5369-84c0-413f-92c6-fdf18cfdaa5e.png)

#### 1.1.3.2 Model Background <a class="anchor" id="1.1.3.2"></a>

![ic_model_background.png](37b6c7a7-9454-4849-951c-8e42c91e9555.png)

#### 1.1.3.3 Deployment Background <a class="anchor" id="1.1.3.3"></a>

![ic_deployment_background.png](d56f11c3-97b4-4211-980d-b599beb0a7c7.png)

## 1.2. Application Programming Interface (API) Development Using the FastAPI Framework <a class="anchor" id="1.2"></a>

### 1.2.1 Categorical Classification <a class="anchor" id="1.2.1"></a>

#### 1.2.1.1 API Building <a class="anchor" id="1.2.1.1"></a>

![cc_fastapi_code.png](0c035228-e098-4344-864f-a2b5502a927e.png)

#### 1.2.1.2 API Testing <a class="anchor" id="1.2.1.2"></a>

![cc_fastapi_activation.png](1ef82129-c142-4aad-a439-7109bea39923.png)

![cc_fastapi_documentation.png](0013a9f8-30b2-4dc7-9e31-62230ecb93e7.png)

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
CC_FASTAPI_BASE_URL = "http://127.0.0.1:8000"

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
response = requests.get(f"{CC_FASTAPI_BASE_URL}/")
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
response = requests.post(f"{CC_FASTAPI_BASE_URL}/predict-individual-logit-probability-class", json=individual_test_case)
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
response = requests.post(f"{CC_FASTAPI_BASE_URL}/predict-list-logit-probability-class", json=batch_test_case)
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
response = requests.post(f"{CC_FASTAPI_BASE_URL}/predict-individual-logit-probability-class", json=malformed_test_case)
if response.status_code == 200:
    display("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
    
```

    Error: 422 {"detail":[{"type":"missing","loc":["body","features_individual"],"msg":"Field required","input":{"features":[1,0,1]}}]}
    

### 1.2.2 Survival Prediction <a class="anchor" id="1.2.2"></a>

#### 1.2.2.1 API Building <a class="anchor" id="1.2.2.1"></a>

![sp_fastapi_code.png](1f99067b-b0a5-4dca-9827-cf2686fc9108.png)

#### 1.2.2.2 API Testing <a class="anchor" id="1.2.2.2"></a>

![sp_fastapi_activation.png](2cc97028-ce4a-4cf0-948f-840b22cdbd0e.png)

![sp_fastapi_documentation.png](30d5a32f-11e4-4175-b7f4-e6ac2358346e.png)

![sp_fastapi_endpoints.png](459dc1e7-32ad-4854-98ac-f12992cce1ca.png)


```python
##################################
# Loading Python Libraries
##################################
import requests
import json
import pandas as pd
import base64
from IPython.display import display
from PIL import Image

```


```python
##################################
# Defining the base URL of the API
# for the survival prediction model
##################################
SP_FASTAPI_BASE_URL = "http://127.0.0.1:8001"

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
response = requests.get(f"{SP_FASTAPI_BASE_URL}/")
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
response = requests.post(f"{SP_FASTAPI_BASE_URL}/compute-individual-coxph-survival-probability-class/", json=single_test_case)
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
response = requests.post(f"{SP_FASTAPI_BASE_URL}/compute-list-coxph-survival-profile/", json=train_list)
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
response = requests.post(f"{SP_FASTAPI_BASE_URL}/bin-numeric-model-feature/", json=bin_request)
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
response = requests.post(f"{SP_FASTAPI_BASE_URL}/plot-kaplan-meier/", json=km_request)
if response.status_code == 200:
    plot_data = response.json()["plot"]
    # Decoding and displaying the plot
    img = base64.b64decode(plot_data)
    with open("kaplan_meier_plot.png", "wb") as f:
        f.write(img)
        display(Image.open("kaplan_meier_plot.png"))
else:
    print("Error:", response.status_code, response.text)
    
```


    
![png](output_59_0.png)
    



```python
##################################
# Sending a POST endpoint request
# using malformed data to evaluate
# the API's error handling function
##################################
malformed_test_case = {"features": [43, 0, 75, 1, 0.75]}
response = requests.post(f"{SP_FASTAPI_BASE_URL}/compute-individual-coxph-survival-probability-class", json=malformed_test_case)
if response.status_code == 200:
    display("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
    
```

    Error: 422 {"detail":[{"type":"missing","loc":["body","features_individual"],"msg":"Field required","input":{"features":[43,0,75,1,0.75]}}]}
    

### 1.2.3 Image Classification <a class="anchor" id="1.2.3"></a>

#### 1.2.3.1 API Building <a class="anchor" id="1.2.3.1"></a>

![ic_fastapi_code.png](64d15f91-4423-4c22-8e59-97fa8a3a5226.png)

#### 1.2.3.2 API Testing <a class="anchor" id="1.2.3.2"></a>

![ic_fastapi_activation.png](03e075dd-63b0-4b41-8479-2b03412459dd.png)

![ic_fastapi_documentation.png](a062362a-84dd-4446-aa30-92dc5ba270e9.png)

![ic_fastapi_endpoints.png](858dcb27-be28-4eb5-bafe-7cd111c7a19b.png)


```python
##################################
# Loading Python Libraries
##################################
import requests
import json
import base64
from IPython.display import display
from PIL import Image
import matplotlib.pyplot as plt
import io
from tensorflow.keras.utils import load_img
import os
import mimetypes

```


```python
##################################
# Defining the base URL of the API
# for the image classification model
##################################
IC_FASTAPI_BASE_URL = "http://127.0.0.1:8002"

```


```python
##################################
# Defining the file path for an individual test image
##################################
IMAGES_PATH = r"image_classification_study\images"
image_path = (os.path.join("..",IMAGES_PATH, "test_image.jpg"))

```


```python
##################################
# Automatically determining the filename and content type
##################################
image_path_filename = os.path.basename(image_path)
image_path_content_type, _ = mimetypes.guess_type(image_path)

```


```python
##################################
# Visualizing the individual test image
##################################
try:
    image = Image.open(image_path)
    print(f"Image File Path: {image_path}")
    print(f"Image Format: {image.format}")
    print(f"Image Size: {image.size}")
    print(f"Image Mode: {image.mode}") 
except Exception as e:
    print(f"Error loading image: {e}")
plt.imshow(image)
plt.axis('off') 
plt.title("Test Image")
plt.show()
```

    Image File Path: ..\image_classification_study\images\test_image.jpg
    Image Format: JPEG
    Image Size: (215, 234)
    Image Mode: RGB
    


    
![png](output_72_1.png)
    



```python
##################################
# Generating a GET endpoint request for
# for validating API service connection
##################################
response = requests.get(f"{IC_FASTAPI_BASE_URL}/")
if response.status_code == 200:
    display("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
    
```


    'Response:'



    {'message': 'Welcome to the Image Classification API!'}



```python
##################################
# Sending a POST endpoint request for
# ensuring that the file upload mechanism is working
# by returning the the file metadata
##################################
with open(image_path, "rb") as file:
    files = {"file": (image_path_filename, file, image_path_content_type)}
    response = requests.post(f"{IC_FASTAPI_BASE_URL}/test-file-upload/", files=files)

    if response.status_code == 200:
        result = response.json()
        print("File Upload Test Result:")
        print(f"Filename: {result['filename']}")
        print(f"Content Type: {result['content_type']}")
        print(f"Size: {result['size']} bytes")
    else:
        print(f"Error: {response.status_code} - {response.text}")
        
```

    File Upload Test Result:
    Filename: test_image.jpg
    Content Type: image/jpeg
    Size: 12103 bytes
    


```python
##################################
# Sending a POST endpoint request for
# predicting the image category and
# estimating class probabilities
# of an individual test image
##################################
with open(image_path, "rb") as file:
    files = {"file": ("image.jpg", file, "image/jpeg")}
    response = requests.post(f"{IC_FASTAPI_BASE_URL}/predict-image-category-class-probability/", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print("Prediction Result:")
        print(f"Predicted Class: {result['predicted_class']}")
        print("Probabilities:")
        for cls, prob in result["probabilities"].items():
            print(f"{cls}: {prob:.5f}%")
    else:
        print(f"Error: {response.status_code} - {response.text}")
            
```

    Prediction Result:
    Predicted Class: Meningioma
    Probabilities:
    No Tumor: 12.94989%
    Glioma: 0.02788%
    Meningioma: 87.02222%
    Pituitary: 0.00002%
    


```python
##################################
# Sending a POST endpoint request for
# formulating the gradient class activation map
# from the output of the first to third convolutional layers and
# and superimposing on the actual image
##################################
with open(image_path, "rb") as file:
    files = {"file": ("image.jpg", file, "image/jpeg")}
    response = requests.post(f"{IC_FASTAPI_BASE_URL}/visualize-image-gradcam/", files=files)
    
    if response.status_code == 200:
        plot_data = response.json()["plot"]
        # Decoding and displaying the plot
        img = base64.b64decode(plot_data)
        with open("image_gradcam_plot.png", "wb") as f:
            f.write(img)
            display(Image.open("image_gradcam_plot.png"))
    else:
        print(f"Error: {response.status_code} - {response.text}")
        
```


    
![png](output_76_0.png)
    



```python
##################################
# Defining the file path for an individual test image
##################################
IMAGES_PATH = r"image_classification_study\images"
malformed_image_path = (os.path.join("..",IMAGES_PATH, "test_image.png"))

```


```python
##################################
# Automatically determining the filename and content type
##################################
malformed_image_path_filename = os.path.basename(image_path)
malformed_image_path_content_type, _ = mimetypes.guess_type(malformed_image_path)

```


```python
##################################
# Sending a POST endpoint request
# using malformed data to evaluate
# the API's error handling function
##################################
with open(malformed_image_path, "rb") as file:
    files = {"file": (malformed_image_path_filename, file, malformed_image_path_content_type)}
    response = requests.post(f"{IC_FASTAPI_BASE_URL}/test-file-upload/", files=files)

    if response.status_code == 200:
        result = response.json()
        print("File Upload Test Result:")
        print(f"Filename: {result['filename']}")
        print(f"Content Type: {result['content_type']}")
        print(f"Size: {result['size']} bytes")
    else:
        print(f"Error: {response.status_code} - {response.text}")
        
```

    Error: 400 - {"detail":"File must be a JPEG image"}
    

## 1.3. Application Programming Interface (API) Development Using the Flask Framework <a class="anchor" id="1.3"></a>

### 1.3.1 Categorical Classification <a class="anchor" id="1.3.1"></a>

#### 1.3.1.1 API Building <a class="anchor" id="1.3.1.1"></a>

![cc_flaskapi_code.png](10723341-eac5-413e-bd18-2e5ca0e740a5.png)

#### 1.3.1.2 API Testing <a class="anchor" id="1.3.1.2"></a>

![cc_flaskapi_activation.png](2c7149d3-ef57-4f6c-b05a-6848369ced82.png)

![cc_flaskapi_documentation.png](58c27091-d998-4dcf-a56a-13e07b5a4589.png)

![cc_flaskapi_endpoints.png](87ff7dad-203d-451c-b981-2cfff81dd15d.png)


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
CC_FLASKAPI_BASE_URL = "http://127.0.0.1:5000/"

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
# validating API service connection
##################################
response = requests.get(f"{CC_FLASKAPI_BASE_URL}/")
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
response = requests.post(f"{CC_FLASKAPI_BASE_URL}/predict-individual-logit-probability-class", json=individual_test_case)
if response.status_code == 200:
    print("Individual Test Case Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
    
```

    Individual Test Case Response: {'logit': -1.2117837409390746, 'probability': 0.22938559072691203, 'risk_class': 'Low-Risk'}
    


```python
##################################
# Sending a POST endpoint request for
# computing the risk index,
# estimating the lung cancer probability,
# and predicting the risk category
# of a list of train cases
##################################
response = requests.post(f"{CC_FLASKAPI_BASE_URL}/predict-list-logit-probability-class", json=batch_test_case)
if response.status_code == 200:
    print("Batch Test Case Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
    
```

    Batch Test Case Response: {'logit': [-1.2117837409390746, 3.4784950973590973], 'logit_sorted': [-1.2117837409390746, 3.4784950973590973], 'probability': [0.22938559072691203, 0.9700696569701589], 'probability_sorted': [0.22938559072691203, 0.9700696569701589]}
    


```python
##################################
# Sending a POST endpoint request
# using malformed data to evaluate
# the API's error handling function
##################################
malformed_test_case = {"features": [1, 0, 1]}
response = requests.post(f"{CC_FLASKAPI_BASE_URL}/predict-individual-logit-probability-class", json=malformed_test_case)
if response.status_code == 200:
    print("Malformed Test Case Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
    
```

    Error: 400 {"error":"'features_individual'"}
    
    

### 1.3.2 Survival Prediction <a class="anchor" id="1.3.2"></a>

#### 1.3.2.1 API Building <a class="anchor" id="1.3.2.1"></a>

![sp_flaskapi_code.png](98b3e8a0-d312-4ee3-a32f-b432cc7986ed.png)

#### 1.3.2.2 API Testing <a class="anchor" id="1.3.2.2"></a>

![sp_flaskapi_activation.png](72d36408-7169-48b5-af94-534e0f3527bf.png)

![sp_flaskapi_documentation.png](9ea041e7-f9ff-4879-9eb9-7c6cd10ab257.png)

![sp_flaskapi_endpoints.png](3835d047-73a3-463a-9d36-5eca9fb587c6.png)


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
# for the survival prediction model
##################################
SP_FLASKAPI_BASE_URL = "http://127.0.0.1:5001"

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
        [70, 1, 20, 1, 0.75, 100]
    ]
}

```


```python
##################################
# Defining the input values for a batch of cases for binning request
##################################
bin_request = {
    "X_original_list": [
        {"AGE": -0.10, "EJECTION_FRACTION": -0.10, "SERUM_CREATININE": -0.10, "SERUM_SODIUM": -0.10},
        {"AGE": 0.20, "EJECTION_FRACTION": 0.20, "SERUM_CREATININE": 0.20, "SERUM_SODIUM": 0.20},
        {"AGE": 0.90, "EJECTION_FRACTION": 0.90, "SERUM_CREATININE": 0.90, "SERUM_SODIUM": 0.90}
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
response = requests.get(f"{SP_FLASKAPI_BASE_URL}/")
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
response = requests.post(f"{SP_FLASKAPI_BASE_URL}/compute-individual-coxph-survival-probability-class/", json=single_test_case)
if response.status_code == 200:
    display("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
    
```


    'Response:'



    {'risk_category': 'Low-Risk',
     'survival_function': [0.9973812917524568,
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
     'survival_probabilities': [90.97550958520674,
      87.64677174183527,
      84.59534502216806,
      78.48178617845467,
      70.70184115456696],
     'survival_time': [50, 100, 150, 200, 250]}



```python
##################################
# Sending a POST endpoint request for
# generating the heart failure survival profile and
# estimating the heart failure survival probabilities
# of a list of train cases
##################################
response = requests.post(f"{SP_FLASKAPI_BASE_URL}/compute-list-coxph-survival-profile/", json=train_list)
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
response = requests.post(f"{SP_FLASKAPI_BASE_URL}/bin-numeric-model-feature/", json=bin_request)
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
response = requests.post(f"{SP_FLASKAPI_BASE_URL}/plot-kaplan-meier/", json=km_request)
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


    
![png](output_113_0.png)
    



```python
##################################
# Sending a POST endpoint request
# using malformed data to evaluate
# the API's error handling function
##################################
malformed_test_case = {"features": [43, 0, 75, 1, 0.75]}
response = requests.post(f"{SP_FLASKAPI_BASE_URL}/compute-individual-coxph-survival-probability-class/", json=malformed_test_case)
if response.status_code == 200:
    display("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)
    
```

    Error: 400 {"error":"Missing 'features_individual' in request"}
    
    

### 1.3.3 Image Classification <a class="anchor" id="1.3.3"></a>

#### 1.3.3.1 API Building <a class="anchor" id="1.3.3.1"></a>

![ic_flaskapi_code.png](666bd6af-8d49-4666-a639-5b817bad1fe7.png)

#### 1.3.3.2 API Testing <a class="anchor" id="1.3.3.2"></a>

![ic_flaskapi_activation.png](6353a838-016b-4d32-bf0e-0ede8dcb4177.png)

![ic_flaskapi_documentation.png](b03dbdbb-2a57-4c85-8943-ec36a66a0f6b.png)

![ic_flaskapi_endpoints.png](4d1ee757-a5ef-4771-821d-ee1be9f6c409.png)


```python
##################################
# Loading Python Libraries
##################################
import requests
import json
import base64
from IPython.display import display
from PIL import Image
import matplotlib.pyplot as plt
import io
from tensorflow.keras.utils import load_img
import os
import mimetypes

```


```python
##################################
# Defining the base URL of the API
# for the image classification model
##################################
IC_FLASKAPI_BASE_URL = "http://127.0.0.1:5002"

```


```python
##################################
# Defining the file path for an individual test image
##################################
IMAGES_PATH = r"image_classification_study\images"
image_path = (os.path.join("..",IMAGES_PATH, "test_image.jpg"))

```


```python
##################################
# Automatically determining the filename and content type
##################################
image_path_filename = os.path.basename(image_path)
image_path_content_type, _ = mimetypes.guess_type(image_path)

```


```python
##################################
# Visualizing the individual test image
##################################
try:
    image = Image.open(image_path)
    print(f"Image File Path: {image_path}")
    print(f"Image Format: {image.format}")
    print(f"Image Size: {image.size}")
    print(f"Image Mode: {image.mode}") 
except Exception as e:
    print(f"Error loading image: {e}")
plt.imshow(image)
plt.axis('off') 
plt.title("Test Image")
plt.show()

```

    Image File Path: ..\image_classification_study\images\test_image.jpg
    Image Format: JPEG
    Image Size: (215, 234)
    Image Mode: RGB
    


    
![png](output_126_1.png)
    



```python
##################################
# Generating a GET endpoint request for
# for validating API service connection
##################################
response = requests.get(f"{IC_FLASKAPI_BASE_URL}/")
if response.status_code == 200:
    display("Response:", response.json())
else:
    print("Error:", response.status_code, response.text)

```


    'Response:'



    {'message': 'Welcome to the Image Classification API!'}



```python
##################################
# Sending a POST endpoint request for
# ensuring that the file upload mechanism is working
# by returning the the file metadata
##################################
with open(image_path, "rb") as file:
    files = {"file": (image_path_filename, file, image_path_content_type)}
    response = requests.post(f"{IC_FLASKAPI_BASE_URL}/test-file-upload/", files=files)

    if response.status_code == 200:
        result = response.json()
        print("File Upload Test Result:")
        print(f"Filename: {result['filename']}")
        print(f"Content Type: {result['content_type']}")
        print(f"Size: {result['size']} bytes")
    else:
        print(f"Error: {response.status_code} - {response.text}")

```

    File Upload Test Result:
    Filename: test_image.jpg
    Content Type: image/jpeg
    Size: 12103 bytes
    


```python
##################################
# Sending a POST endpoint request for
# predicting the image category and
# estimating class probabilities
# of an individual test image
##################################
with open(image_path, "rb") as file:
    files = {"file": ("image.jpg", file, "image/jpeg")}
    response = requests.post(f"{IC_FLASKAPI_BASE_URL}/predict-image-category-class-probability/", files=files)
    
    if response.status_code == 200:
        result = response.json()
        print("Prediction Result:")
        print(f"Predicted Class: {result['predicted_class']}")
        print("Probabilities:")
        for cls, prob in result["probabilities"].items():
            print(f"{cls}: {prob:.5f}%")
    else:
        print(f"Error: {response.status_code} - {response.text}")

```

    Prediction Result:
    Predicted Class: Meningioma
    Probabilities:
    Glioma: 0.02788%
    Meningioma: 87.02222%
    No Tumor: 12.94989%
    Pituitary: 0.00002%
    


```python
##################################
# Sending a POST endpoint request for
# formulating the gradient class activation map
# from the output of the first to third convolutional layers and
# and superimposing on the actual image
##################################
with open(image_path, "rb") as file:
    files = {"file": ("image.jpg", file, "image/jpeg")}
    response = requests.post(f"{IC_FLASKAPI_BASE_URL}/visualize-image-gradcam/", files=files)
    
    if response.status_code == 200:
        plot_data = response.json()["plot"]
        # Decoding and displaying the plot
        img = base64.b64decode(plot_data)
        with open("image_gradcam_plot.png", "wb") as f:
            f.write(img)
            display(Image.open("image_gradcam_plot.png"))
    else:
        print(f"Error: {response.status_code} - {response.text}")

```


    
![png](output_130_0.png)
    



```python
##################################
# Defining the file path for an individual test image
##################################
IMAGES_PATH = r"image_classification_study\images"
malformed_image_path = (os.path.join("..",IMAGES_PATH, "test_image.png"))

```


```python
##################################
# Automatically determining the filename and content type
##################################
malformed_image_path_filename = os.path.basename(image_path)
malformed_image_path_content_type, _ = mimetypes.guess_type(malformed_image_path)

```


```python
##################################
# Sending a POST endpoint request
# using malformed data to evaluate
# the API's error handling function
##################################
malformed_image_path = (os.path.join("..",IMAGES_PATH, "test_image.png"))
with open(malformed_image_path, "rb") as file:
    files = {"file": (malformed_image_path_filename, file, malformed_image_path_content_type)}
    response = requests.post(f"{IC_FLASKAPI_BASE_URL}/test-file-upload/", files=files)

    if response.status_code == 200:
        result = response.json()
        print("File Upload Test Result:")
        print(f"Filename: {result['filename']}")
        print(f"Content Type: {result['content_type']}")
        print(f"Size: {result['size']} bytes")
    else:
        print(f"Error: {response.status_code} - {response.text}")
        
```

    Error: 400 - {"error":"File must be a JPEG image"}
    
    

## 1.4. Consolidated Findings <a class="anchor" id="1.4"></a>

# 2. Summary <a class="anchor" id="Summary"></a>

# 3. References <a class="anchor" id="References"></a>

* **[Book]** [Building Machine Learning Powered Applications: Going From Idea to Product](https://www.oreilly.com/library/view/building-machine-learning/9781492045106/) by Emmanuel Ameisen
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
* **[Article]** [Deploy a Machine Learning Model using Flask: Step-By-Step](https://codefather.tech/blog/deploy-machine-learning-model-flask/) by Claudio Sabato (CodeFather.Tech)
* **[Article]** [How to Deploy a Machine Learning Model using Flask?](https://datadance.ai/machine-learning/how-to-deploy-a-machine-learning-model-using-flask/) by DataDance.AI Team (DataDance.AI)
* **[Article]** [A Comprehensive Guide on Deploying Machine Learning Models with Flask](https://machinelearningmodels.org/a-comprehensive-guide-on-deploying-machine-learning-models-with-flask/) by MachineLearningModels.Org Team (MachineLearningModels.Org)
* **[Article]** [How to Deploy Machine Learning Models with Flask and Docker](https://python.plainenglish.io/how-to-deploy-machine-learning-models-with-flask-and-docker-3c4d6116e809) by Usama Malik (Medium)
* **[Article]** [Deploying Machine Learning Models with Flask: A Step-by-Step Guide](https://medium.com/@sukmahanifah/deploying-machine-learning-models-with-flask-a-step-by-step-guide-cd22967c1f66) by Sukma Hanifa (Medium)
* **[Article]** [Machine Learning Model Deployment on Heroku Using Flask](https://towardsdatascience.com/machine-learning-model-deployment-on-heroku-using-flask-467acb4a34da) by Charu Makhijani (Medium)
* **[Article]** [Complete Guide on Model Deployment with Flask and Heroku](https://towardsdatascience.com/complete-guide-on-model-deployment-with-flask-and-heroku-98c87554a6b9) by Tarek Ghanoum (Medium)
* **[Article]** [Turning Machine Learning Models into APIs in Python](https://www.datacamp.com/tutorial/machine-learning-models-api-python) by Sayak Paul (DataCamp)
* **[Article]** [Machine Learning, Pipelines, Deployment and MLOps Tutorial](https://www.datacamp.com/tutorial/tutorial-machine-learning-pipelines-mlops-deployment) by Moez Ali (DataCamp)
* **[Video Tutorial]** [Machine Learning Models Deployment with Flask and Docker](https://www.youtube.com/watch?v=KTd2a1QKlwo) by Data Science Dojo (YouTube)
* **[Video Tutorial]** [Deploy Machine Learning Model Flask](https://www.youtube.com/watch?v=MxJnR1DMmsY) by Stats Wire (YouTube)
* **[Video Tutorial]** [Deploy Machine Learning Models with Flask | Using Render to host API and Get URL :Step-By-Step Guide](https://www.youtube.com/watch?v=LBlvuUaIg58) by Prachet Shah (YouTube)
* **[Video Tutorial]** [Deploy Machine Learning Model using Flask](https://www.youtube.com/watch?app=desktop&v=UbCWoMf80PY&t=597s) by Krish Naik (YouTube)
* **[Video Tutorial]** [Deploy Your ML Model Using Flask Framework](https://www.youtube.com/watch?v=PtyyVGsE-u0) by MSFTImagine (YouTube)
* **[Video Tutorial]** [Build a Machine Learning App From Scratch with Flask & Docker](https://www.youtube.com/watch?v=S--SD4QbGps) by Patrick Loeber (YouTube)
* **[Video Tutorial]** [Deploying a Machine Learning Model to a Web with Flask and Python Anywhere](https://www.youtube.com/watch?v=3w3vBu2WMvk) by Prof. Phd. Manoel Gadi (YouTube)
* **[Video Tutorial]** [End To End Machine Learning Project With Deployment Using Flask](https://www.youtube.com/watch?v=RnOU2bumBPE) by Data Science Diaries (YouTube)
* **[Video Tutorial]** [Publish ML Model as API or Web with Python Flask](https://www.youtube.com/watch?v=_cLbGKKrggs) by Python ML Daily (YouTube)
* **[Video Tutorial]** [Deploy a Machine Learning Model using Flask API to Heroku](https://www.youtube.com/watch?v=Q_Z5kzKpofk) by Jackson Yuan (YouTube)
* **[Video Tutorial]** [Deploying Machine Learning Model with FlaskAPI - CI/CD for ML Series](https://www.youtube.com/watch?v=vxF5uEoL1C4) by Anthony Soronnadi (YouTube)
* **[Video Tutorial]** [Deploy ML model as Webservice | ML model deployment | Machine Learning | Data Magic](https://www.youtube.com/watch?v=3U1T8cLL-1M) by Data Magic (YouTube)
* **[Video Tutorial]** [Deploying Machine Learning Model Using Flask](https://www.youtube.com/watch?v=ng15EVDrL28) by DataMites (YouTube)
* **[Video Tutorial]** [ML Model Deployment With Flask On Heroku | How To Deploy Machine Learning Model With Flask | Edureka](https://www.youtube.com/watch?v=pMIwu5FwJ78) by Edureka (YouTube)
* **[Video Tutorial]** [ML Model Deployment with Flask | Machine Learning & Data Science](https://www.youtube.com/watch?v=Od0gS3Qeges) by Dan Bochman (YouTube)
* **[Video Tutorial]** [How to Deploy ML Solutions with FastAPI, Docker, & AWS](https://www.youtube.com/watch?v=pJ_nCklQ65w) by Shaw Talebi (YouTube)
* **[Video Tutorial]** [Deploy ML models with FastAPI, Docker, and Heroku | Tutorial](https://www.youtube.com/watch?v=h5wLuVDr0oc) by AssemblyAI (YouTube)
* **[Video Tutorial]** [Machine Learning Model Deployment Using FastAPI](https://www.youtube.com/watch?v=0s-oat69UqU) by TheOyinbooke (YouTube)
* **[Video Tutorial]** [Creating APIs For Machine Learning Models with FastAPI](https://www.youtube.com/watch?v=5PgqzVG9SCk) by NeuralNine (YouTube)
* **[Video Tutorial]** [How To Deploy Machine Learning Models Using FastAPI-Deployment Of ML Models As API’s](https://www.youtube.com/watch?v=b5F667g1yCk) by Krish Naik (YouTube)
* **[Video Tutorial]** [Machine Learning Model with FastAPI, Streamlit and Docker](https://www.youtube.com/watch?v=cCsnmxXxWaM) by CodeTricks (YouTube)
* **[Video Tutorial]** [FastAPI Machine Learning Model Deployment | Python | FastAPI](https://www.youtube.com/watch?v=DUhzTi3w5KA) by Stats Wire (YouTube)
* **[Video Tutorial]** [Deploying Machine Learning Models - Full Guide](https://www.youtube.com/watch?v=oyYur3uVl4w) by NeuralNine (YouTube)
* **[Video Tutorial]** [Model Deployment FAST API - Docker | Machine Learning Model Deployment pipeline | FastAPI VS Flask](https://www.youtube.com/watch?v=YvvOuY9L_Yw) by 360DigiTMG (YouTube)
* **[Video Tutorial]** [Build an AI app with FastAPI and Docker - Coding Tutorial with Tips](https://www.youtube.com/watch?v=iqrS7Q174Ac) by Patrick Loeber (YouTube)
* **[Video Tutorial]** [Create a Deep Learning API with Python and FastAPI](https://www.youtube.com/watch?v=NrarIs9n24I) by DataQuest (YouTube)
* **[Video Tutorial]** [Fast API Machine Learning Web App Tutorial + Deployment on Heroku](https://www.youtube.com/watch?v=LSXU3dEDg9A) by Greg Hogg (YouTube)
* **[Course]** [Deeplearning.AI Machine Learning in Production](https://www.coursera.org/learn/introduction-to-machine-learning-in-production) by DeepLearning.AI Team (Coursera)
* **[Course]** [IBM AI Workflow: Enterprise Model Deployment](https://www.coursera.org/learn/ibm-ai-workflow-machine-learning-model-deployment) by IBM Team (Coursera)
* **[Course]** [DataCamp Machine Learning Engineer Track](https://app.datacamp.com/learn/career-tracks/machine-learning-engineer) by DataCamp Team (DataCamp)
* **[Course]** [DataCamp Designing Machine Learning Workflows in Python](https://app.datacamp.com/learn/courses/designing-machine-learning-workflows-in-python) by DataCamp Team (DataCamp)
* **[Course]** [DataCamp Building APIs in Python](https://app.datacamp.com/learn/skill-tracks/building-apis-in-python) by DataCamp Team (DataCamp)



```python
from IPython.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>"))
```


<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>

