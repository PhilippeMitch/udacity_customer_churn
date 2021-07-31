# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

Customer Churn is a condition in which consumers do not continue or leave the services offered by an industry. If not handled, it will lead to loss of revenue and even loss. To overcome this, we need to know which consumers have the potential to churn, so that we can focus on grabbing those consumers and know the constraints in the services the industry provides.

This project seeks to answer the customer churn that is happening in the banking industry. The dataset used in this project is obtained from [Kaggle](https://www.kaggle.com/sakshigoyal7/credit-card-customers/code). 

Workflows used to answer these problems include:

1. import data
2. Exploratory Data Analysis :  Explore data features, such as the distribution of values in data and analyze correlations between features in data
3. Feature Engineering : These stages consist of: one-hot-encoding categorical features, standardization (homogenizing the range of numerical values), and data splitting
4. Modeling : This stage consists of training two models (logistic regression and random forest), hyper-parameter tuning, model evaluation (ROC-Curve and confussion matrices)

## Project Structure

The structure of this project directory tree is displayed as follows:

```
.
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py
├── data
│   └── bank_data.csv
├── Guide.ipynb
├── images
│   ├── eda
│   │   ├── Churn.png
│   │   ├── Customer_Age.png
│   │   ├── Gender.png
│   │   ├── heatmap.png
│   │   ├── Marital_Status.png
│   │   └── Total_Trans_Amt.png
│   └── results
│       ├── feature_importances.png
│       ├── logistic_results.png
│       ├── rf_results.png
│       └── roc_curve_result.png
├── logs
│   └── churn_library.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── __pycache__
│   ├── churn_library2.cpython-36.pyc
│   └── churn_library.cpython-36.pyc
├── README.md
└── requirements.txt
```

there are 4 main folders in the repository:

1. `data` :location of data saved in `csv` format
2. `images` : in this folder there are two sub-folders, namely: `eda` and `results`. The `eda` sub-folder is used to store the results of visualizations of numerical, categorical, and heatmap data distributions of correlation values between variables. The `results` sub-folder stores visualization and evaluation of models, such as: ROC-curve, feature importance, confussion matrix cards.
3. `logs` : This folder stores the logs of function test results on the churn_library.py file
4. `models` : This folder is used to store model objects with the ``.pkl` extension.

Other important files in this repository include:

1. `churn_library.py` : This file is used to write down modeling functions, such as: reading data, eda, feature engineering (one-hot-encoding, standardization, and data splitting), as well as modeling with logistic regression and random forest (training, prediction, and evaluation)
2. `churn_script_logging_and_tests.py` : This file is used to test and logging workflow modeling written in the file `churn_library.py`.
3. `requirements.txt` : This file contains the libraries used in modeling and versions of those libraries.
4. `churn_notebook.ipynb` : A file containing a modeling workflow prototype before it is converted into a function in a file `churn_library.py`

## Running Files

### How to clone the project

to clone this project, make sure you have git installed in your computer. If you have already installed git, run this command

```
git clone https://github.com/mohrosidi/udacity_customer_churn.git
```

### Dependencies

Here is a list of libraries used in this repository:

```
autopep8==1.5.7
joblib==0.11
matplotlib==2.1.0
numpy==1.12.1
pandas==0.23.3
pylint==2.9.6
scikit-learn==0.22
seaborn==0.8.1
```

To be able to run this project, you must install python library using the following command:

```
pip install -r requirements.txt
```

### Modeling

To run the workflow, simply run the `churn_library.py` in your terminal using command bellow:

```
ipython churn_library.py
```

### Testing and Logging

In other conditions, suppose you want to change the configuration of the modeling workflow, such as: changing the path of the data location, adding other models, adding feature engineering stages. You can change it in `churn_library.py` files. To test if your changes are going well, you need to do testing and logging.

To do testing and logging, you need to change a number of configurations in the `churn_script_logging_and_tests.py` file, such as: target column name, categorical column name list, data location, etc. After that, run the following command in the terminal to perform testing and loggingAfter that, run the following command in the terminal to perform testing and logging:

```
ipython churn_script_logging_and_tests.py
```

### Cleaning up your code

Make sure the code you create complies with `PEP 8` rules. To check it automatically, run pylint on the terminal.

```
pylint churn_library.py
pylint churn_script_logging_and_tests.py
```

`Pylint` will provide recommendations for improvements in your code. A good code is a code that has a score close to 10.

To make repairs automatically, you can use autopep8.

```
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_library.py
```
