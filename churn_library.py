'''
Collection of function to do Customer Churn analysis and modeling

Author: Mohammad Rosidi
Date  : July 2021
'''

# import libraries
import os
# from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
os.environ['QT_QPA_PLATFORM']='offscreen'

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
    output:
             None
    '''
    #Random forest report
    plt.rc('figure', figsize=(5, 5))
    #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'),
             {'fontsize': 10},
             fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test,y_test_preds_rf)),
             {'fontsize':10},
             fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'),
             {'fontsize': 10},
             fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    plt.savefig('./images/results/rf_results.png')
    plt.close()
    # Logistic regression report
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10},
             fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10},
             fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'),
             {'fontsize': 10},
             fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test,
                                                  y_test_preds_lr)),
             {'fontsize': 10},
             fontproperties = 'monospace')
    plt.axis('off')
    plt.savefig('./images/results/logistic_results.png')
    plt.close()

def feature_importance_plot(model, x_train, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_train: pandas dataframe of x train values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [x_train.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(20,5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(x_train.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(x_train.shape[1]), names, rotation=90)
    plt.savefig('{0}/feature_importances.png'.format(output_pth))
    plt.close()

class Model:
    '''
    Class model for classification workflow:
    Instances:
        import_data: read data from path
        perform_eda: create eda plot (distribution plot)
            and heatmap
        encoder_helper: helper function to one-hot-encoding
            categorical columns
        perform_feature_engineering: standardize numerical columns
            and data splitting
        train_models: function to modeling, evaluating the models,
            and create predictions
    '''
    def __init__(self):
        '''
        Class initialization
        '''
        self.data_frame = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def import_data(self, pth):
        '''
        returns dataframe for the csv found at pth
        input:
            pth: a path to the csv
        output:
                data_frame: pandas dataframe
        '''
        print("[INFO] read the data from {0}".format(pth))
        raw_data = pd.read_csv(pth)
        # Flag churn customer
        raw_data['Churn'] = raw_data['Attrition_Flag']\
            .apply(lambda val: 0 if val == "Existing Customer" else 1)
        # Drop column
        self.data_frame = raw_data.drop(['Unnamed: 0','CLIENTNUM','Attrition_Flag'], axis=1)
        return self.data_frame

    def perform_eda(self):
        '''
        perform eda on data_frame and save figures to images folder
        input:
                data_frame: pandas dataframe
        output:
                None
        '''
        # grouping columns by data type
        print("[INFO] Perform EDA")
        num_columns = self.data_frame.select_dtypes(include = "number")
        cat_columns = self.data_frame.select_dtypes(exclude = "number")
        # num_columns = self.data_frame[['Churn', 'Customer_Age', 'Total_Trans_Amt']]
        # cat_columns = self.data_frame[['Marital_Status', 'Gender']]
        # numeric data distribution plot
        for i in num_columns.columns:
            plt.figure(figsize=(20, 10))
            self.data_frame[i].hist()
            print("[INFO] Create Histogram plot of {0} column".format(i))
            plt.title("{0} Distribution".format(i))
            plt.savefig('./images/eda/{0}.png'.format(i))
            plt.close()
        # categorical data distribution plot
        for i in cat_columns.columns:
            plt.figure(figsize=(20, 10))
            self.data_frame[i].value_counts('normalize').plot(kind='bar')
            print("[INFO] Create Bar plot of {0} column".format(i))
            plt.title("{0} Distribution".format(i))
            plt.savefig('./images/eda/{0}.png'.format(i))
            plt.close()
        # heatmap plot
        print("[INFO] Create Heatmap Plot")
        plt.figure(figsize=(20, 10))
        sns.heatmap(self.data_frame.corr(), annot=False,
                    cmap='Dark2_r', linewidths=2)
        plt.savefig('./images/eda/heatmap.png')
        plt.close()

    def encoder_helper(self, category_lst):
        '''
        helper function to encode each categorical columns using
        One-Hot-Encoding method
        input:
                data_frame: pandas dataframe
                category_lst: list of columns that contain categorical features
        output:
                data_frame: pandas dataframe with new columns for
        '''
        print("[INFO] One-hot encoding categorical columns")
        self.data_frame = pd.get_dummies(self.data_frame, columns = category_lst)
        return self.data_frame

    def perform_feature_engineering(self, response, test_size = 0.3):
        '''
        input:
                  data_frame: pandas dataframe
                  response: string of response name [optional argument
                            that could be used for naming variables or index y column]
                  test_size: proportion of test data
        output:
                  x_train: X training data
                  x_test: X testing data
                  y_train: y training data
                  y_test: y testing data
        '''
        print("[INFO] Feature engineering")
        copy_data = self.data_frame.copy()
        predictor_data = copy_data.drop([response], axis = 1)
        response_data = copy_data[response]
        predictor_column_names = predictor_data.columns
        # Standardization
        print("[INFO] Standardization process")
        scaler = StandardScaler()
        scaled = scaler.fit_transform(predictor_data)
        predictor_scaled = pd.DataFrame(scaled, columns=predictor_column_names)
        print("[INFO] Data splitting")
        self.x_train, self.x_test,self.y_train, self.y_test =\
            train_test_split(predictor_scaled,response_data,
                             test_size = test_size,random_state = 123)
        return self.x_train, self.x_test,self.y_train, self.y_test

    def train_models(self):
        '''
        train, store model results: images + scores, and store models
        output:
                  None
        '''
        # Model object initiation
        print("[INFO] Model object inititation")
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression()
        # set parameters for tuning
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth' : [4,5,100],
            'criterion' :['gini', 'entropy']
        }
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        # model fitting
        print("[INFO] model training")
        cv_rfc.fit(self.x_train, self.y_train)
        lrc.fit(self.x_train, self.y_train)
        # Choosing best estimator
        y_train_preds_rf = cv_rfc.best_estimator_.predict(self.x_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(self.x_test)
        y_train_preds_lr = lrc.predict(self.x_train)
        y_test_preds_lr = lrc.predict(self.x_test)
        # save best model
        print("[INFO] Save the model objects")
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc, './models/logistic_model.pkl')
        # save roc curve
        print("[INFO] Create ROC curve")
        plt.figure(figsize=(15,8))
        plot_roc_curve(lrc,self.x_test,self.y_test,
                                 ax=plt.gca(),alpha=0.8)
        plot_roc_curve(cv_rfc.best_estimator_,
                       self.x_test,
                       self.y_test,
                       ax=plt.gca(),
                       alpha=0.8)
        plt.savefig('./images/results/roc_curve_result.png')
        plt.close()
        # Model report
        print("[INFO] Create classification report")
        classification_report_image(self.y_train,
                                    self.y_test,
                                    y_train_preds_lr,
                                    y_train_preds_rf,
                                    y_test_preds_lr,
                                    y_test_preds_rf)
        # feature importance
        print("[INFO] Create feature importance plot")
        feature_importance_plot(cv_rfc.best_estimator_,self.x_train,"./images/results")

if __name__ == "__main__":
    # object configuration
    PATH = "./data/bank_data.csv"

    CAT_COLUMNS = [
                'Gender',
                'Education_Level',
                'Marital_Status',
                'Income_Category',
                'Card_Category'
            ]

    RESPONSE = 'Churn'

    # model object initiation
    MODEL_INS = Model()

    # read the data
    MODEL_INS.import_data(PATH)

    # create eda plot and save the result in images/eda
    MODEL_INS.perform_eda()

    # encoding categorical feature
    MODEL_INS.encoder_helper(CAT_COLUMNS)

    # feature engineering (standardization and data splitting)
    MODEL_INS.perform_feature_engineering(RESPONSE)

    # model training and evaluation
    # model object was saved with .pkl extension in models folder
    # model evaluation result was saved in images/results
    MODEL_INS.train_models()
        