import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.preprocessing import OneHotEncoder

def col_renamer_for_df(data):
    transformed_data = data.copy()
    new_column_names = {
        'person_age': 'Age',
        'person_income': 'Income',
        'person_home_ownership': 'Home_Ownership',
        'person_emp_length': 'Employment_Length',
        'loan_intent': 'Loan_Purpose',
        'loan_grade': 'Loan_Grade',
        'loan_amnt': 'Loan_Amount',
        'loan_int_rate': 'Interest_Rate',
        'loan_status': 'Loan_Status',
        'loan_percent_income': 'Loan_Income_ratio',
        'cb_person_default_on_file': 'Default_in_History',
        'cb_person_cred_hist_length': 'Credit_History_Length'
    }
    transformed_data.rename(columns=new_column_names, inplace=True)
    return transformed_data


def col_renamer_for_sip(data):#sip->single_input
    transformed_data = data.copy()
    new_column_names = {
        0 : 'Age',
        1 : 'Income',
        2 : 'Home_Ownership',
        3 : 'Employment_Length',
        4 : 'Loan_Purpose',
        5 : 'Loan_Grade',
        6 : 'Loan_Amount',
        7 : 'Interest_Rate',
        8 : 'Loan_Income_ratio', 
        9 : 'Default_in_History',
        10: 'Credit_History_Length'
    }
    transformed_data.rename(columns=new_column_names, inplace=True)
    return transformed_data

def null_filler(data):
    cols = list(data.columns)
    data_new = data.copy()
    for col in cols:
        if data[col].isnull().sum() > 0:
            data_new[col] = data_new[col].fillna(data_new[col].mode()[0])
            data_new.reset_index(drop=True , inplace=True)
    return data_new

def one_hot_encode(input_data):
    encoder = joblib.load('train_ohe.pkl')
    ohe_data = encoder.transform(input_data[['Loan_Purpose', 'Home_Ownership']])
    ohe_data = pd.DataFrame(ohe_data, columns=encoder.get_feature_names_out(['Loan_Purpose', 'Home_Ownership']))
    ohe_data = ohe_data.astype(int)
    return ohe_data

def label_encode(input_data, file='train_label_encoder.pkl'):
    le = joblib.load('train_le.pkl')
    transformed_data = input_data.copy()
    transformed_data['Loan_Grade'] = le.transform(transformed_data['Loan_Grade'])
    transformed_data['Default_in_History'] = transformed_data['Default_in_History'].map({'Y': 1, 'N': 0})
    return transformed_data

def column_dropper(input_data, cols=['Loan_Purpose', 'Home_Ownership']):
    transformed_data = input_data.drop(columns=cols)
    return transformed_data

def df_merger(data1, data2):
    transformed_data = pd.merge(left=data1, right=data2, left_index=True, right_index=True)
    return transformed_data

class OutlierHandler:
    def __init__(self, data):
        self.data = data

    def outlier_counter(self, col):
        q1 = np.percentile(col, 25)
        q3 = np.percentile(col, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = len(col[(col < lower_bound) | (col > upper_bound)])
        return outliers, lower_bound, upper_bound

    def outlier_capper(self, col_name, lower_bound, upper_bound):
        self.data[col_name] = np.where(self.data[col_name] < lower_bound, lower_bound, self.data[col_name])
        self.data[col_name] = np.where(self.data[col_name] > upper_bound, upper_bound, self.data[col_name])

    def outlier_handling(self, cols_to_handle=['Employment_Length', 'Interest_Rate', 'Loan_Income_ratio', 'Income',
                                                  'Loan_Amount', 'Age']):
        warnings.filterwarnings('ignore')
        for col in cols_to_handle:
            outliers, lb, ub = self.outlier_counter(self.data[col])
            if outliers > 0:
                self.outlier_capper(col, lb, ub)
            else:
                pass
        return self.data

def test_preprocess(test_data):
    scaler = joblib.load('train_scaler.pkl')
    if isinstance(test_data, pd.DataFrame):
        data1 = col_renamer_for_df(test_data)
        data2 = null_filler(data1)
        data3, data4 = one_hot_encode(data2), label_encode(data2)
        data5 = column_dropper(data4)
        data6 = df_merger(data5, data3)
        ol_handler = OutlierHandler(data6)
        data7 = ol_handler.outlier_handling()
        data7.columns = data7.columns.astype(str)
        test_data_scaled = scaler.transform(data7)
        return test_data_scaled
    elif isinstance(test_data, list): #single input
        data = pd.DataFrame([test_data])
        data1 = col_renamer_for_sip(data)
        data2 = null_filler(data1)
        data3, data4 = one_hot_encode(data2), label_encode(data2)
        data5 = column_dropper(data4)
        data6 = df_merger(data5, data3)
        ol_handler = OutlierHandler(data6)
        data7 = ol_handler.outlier_handling()
        data7.columns = data7.columns.astype(str)
        test_data_scaled = scaler.transform(data7)
        return test_data_scaled
    else:
        raise ValueError("Input must be a dataframe or a list of dataframes")