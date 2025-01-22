import pandas as pd
import numpy as np
import pickle
print("\nLOGS :")

# LOADING FILE CONTAINING DICTS TO IMPUTE MISSING VALUES AND HANDLE OUTLIERS
print('Importing Values Dicts')  # LOGGING
with open('preprocessing_dicts/preprocessing_dicts.pkl', 'rb') as pickle_file:
    loaded_dicts = pickle.load(pickle_file)
print('\t\t Values Dicts Imported ✔️')  # LOGGING


print('Importing Feature Encoders')  # LOGGING
with open('feature_encoders/label_enc.pkl', 'rb') as pickle_file:
    label_encoder = pickle.load(pickle_file)
with open('feature_encoders/one_hot_encoder.pkl', 'rb') as pickle_file:
    one_hot_encoder = pickle.load(pickle_file)
print('\t\t Feature Encoders Imported ✔️')  # LOGGING


print('Importing Scaler ')  # LOGGING
with open('scalers/scaler.pkl', 'rb')as scaler :
    scaler = pickle.load(scaler)
print('\t\t Scaler Imported ✔️')  # LOGGING


print('Importing Model')  # LOGGING
with open('ml_model/model.pkl', 'rb')as model:
    ml_model = pickle.load(model)
print('\t\t Model Imported ✔️')  # LOGGING


# Creating a Dataframe
def create_df(dict): 
    print('Creating DataFrame ')  # LOGGING

    #sip->single_input
    df = pd.DataFrame([dict])

    print('\t\t DataFrame Created ✔️')  # LOGGING
    return df


# Renaming columns
def column_renamer(data): 
    print('Renaming Columns ')  # LOGGING

    renamed_data = data.copy()
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
    renamed_data.rename(columns=new_column_names, inplace=True)

    print('\t\t Columns Renamed ✔️')  # LOGGING
    return renamed_data


 # Handling missing values
def null_filler(data, loaded_dicts=loaded_dicts):
    print('Filling Null Features ')  # LOGGING
    loan_grade_bounds, mean_dict = loaded_dicts[0], loaded_dicts[1]

    def impute_interest_rate(row):
        if np.isnan(row['Interest_Rate']): 
            grade = row['Loan_Grade']
            if grade in loan_grade_bounds:
                lb, ub = loan_grade_bounds[grade]
                return round(np.random.uniform(lb, ub),2)  
        return row['Interest_Rate']

    data['Interest_Rate'] = data.apply(impute_interest_rate, axis=1)
    print('-Interest Rate ')  # LOGGING

    for col in data.select_dtypes(include=[int, float]).columns:
        print(f'-{col} ')  # LOGGING
        if data[col].isna().sum() > 0:
            data[col] = data[col].fillna(mean_dict[col])

    print('\t\t Filled Null Features ✔️')  # LOGGING
    return data


# Encoding data
def data_encoder(data, one_hot_encoder=one_hot_encoder, label_encoder=label_encoder): 
    print('Encoding Categoric Features ')  # LOGGING

    # ONE HOT ENCODING
    ohe_data = one_hot_encoder.transform(data[['Loan_Purpose', 'Home_Ownership']])
    ohe_data = pd.DataFrame(ohe_data, columns=one_hot_encoder.get_feature_names_out(['Loan_Purpose',
                                                                                      'Home_Ownership']))
    ohe_data = ohe_data.astype(int)
    print('-One Hot Encoding  ')  # LOGGING

    # LABEL ENCODING
    data['Loan_Grade']= label_encoder.transform(data['Loan_Grade'])
    data['Default_in_History'] = data['Default_in_History'].map({'Y':1, 'N':0})
    encoded_data = pd.merge(left=data.drop(['Home_Ownership', 'Loan_Purpose'], axis=1),
                        right=ohe_data, left_index=True, right_index=True)
    print('-Label Encoding  ')  # LOGGING
    
    print('\t\t Encoding Done  ✔️')  # LOGGING
    return encoded_data

# Handling outliers
def outlier_handling(data, loaded_dicts=loaded_dicts): 
    print('Capping Outliers ')  # LOGGING

    feature_bounds = loaded_dicts[2]
    for col in data.columns:
        lb, ub = feature_bounds[col][0], feature_bounds[col][1]
        data[col] = np.where(data[col] > ub, ub, data[col])
        data[col] = np.where(data[col] < lb, lb, data[col])
        print(f'-{col}')  # LOGGING

    print('\t\t Capped Outliers  ✔️')  # LOGGING
    return data


def column_dropper(data, cols=['Credit_History_Length', 'Interest_Rate']): # Dropping features causing multicollinearity
    print('Dropping Multicollinear Features ')  # LOGGING

    for col in cols:
        if col in data.columns:
            data = data.drop(columns=[col], axis=1)
            print(f'-{col}')  # LOGGING

    print('\t\t Multicollinear Features Dropped  ✔️')  # LOGGING
    return data


def data_scaler(data, scaler=scaler):
    print('Scaling Features ')  # LOGGING
    
    data = data.to_numpy()
    scaled_data = scaler.transform(data)

    print('\t\t Features Scaled  ✔️')  # LOGGING
    return scaled_data


def predictor_model(data, ml_model=ml_model):
    print(' Predicting New Data')  # LOGGING

    result = ml_model.predict_proba(data)

    print('\t\t Prediction Done  ✔️')  # LOGGING
    return result


def entire_pipeline(dict_data):
    print("\nLOGS :")
    print('------------------ Pipeline Started ------------------')  # LOGGING

    if isinstance(dict_data, dict):
        df = create_df(dict_data)
    else:
        df = dict_data 
    renamed_df = column_renamer(df)
    imputed_df = null_filler(renamed_df)
    encoded_df = data_encoder(imputed_df)
    capped_df = outlier_handling(encoded_df)
    processed_df = column_dropper(capped_df)
    scaled_df = data_scaler(processed_df)
    results = predictor_model(scaled_df)
    
    print('------------------ Pipeline Finished  ------------------')  # LOGGING
    return results