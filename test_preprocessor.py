import pandas as pd
import numpy as np
import pickle

# LOADING FILE CONTAINING DICTS TO IMPUTE MISSING VALUES AND HANDLE OUTLIERS
with open('preprocessing_dicts/preprocessing_dicts.pkl', 'rb') as pickle_file:
    loaded_dicts = pickle.load(pickle_file)


def create_df(dict): # Creating a Dataframe
    #sip->single_input
    df = pd.DataFrame(dict)
    return df


def column_renamer(data): # Renaming columns
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
    return renamed_data


def null_filler(data, loaded_dicts=loaded_dicts): # Handling missing values
    loan_grade_bounds, mean_dict = loaded_dicts[0], loaded_dicts[1]

    def impute_interest_rate(row):
        if np.isnan(row['Interest_Rate']): 
            grade = row['Loan_Grade']
            if grade in loan_grade_bounds:
                lb, ub = loan_grade_bounds[grade]
                return round(np.random.uniform(lb, ub),2)  
        return row['Interest_Rate']

    data['Interest_Rate'] = data.apply(impute_interest_rate, axis=1)

    for col in data.columns:
        if data[col].isna().sum() > 0:
            data[col] = data[col].fillna(mean_dict[col])

    return data


def data_encoder(data): # Encoding data
    with open('feature_encoders/label_enc.pkl', 'rb') as pickle_file:
        label_encoder = pickle.load(pickle_file)
    with open('feature_encoders/one_hot_encoder.pkl', 'rb') as pickle_file:
        one_hot_encoder = pickle.load(pickle_file)

    # ONE HOT ENCODING
    ohe_data = one_hot_encoder.transform(data[['Loan_Purpose', 'Home_Ownership']])
    ohe_data = pd.DataFrame(ohe_data, columns=one_hot_encoder.get_feature_names_out(['Loan_Purpose',
                                                                                      'Home_Ownership']))
    ohe_data = ohe_data.astype(int)

    # LABEL ENCODING
    data['Loan_Grade']= label_encoder.transform(data['Loan_Grade'])
    data['Default_in_History'] = data['Default_in_History'].map({'Y':1, 'N':0})
    encoded_data = pd.merge(left=data.drop(['Home_Ownership', 'Loan_Purpose'], axis=1),
                        right=ohe_data, left_index=True, right_index=True)

    return encoded_data


def outlier_handling(data, loaded_dicts=loaded_dicts): # Handling outliers
    feature_bounds = loaded_dicts[2]
    for col in data.columns:
        lb, ub = feature_bounds[col][0], feature_bounds[col][1]
        data[col] = np.where(data[col] > ub, ub, data[col])
        data[col] = np.where(data[col] < lb, lb, data[col])
    return data


def column_dropper(data, cols=['Credit_History_Length', 'Interest_Rate']): # Dropping features causing multicollinearity
    for col in cols:
        if col in data.columns:
            data = data.drop(columns=cols, axis=1)
    return data


def data_scaler(data):
    data = data.to_numpy()
    with open('scalers/scaler.pkl', 'rb')as scaler :
        scaler = pickle.load(scaler)
    scaled_data = scaler.transform(data)
    return scaled_data


def predictor_model(data):
    with open('ml_model/model.pkl', 'rb')as model:
        ml_model = pickle.load(model)

    result = ml_model.predict_proba(data)
    return result


def entire_pipeline(dict_data):
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
    return results