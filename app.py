from flask import Flask, render_template, request, send_file
import joblib
import matplotlib.pyplot as plt
import time

app = Flask(__name__, static_folder='static')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

import test_preprocessor #lazy loading
 
@app.route('/preds', methods=['GET', 'POST'])
def prediction():

    data_dict = {
        'person_age': int(request.form['person_age']),
        'person_income': int(request.form['person_income']),
        'person_home_ownership': request.form['person_home_ownership']  ,
        'person_emp_length': int(request.form['person_emp_length']) ,
        'loan_intent': request.form['loan_intent']  ,
        'loan_grade': request.form['loan_grade'] ,
        'loan_amnt': int(request.form['loan_amnt']) ,
        'loan_int_rate': float(request.form['loan_int_rate']),
        'loan_percent_income': float(request.form['loan_percent_income']) ,
        'cb_person_default_on_file': request.form['cb_person_default_on_file'] ,
        'cb_person_cred_hist_length': int(request.form['cb_person_cred_hist_length']) 
    }
    start = time.time()
    predictions = test_preprocessor.entire_pipeline(data_dict)
    predictions = predictions[0][1]
    labels = ['Wont Default', 'Will Default']
    sizes = [1 - predictions, predictions]

    plt.figure(figsize=(7, 6.5))
    plt.title('Probability for Default', fontsize="16")
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=45)
    plt.axis('equal') 

    plt.savefig('static/probability.png')
    time_taken = time.time() - start
    print('Time taken : ', time_taken)
    return render_template('predictions_page.html')


if __name__ == "__main__":
    app.run(debug=True)