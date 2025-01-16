from flask import Flask, render_template, request, send_file
import joblib
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder='static')
test_preprocessing = joblib.load('test_preprocessor.pkl')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
@app.route('/preds', methods=['GET', 'POST'])

def prediction():
    model_ = joblib.load('model.pkl')    
    new_input =request.form.to_dict()
    new_input_l = list(new_input.values())
    for index, element in enumerate(new_input_l):
        if index != len(new_input_l)-2:
            try :
                element = int(element)
            except Exception as ex:
                element = str(element).upper()
            finally:
                new_input_l[index] = element
        else:
            if new_input_l[index] == 'Yes':
                new_input_l[index] = 'Y'
            else :
                new_input_l[index] ='N'

    processed_input = test_preprocessing.transform(new_input_l)
    predictions = model_.predict_proba(processed_input)
    predictions = predictions[0][1]
    labels = ['Wont Default', 'Will Default']
    sizes = [1 - predictions, predictions]

    plt.figure(figsize=(7, 6.5))
    plt.title('Probability for Default', fontsize="16")
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=45)
    plt.axis('equal') 

    plt.savefig('static/probability.png')
    return render_template('predictions_page.html')

if __name__ == "__main__":
    app.run(debug=True)
