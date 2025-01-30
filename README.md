# Credit Risk Assessment 💵✔️ - Supervised ML
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-red.svg)](https://scikit-learn.org/)

## Overview 🔎
This project is a machine learning-based web application for assessing credit risk. It utilizes a comprehensive data preprocessing pipeline and advanced machine learning algorithms to classify loan applications based on the likelihood of default. The application is built using Python, Flask for the web interface, and scikit-learn for the machine learning model.

## Aim 🎯
- Accurately classify loan applicants into risk categories (low, medium, or high risk).
- Provide insights into factors affecting creditworthiness.
- Support financial institutions in making informed lending decisions.
- Create a scalable, interpretable, and robust credit risk assessment system.

### Terminal logging
<img src="https://github.com/anthonyrodrigues443/Credit_Risk_Assement_Project/blob/main/web_page_images/terminal_logs.png" width="400px"><img src="https://github.com/anthonyrodrigues443/Credit_Risk_Assement_Project/blob/main/web_page_images/terminal_logs2.png" width="400px"><img src="https://github.com/anthonyrodrigues443/Credit_Risk_Assement_Project/blob/main/web_page_images/terminal_logs3.png" width="400px">

### Webpage
<img src="https://github.com/anthonyrodrigues443/Credit_Risk_Assement_Project/blob/main/web_page_images/web_1.png" width="400px"><img src="https://github.com/anthonyrodrigues443/Credit_Risk_Assement_Project/blob/main/web_page_images/web_2.png" width="400px">

### Output page
<img src="https://github.com/anthonyrodrigues443/Credit_Risk_Assement_Project/blob/main/web_page_images/web_op.png" width="400px">
  
## Project Structure 🗂️
```
Credit_Risk_Assesment_Project/
├── credit_risk_proj/         # python env
├── dataset/                  # dataset
├── eda_reports/              # Exploratory Data Analysis reports
├── feature_encoders/         # Saved encoders for categorical features
├── ml_model/                 # Trained machine learning voting classifier(lgbm+xgb) model
├── plotly_graphs/            # Interactive visualizations
├── preprocessing_dicts/      # Preprocessing configurations
├── scalers/                  # Feature scaler
├── static/                   # Static files (web bg images)
├── templates/                # HTML templates(home page and prediction page) for Flask
├── web_page_images/          # Images of web page and terminal logging
├── app.py                    # Flask application entry point
├── credit_risk_assesment.ipynb # Jupyter Notebook for model building
├── requirements.txt          # Python dependencies
├── test_preprocessor.py      # test data preprocessing pipeline
```

## Features ⭐
- Comprehensive preprocessing pipeline:
  - Missing value treatment
  - Categorical feature encoding
  - Numerical feature scaling
  - Handling outliers
- Machine learning model for credit risk classification:
  - Voting Classifier consisting - Light Gradient Boosting Machine and XBG Classifier
- Interactive web interface for:
  - Inputting applicant details
  - Displaying risk prediction results

## Installation 🧑‍🔧
1. Clone the repository:
   ```bash
   git clone https://github.com/Sharkytony/Credit_Risk_Assement_Project.git
   cd Credit_Risk_Assement_Project
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage 👨🏻‍💻
1. Start the Flask application:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to `http://localhost:5000`.
3. Enter applicant details in the provided form.
4. View the predicted credit risk category and supporting visualizations.

## Data Preprocessing Pipeline ⛓️
- **Handling Multicollinearity:**
  - Debt-to-Income Ratio calculation
  - Credit History length extraction
- **Encoding:**
  - One-hot encoding for cardinal features
  - Label encoding for ordinal features
- **Scaling:**
  - Standard scaling for numerical features
- **Outlier Handling:**
  - Winsorization for extreme values

### Example Usage:
```python
import pipeline

# Create input DataFrame
predictions = pipeline.entire_pipeline(
    age=35, income=50000, credit_score=700, loan_amount=20000, ...
)
# Predict risk
print(predictions)

# Output Format :
# [ probability for class 0, probability for class 1]
```

## Model Details 🤖
- Algorithms Used:
  - XGB Classifier (baseline)
  - LGBM Classifier (baseline)
  - Voting Classifier
- Evaluation Metrics:
  - Specificity, Recall
  - ROC-AUC for model comparison
- Hyperparameter Tuning

## Contributing 🤝
1. Fork the repository.
2. Create a new feature branch:
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a pull request.

## License 📋
This project is licensed under the MIT License. See the [LICENSE](https://github.com/Sharkytony/Credit_Risk_Assement_Project/blob/main/LICENSE) file for more details.

<h3>⭐ Don't forget to star the repository if you find it helpful!
