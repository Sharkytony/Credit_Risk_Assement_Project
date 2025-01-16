<pre><h1>Project Documentation: Credit Risk Assessment
Author: Anthony Rodrigues</h1></pre>

<h2>Section 1: Data Preparation and Exploration</h2>

<h3>1.1 Data Ingestion and Library Setup</h3>
<ul>
  <li>Importing Data</li>
  <li>Preprocessing Libraries</li>
  <li>Statistical Libraries</li>
  <li>Visualization Libraries</li>
</ul>

<h3>1.2 Exploratory Data Analysis (EDA)</h3>
<ul>
  <li>EDA Tool</li>
  <li>Statistical Data Overview</li>
  <ul>
    <li>Mean, Median, Standard Deviation, Variance</li>
  </ul>
  <li>Data Visualization</li>
  <ul>
    <li>Histograms</li>
    <li>Correlation Matrix Heatmap</li>
    <li>Parallel Plot</li>
    <li>Pairplot</li>
  </ul>
</ul>

<h2>Section 2: Data Cleaning and Preprocessing</h2>

<h3>2.1 Handling Data Quality Issues</h3>
<ul>
  <li>Duplicate Handling</li>
  <li>Handling Constant Columns</li>
  <li>Handling Missing Values (Pandas-Profiling)</li>
</ul>

<h3>2.2 Data Transformation</h3>
<ul>
  <li>One-Hot Encoding for Cardinal Features</li>
  <li>Label Encoding for Ordinal Features</li>
  <li>Merging Dataframes</li>
</ul>

<h3>2.3 Outlier Detection and Treatment</h3>
<ul>
  <li>Identifying Gaussian and Skewed Distributions</li>
  <li>Outlier Detection using IQR and Percentile Methods</li>
  <li>Trimming and Capping Outliers</li>
</ul>

<h3>2.4 Data Balancing</h3>
<ul>
  <li>Downsampling Majority Class</li>
  <li>Upsampling Minority Class</li>
  <li>Merging Balanced Data</li>
  <li>Feature Scaling using StandardScaler</li>
  <li>Dimensionality Reduction</li>
</ul>

<h2>Section 3: Model Development</h2>

<h3>3.1 Data Splitting and Cross-Validation</h3>
<ul>
  <li>Stratified K-Fold Cross-Validation</li>
  <li>Train-Test Split</li>
</ul>

<h3>3.2 Model Selection and Training</h3>
<ul>
  <li>Boosting Models</li>
  <li>Bayesian Models</li>
  <li>K-Nearest Neighbors (KNN)</li>
  <li>Logistic Regression</li>
  <li>Fitting and Training Models</li>
</ul>

<h3>3.3 Hyperparameter Tuning</h3>
<ul>
  <li>Pruning Decision Trees</li>
  <li>Visualizing Pruning</li>
  <li>Model Tuning (GridSearchCV)</li>
</ul>

<h3>3.4 Model Evaluation</h3>
<ul>
  <li>Plotting Evaluation Scores</li>
</ul>

<h3>3.5 Ensemble Models</h3>
<ul>
  <li>Stacking Classifier</li>
  <li>Voting Classifier</li>
</ul>

<h2>Section 4: Pipeline Creation</h2>

<h3>4.1 Preprocessing Pipeline</h3>
<ul>
  <li>Preprocessor Functions (preprocessor.py)</li>
  <li>Training Preprocessor Pipeline (train_preprocessor.pkl)</li>
</ul>

<h3>4.2 Testing Pipeline</h3>
<ul>
  <li>Testing Preprocessor Functions (test_preprocessor.py)</li>
  <li>Testing Preprocessor Pipeline (test_preprocessor.pkl)</li>
</ul>

<h3>4.3 Model Pipeline</h3>
<ul>
  <li>Building and Saving Model Pipeline (model.pkl)</li>
  <li>Applying Pipelines for Fitting, Training, and Evaluation</li>
</ul>
