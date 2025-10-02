Water Quality Prediction and Analysis

This project is built to help analyze and predict water quality using machine learning and deep learning models.

It takes raw water test data (like pH, minerals, salts, etc.), cleans and processes it, calculates important water quality indices, and then uses neural networks (MLP and LSTM) to predict and compare results.

The goal is simple: make water quality monitoring easier, more accurate, and supported by data-driven predictions.

What this project does

Cleans your dataset – Handles missing values, fixes errors, and standardizes column names.

Calculates key water quality indices such as:

WQI (Water Quality Index) – overall water quality score

SAR (Sodium Adsorption Ratio) – suitability for irrigation

PI (Permeability Index) – effect on soil permeability

Na% (Sodium Percentage) – sodium balance in water

RSC (Residual Sodium Carbonate) – carbonate hazard

Builds prediction models – Uses two types of models:

MLP (Multi-Layer Perceptron) – a simple artificial neural network

LSTM (Long Short-Term Memory) – a deep learning model good at learning sequences

Compares model performance – Checks accuracy using RMSE, R², and DTW.

Generates reports and visuals – Exports CSV files with predictions and creates charts to compare actual vs predicted results.

How to use it

Put your water test data in a file called data.csv.

The file should have columns like pH, TDS, Calcium, Magnesium, Sodium, etc.

Run the script:

python new.py


The program will:

Preprocess the dataset

Calculate water quality indices

Train both MLP and LSTM models

Save results and graphs for you to analyze

What you get

A new CSV file (water_quality_predictions.csv) with both actual and predicted values.

Another CSV (model_comparison_metrics.csv) that summarizes how well the models performed.

Charts that make it easy to compare actual vs predicted data:

RMSE comparison

R² score comparison

Heatmap of model performance

Line plots of actual vs predicted trends

Why this matters

Water is one of the most important resources, and monitoring its quality is critical for drinking, agriculture, and industry. This project uses AI to:

Automate analysis instead of manual calculations.

Predict trends in water quality using past data.

Visualize results so they’re easier to understand for decision-making.

Next steps

Add support for larger datasets and real-time monitoring.

Improve accuracy with hyperparameter tuning.

Build a simple dashboard so non-technical users can upload data and see results visually.
