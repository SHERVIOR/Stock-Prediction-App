# Stock Price Prediction App

## Overview
This project compares two models — Linear Regression and Long Short-Term Memory (LSTM) — for predicting stock prices using historical data. The goal is to evaluate their effectiveness in modeling time-series financial data.

## Dataset Used
- **Stock Price Dataset** from Kaggle  
  URL: https://www.kaggle.com/datasets/mrsimple07/stock-price-prediction

## Models Applied
- Linear Regression (Classical Machine Learning)
- LSTM (Long Short-Term Memory Neural Network)

## Technologies
- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- TensorFlow / Keras
- Jupyter Notebook

## Workflow

### 1. Data Preprocessing
- Loading CSV stock price data
- Normalization / Scaling
- Feature selection (e.g., 'Close' price)
- Train-test split

### 2. Model Training
- Linear Regression model with sklearn
- LSTM model using Keras Sequential API
- Evaluation using MSE (Mean Squared Error)

### 3. Visualization and Comparison
- Plotting predicted vs. actual prices
- Visual and quantitative comparison of both models
- Interpretation of LSTM sequence learning vs. Linear trends

## Key Insights
- LSTM is better suited for capturing temporal dependencies in stock data
- Linear Regression offers faster computation but limited pattern recognition
- Proper preprocessing is essential for both models
- Visualizations help interpret and validate predictions
