#Stock prediction using linear regression and LSTM
#Made by SHERVIOR


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def load_kaggle_data(file_path):
    df = pd.read_csv(file_path, header=0, parse_dates=[0], dayfirst=True)
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    df.set_index('Date', inplace=True)
    df.rename(columns={'Stock_1': 'Close'}, inplace=True)
    df = df[['Close']]
    return df


def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['Close'] = scaler.fit_transform(df[['Close']])
    return df, scaler


def create_dataset(df, train_size=0.8):
    train_size = int(len(df) * train_size)
    train, test = df[:train_size], df[train_size:]
    return train, test


def train_linear_regression(train, test):
    X_train = np.array(range(len(train))).reshape(-1, 1)
    y_train = train['Close'].values
    model = LinearRegression()
    model.fit(X_train, y_train)
    X_test = np.array(range(len(train), len(train) + len(test))).reshape(-1, 1)
    predictions = model.predict(X_test)
    return predictions, model


def prepare_lstm_data(data, time_step=10):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step])
        Y.append(data[i + time_step])
    return np.array(X), np.array(Y)


def train_lstm(train):
    time_step = 10  # Can be adjusted
    X_train, Y_train = prepare_lstm_data(train['Close'].values, time_step)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(time_step, 1)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, Y_train, epochs=50, batch_size=16, verbose=1)
    return model


def predict_lstm(model, train, test, time_step=10):
    full_data = np.concatenate((train['Close'].values, test['Close'].values))
    X_test, _ = prepare_lstm_data(full_data, time_step)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    predictions = model.predict(X_test)
    predictions_test = predictions[len(train) - time_step:]
    return predictions_test


def evaluate_model(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse


def plot_predictions(test_index, actual, predictions_lr, predictions_lstm):
    plt.figure(figsize=(12, 6))
    plt.plot(test_index, actual, label="Actual Prices")
    plt.plot(test_index, predictions_lr, label="Linear Regression", linestyle="dashed")
    plt.plot(test_index, predictions_lstm, label="LSTM Prediction", linestyle="dotted")
    plt.title("Stock Price Prediction Comparison")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()


def main():
    file_path = "stock_data.csv"  # input data from kaggle
    df = load_kaggle_data(file_path)
    df, scaler = preprocess_data(df)
    train, test = create_dataset(df)
    predictions_lr, lr_model = train_linear_regression(train, test)
    lstm_model = train_lstm(train)
    predictions_lstm = predict_lstm(lstm_model, train, test)
    predictions_lr = scaler.inverse_transform(predictions_lr.reshape(-1, 1))
    predictions_lstm = scaler.inverse_transform(predictions_lstm.reshape(-1, 1))
    actual = scaler.inverse_transform(test['Close'].values.reshape(-1, 1))
    mae_lr, rmse_lr = evaluate_model(actual, predictions_lr)
    mae_lstm, rmse_lstm = evaluate_model(actual, predictions_lstm)
    print(f"Linear Regression -> MAE: {mae_lr:.4f}, RMSE: {rmse_lr:.4f}")
    print(f"LSTM -> MAE: {mae_lstm:.4f}, RMSE: {rmse_lstm:.4f}")
    test_index = test.index
    plot_predictions(test_index, actual, predictions_lr, predictions_lstm)

if __name__ == "__main__":
    main()
