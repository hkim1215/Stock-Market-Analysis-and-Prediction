import yfinance as yf 
import pandas as pd 
import numpy as np 
import math

from datetime import datetime, timedelta
from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler 
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

app = Flask(__name__)

# fetching stock data 
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    df.dropna(inplace=True)  # Drop rows with missing values
    return df

# pre-process data for LSTM Model 
def preprocess_data(df):
    df.index = df.index.date 
    data = df.filter(['Close']) # create a new dataframe with only the 'Close' column
    dataset = data.values # convert the dataframe to a numpy array
    training_ratio = 0.8
    training_data_len = math.ceil(len(dataset) * training_ratio) # calculate the length of the training set 
    n_lookback = 60 # the number of past time steps that the model will use to predict the next value

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[0:training_data_len, :] # create training data set 

    x_train = []
    y_train = []

    for i in range(n_lookback, len(train_data)):
        x_train.append(train_data[i-n_lookback:i, 0])
        y_train.append(train_data[i, 0])

    x_train = np.array(x_train) # convert set into NumPy array 
    y_train = np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # Reshape the array to be 3-dimensional for LSTM input.

    return x_train, y_train, scaler, scaled_data, training_data_len, n_lookback, dataset

# define LSTM Model 
def create_model(x_train):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))  # Added dropout layer
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))  # Added dropout layer
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error') # compile model 
    return model 

# train LSTM model and make predictions 
def train_and_predict(df):
    x_train, y_train, scaler, scaled_data, training_data_len, n_lookback, dataset = preprocess_data(df)
    model = create_model(x_train)
    model.fit(x_train, y_train, batch_size=16, epochs=5)

    test_data = scaled_data[training_data_len - n_lookback:, :]  # create the testing data set 

    x_test = []

    for i in range(n_lookback, len(test_data)):
        x_test.append(test_data[i - n_lookback:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    return predictions, model, scaler, scaled_data

# predict the future 7 days
def predict_future_prices(model, scaler, scaled_data):
    last_60_days = scaled_data[-60:]
    input_seq = last_60_days.reshape(1, 60, 1)
    predictions = []

    for _ in range(7):
        predicted_price = model.predict(input_seq)
        predictions.append(scaler.inverse_transform(predicted_price).flatten()[0])
        input_seq = np.append(input_seq[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)

    day_labels = [f"+ {i+1}" for i in range(7)]

    predicted_df = pd.DataFrame({
        'Days': day_labels,
        'Predicted Price': predictions
    })

    return predicted_df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = (request.form['ticker']).upper()
    start_date = datetime.today() - timedelta(days=3650)
    end_date = datetime.today()

    df = fetch_stock_data(ticker, start_date, end_date)
    predictions, model, scaler, scaled_data = train_and_predict(df)
    future_predictions = predict_future_prices(model, scaler, scaled_data)

    return render_template('results.html', 
                           future_predictions=future_predictions.to_dict(orient='records'))



if __name__ == '__main__':
    app.run(debug=True)
