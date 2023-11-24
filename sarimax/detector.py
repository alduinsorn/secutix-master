# Faire les prédictions avec le modèle puis on va essayer de détecter les anomalies suivants certains critères

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import ARIMA
import joblib


ORDERS = (1, 0, 2)
SEASONAL_ORDERS = (0, 1, 1, 24)
MODEL_PATH = './model.pkl'


def save_model(model):
    joblib.dump(model, MODEL_PATH)

def load_model():
    model = joblib.load(MODEL_PATH)
    return model

def load_data(fn):
    data = pd.read_csv(fn)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')
    data.index = pd.DatetimeIndex(data.index.values, freq=data.index.inferred_freq)

    return data

def train_model(data):
    model = ARIMA(order=ORDERS, seasonal_order=SEASONAL_ORDERS)
    model.fit(data)
    return model

def create_trained_model(fn):
    data = load_data(fn)['paid_rate']
    model = train_model(data[:'2023-08-31']) # the last month is used for later testing
    save_model(model)

def predict(model, length):
    return model.predict(n_periods=length)

def detect_anomalies(model, data):
    # generate predictions and compute residuals
    predictions = predict(model, len(data))
    residuals = data - predictions

    # take into account only the negative residuals because problems are only when the paid rate is lower than expected
    residuals = residuals[residuals < 0]
    residuals = residuals.sort_values()

    # compute z-score that allow to have a nice metrics to detect anomalies
    z = (residuals - residuals.mean()) / residuals.std()
    z = z.sort_values()
    print("z-score") 
    print(z[:int(len(residuals) * 0.05)])

    # create a dataframe with the data, the predictions, the residuals and the z-score
    df = pd.DataFrame({'paid_rate': data, 'predictions': predictions, 'residuals': residuals, 'z-score': z})
    # print(df[(df['residuals'] <= -3.5) & (df.index.hour > 6)])

    return df
    


# create_trained_model('./data_ogone_norm_morning.csv')


model = load_model()
data = load_data('./data_ogone_norm_morning.csv')
data_test = data['2023-09-01':]['paid_rate']
df = detect_anomalies(model, data_test)
# add the 'total_transactions' column to the dataframe
df['total_transaction'] = data['total_transaction']

anomalies = df[df['residuals'] <= -3.5]
anomalies = anomalies[anomalies.index.hour > 6] # during the night it's not that relevant because there are not a lot of transactions
print(anomalies[anomalies['paid_rate'] <= 70])