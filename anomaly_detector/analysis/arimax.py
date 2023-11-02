import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import standard scaler
from sklearn.preprocessing import StandardScaler
from time import time

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def basic_analysis(data):
    # Tracer l'ACF et la PACF, lags = 24 car on recupère les données chaque heure
    plot_acf(data['paid_rate'], lags=25)
    plt.savefig('analyses/acf_adyen.png')
    acf_vals = sm.tsa.stattools.acf(data['paid_rate'], nlags=25)
    print(f"ACF: {acf_vals}")
    plot_pacf(data['paid_rate'], lags=25)
    plt.savefig('analyses/pacf_adyen.png')
    pacf_vals = sm.tsa.stattools.pacf(data['paid_rate'], nlags=25)
    print(f"PACF: {pacf_vals}")
    # exit()

    # Effectuer le test ADF
    result = adfuller(data['paid_rate'])
    print('Statistique ADF :', result[0])
    print('p-value :', result[1])
    print('Valeurs critiques :', result[4])

    return

def make_train_arimax(train_data, test_data, headers, order, trend, output_folder, save=False):
    time_start = time()
    ## YEARLY
    # arima_model = sm.tsa.ARIMA(training_data['paid_rate'], exog=training_data[headers_alone], order=order_alone, trend=trend_alone)
    # arima_model = sm.tsa.ARIMA(train_data['paid_rate'], exog=train_data[headers], order=order, trend=trend)

    ## MONTHLY
    # arima_model = sm.tsa.ARIMA(training_data['paid_rate'], exog=training_data[headers[1:]], order=(22, 1, 19), trend='t')

    ## SARIMAX models aren't working well
    # arima_model = sm.tsa.statespace.SARIMAX(training_data['paid_rate'], exog=training_data[headers], order=(0, 1, 19), seasonal_order=(19, 0, 16, 18), trend='t')
    arima_model = sm.tsa.statespace.SARIMAX(training_data['paid_rate'], exog=training_data[headers], order=(4, 0, 4), seasonal_order=(4, 0, 4, 24))


    arima_model = arima_model.fit()
    # print(arima_model.summary())

    print(f"Training time: {round(time() - time_start, 2)} seconds")
    time_start = time()

    predictions = arima_model.get_forecast(steps=len(test_data), exog=test_data[headers])
    predictions = predictions.predicted_mean
    predictions = pd.DataFrame(predictions, index=test_data.index)
    predictions.columns = ['pred_paid_rate']

    # display only 2 decimals of the time taken to predict
    print(f"Prediction time: {round(time() - time_start, 2)} seconds")

    # evaluate the model
    # calculate the mean absolute error and RMSE
    mae = np.mean(np.abs(predictions['pred_paid_rate'].values - test_data['paid_rate'].values))
    rmse = np.sqrt(np.mean((predictions['pred_paid_rate'].values - test_data['paid_rate'].values) ** 2))

    print(f"MAE: {round(mae, 2)}")
    print(f"RMSE: {round(rmse, 2)}")

    # print AIC, BIC and HQIC
    print(f"AIC: {round(arima_model.aic, 2)}")
    print(f"BIC: {round(arima_model.bic, 2)}")
    print(f"HQIC: {round(arima_model.hqic, 2)}")

    # plot the predictions
    plt.figure(figsize=(12, 7))
    plt.plot(test_data.index, test_data['paid_rate'], label='testing')
    plt.plot(predictions.index, predictions['pred_paid_rate'], label='predicted', linestyle='--')
    plt.legend(loc='upper left', fontsize=8)
    plt.xlabel('Date')
    plt.ylabel('Paid rate')
    if save: plt.savefig(f'{output_folder}/arimax.png')
    plt.show()

    # Calculer les résidus en soustrayant les valeurs réelles des prédictions
    residuals = test_data['paid_rate'] - predictions['pred_paid_rate']

    # Tracer les résidus
    plt.figure(figsize=(10, 6))
    plt.plot(residuals)
    plt.title('Residuals Plot')
    plt.xlabel('Time')
    plt.ylabel('Residuals')
    if save: plt.savefig(f'{output_folder}/residuals.png')
    plt.show()

    # Afficher un histogramme des résidus
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, density=True, alpha=0.6, color='b', label='Residuals')
    plt.title('Residuals Histogram')
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.legend()
    if save: plt.savefig(f'{output_folder}/residuals_hist.png')
    plt.show()

    # Calculer et afficher la moyenne des résidus
    mean_residuals = np.mean(residuals)
    print(f"Mean Residuals: {round(mean_residuals, 2)}")

    # Calculer et afficher l'écart type des résidus
    std_residuals = np.std(residuals)
    print(f"Standard Deviation of Residuals: {round(std_residuals, 2)}")

# data_fn = '../database/data/real_data_ogone.csv'
data_fn = '../database/data/real_data_ogone_incidents.csv'

# load the data into a pandas dataframe, then separate the training and testing data
data = pd.read_csv(data_fn)

# convert the 'incident' column that is boolean to int (with true=1 and false=0)
data['incident'] = data['incident'].astype(int)

training_data = data[:int(0.8 * len(data))]
# testing_data = data[int(0.8 * len(data)):]
# get the last month -> 24*30 = 720
testing_data = data[-720:]

print(f"Size of training data: {len(training_data)} & size of testing data: {len(testing_data)}")

headers_alone = ['total_transaction_count']
order_alone = (29, 0, 9)
trend_alone = 'c'
headers_combined = ['total_transaction_count', 'incident']
order_combined = (12, 1, 11)
trend_combined = 't'

print("Start training...")

make_train_arimax(training_data, testing_data, headers_alone, order_alone, trend_alone, 'alone')
# make_train_arimax(training_data, testing_data, headers_combined, order_combined, trend_combined, 'combined')