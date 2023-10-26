import statsmodels.api as sm
from pmdarima.arima import auto_arima
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import standard scaler
from sklearn.preprocessing import StandardScaler


from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data_fn = '../database/data/data_ogone.csv'
# data_fn = '../database/data/data_adyen.csv'

# load the data into a pandas dataframe, then separate the training and testing data
data = pd.read_csv(data_fn)

# # Tracer l'ACF et la PACF, lags = 24 car on recupère les données chaque heure
# plot_acf(data['paid_rate'], lags=25)
# plt.savefig('analyses/acf_adyen.png')
# acf_vals = sm.tsa.stattools.acf(data['paid_rate'], nlags=25)
# print(f"ACF: {acf_vals}")
# plot_pacf(data['paid_rate'], lags=25)
# plt.savefig('analyses/pacf_adyen.png')
# pacf_vals = sm.tsa.stattools.pacf(data['paid_rate'], nlags=25)
# print(f"PACF: {pacf_vals}")
# # exit()

# # Effectuer le test ADF
# result = adfuller(data['paid_rate'])
# print('Statistique ADF :', result[0])
# print('p-value :', result[1])
# print('Valeurs critiques :', result[4])

# exit()


training_data = data[:int(0.8 * len(data))]
testing_data = data[int(0.8 * len(data)):]
# get headers
headers = list(data.columns.values)


# arima_model = auto_arima(y=training_data['paid_rate'],
#                          exogenous=training_data[headers[1:]],
#                          seasonal=True,
#                          stepwise=True,
#                          trace=True)
# print(arima_model.summary())

# exit()

# arima_model = sm.tsa.ARIMA(training_data['paid_rate'], exog=training_data[headers[1:]], order=(2, 0, 2))


arima_model = sm.tsa.SARIMAX(training_data['paid_rate'], exog=training_data[headers[1]], order=(4, 0, 2), seasonal_order=(0, 0, 0, 24))
# arima_model = sm.tsa.SARIMAX(training_data['paid_rate'], exog=training_data[headers[1:]], order=(4, 0, 4), seasonal_order=(4, 0, 4, 24))
arima_model = arima_model.fit()
# print(arima_model.summary())


predictions = arima_model.get_forecast(steps=len(testing_data), exog=testing_data[headers[1]])
predictions = predictions.predicted_mean
predictions = pd.DataFrame(predictions, index=testing_data.index)
predictions.columns = ['pred_paid_rate']

# make that the prediction doesnt go below 0 and above 100
# predictions['pred_paid_rate'] = predictions['pred_paid_rate'].apply(lambda x: 0 if x < 0 else x)
# predictions['pred_paid_rate'] = predictions['pred_paid_rate'].apply(lambda x: 100 if x > 100 else x)

# predict the success rate for the testing data
# predictions = arima_model.predict(start=len(training_data), end=len(training_data) + len(testing_data) - 1, exog=testing_data[headers[1:]])
# convert the predictions into a pandas dataframe
# predictions = pd.DataFrame(predictions, index=testing_data.index)
# predictions.columns = ['pred_paid_rate']

# evaluate the model
# calculate the mean absolute error and RMSE
mae = np.mean(np.abs(predictions['pred_paid_rate'].values - testing_data['paid_rate'].values))
rmse = np.sqrt(np.mean((predictions['pred_paid_rate'].values - testing_data['paid_rate'].values) ** 2))

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# plot the predictions
plt.figure(figsize=(12, 7))
# plt.plot(training_data.index, training_data['paid_rate'], label='training')
# make the real data heavier
plt.plot(testing_data.index, testing_data['paid_rate'], label='testing')
plt.legend(loc='upper left', fontsize=8)
plt.xlabel('Date')
plt.ylabel('Paid rate')
plt.plot(predictions.index, predictions['pred_paid_rate'], label='predicted', linestyle='--')
plt.show()
plt.savefig('arima.png')




