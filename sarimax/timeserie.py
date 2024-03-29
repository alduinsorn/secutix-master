# Libraries
# ======================================================================================
import numpy as np
import pandas as pd
from io import StringIO
import contextlib
import re
import matplotlib.pyplot as plt
# plt.style.use('seaborn-v0_8-darkgrid')
import time
import os

# pmdarima
from pmdarima import ARIMA, auto_arima

# statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# skforecast
from skforecast.Sarimax import Sarimax
from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.model_selection_sarimax import backtesting_sarimax
from skforecast.model_selection_sarimax import grid_search_sarimax
from sklearn.metrics import mean_absolute_error
from skforecast.ForecasterAutoreg import ForecasterAutoreg

import warnings
import pickle
import random




def load_data(fn, exog=False):
    # data = pd.read_csv('./data_ogone_norm_global.csv')
    data = pd.read_csv(fn)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')
    data.index = pd.DatetimeIndex(data.index.values, freq=data.index.inferred_freq)

    if not exog:
        data.drop(['total_transaction'], axis=1, inplace=True)

    return data

# data_train = data.loc[:'2023-08-31']
# data_test = data.loc['2023-09-01':]


# p,d,q = 1,1,1
# P,D,Q = 1,1,1
# p,d,q = 1,0,2
# P,D,Q = 0,1,1
# s = 24


def test_stationarity(data, data_diff_1, data_diff_2):
    # Test stationarity
    # ==============================================================================
    warnings.filterwarnings("ignore")

    print('Test stationarity for original series')
    print('-------------------------------------')
    adfuller_result = adfuller(data)
    kpss_result = kpss(data)
    print(f'ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}')
    print(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')

    print('\nTest stationarity for differenced series (order=1)')
    print('--------------------------------------------------')
    adfuller_result = adfuller(data_diff_1)
    kpss_result = kpss(data.diff().dropna())
    print(f'ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}')
    print(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')

    print('\nTest stationarity for differenced series (order=2)')
    print('--------------------------------------------------')
    adfuller_result = adfuller(data_diff_2)
    kpss_result = kpss(data.diff().diff().dropna())
    print(f'ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}')
    print(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')

    warnings.filterwarnings("default")

    ### RESULT ###
    # -------------------------------------
    # ADF Statistic: -9.39059126185539, p-value: 6.577385914466176e-16          -> Best order = 0 because < 0.05 compare to the others
    # KPSS Statistic: 3.4135156375023734, p-value: 0.01
    # 
    # Test stationarity for differenced series (order=1)
    # --------------------------------------------------
    # ADF Statistic: -31.748384687836403, p-value: 0.0
    # KPSS Statistic: 0.012487368140781507, p-value: 0.1
    # 
    # Test stationarity for differenced series (order=2)
    # --------------------------------------------------
    # ADF Statistic: -31.435599564250282, p-value: 0.0
    # KPSS Statistic: 0.03157325152866638, p-value: 0.1


    # Plot series
    # ==============================================================================
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(7, 5), sharex=True)
    data.plot(ax=axs[0], title='Original time series')
    data_diff_1.plot(ax=axs[1], title='Differenced order 1')
    data_diff_2.plot(ax=axs[2], title='Differenced order 2')
    fig.tight_layout()
    plt.show()


def plot_acf_pacf(data, data_diff_1, savefig=False):
    # Autocorrelation plot for original and differenced series
    # ==============================================================================
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 4), sharex=True)
    plot_acf(data, ax=axs[0], lags=50, alpha=0.05)
    axs[0].set_title('Autocorrelation original series')
    plot_acf(data_diff_1, ax=axs[1], lags=50, alpha=0.05)
    axs[1].set_title('Autocorrelation differenced series (order=1)')


    # Partial autocorrelation plot for original and differenced series
    # ==============================================================================
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 3), sharex=True)
    plot_pacf(data, ax=axs[0], lags=50, alpha=0.05)
    axs[0].set_title('Partial autocorrelation original series')
    plot_pacf(data_diff_1, ax=axs[1], lags=50, alpha=0.05)
    axs[1].set_title('Partial autocorrelation differenced series (order=1)')
    plt.tight_layout()
    plt.show()

    ### RESULT ###
    # ACF -> 1 so q = 1 (seasonality so using the differenced series)
    # PACF -> 1 so p = 1


def plot_decomposition(data, data_diff_1):
    # Time series decomposition of original versus differenced series
    # ==============================================================================
    res_decompose = seasonal_decompose(data, model='additive', extrapolate_trend='freq')
    res_descompose_diff_2 = seasonal_decompose(data_diff_1, model='additive', extrapolate_trend='freq')

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(9, 6), sharex=True)

    res_decompose.observed.plot(ax=axs[0, 0])
    axs[0, 0].set_title('Original series')
    res_decompose.trend.plot(ax=axs[1, 0])
    axs[1, 0].set_title('Trend')
    res_decompose.seasonal.plot(ax=axs[2, 0])
    axs[2, 0].set_title('Seasonal')
    res_decompose.resid.plot(ax=axs[3, 0])
    axs[3, 0].set_title('Residuals')
    res_descompose_diff_2.observed.plot(ax=axs[0, 1])
    axs[0, 1].set_title('Differenced series (order=1)')
    res_descompose_diff_2.trend.plot(ax=axs[1, 1])
    axs[1, 1].set_title('Trend')
    res_descompose_diff_2.seasonal.plot(ax=axs[2, 1])
    axs[2, 1].set_title('Seasonal')
    res_descompose_diff_2.resid.plot(ax=axs[3, 1])
    axs[3, 1].set_title('Residuals')
    fig.suptitle('Time serie decomposition original series versus differenced series', fontsize=14)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    ### RESULT ###
    # on remarque qu'il y a de la saisonnalité dans les données tout les 24 heures (car données sont prise toutes les heures) et que chaque jour il se passe la même chose
    # Au vu des graphiques il n'est pas necessaire de faire une différence d'ordre car les données sembles suffisament stationnaire

def adfuller_test(data_train):
    # Donc on va prendre les données et les différencier d'ordre 24 pour enlever la saisonnalité
    data_diff_0_24 = data_train.diff(24).dropna()

    warnings.filterwarnings("ignore")
    adfuller_result = adfuller(data_diff_0_24)
    print(f'ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}')
    kpss_result = kpss(data_diff_0_24)
    print(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')
    warnings.filterwarnings("default")

    ### RESULT ###
    # ADF Statistic: -17.636408209961957, p-value: 3.776999068431006e-30
    # KPSS Statistic: 0.0047628837985794915, p-value: 0.1

def sarima_forecaster(data_train, data_test, exog=False, plot=False):
    forecaster = ForecasterSarimax(regressor=Sarimax(order=(p,d,q), seasonal_order=(P,D,Q,s)))
    
    if exog:
        forecaster.fit(y=data_train['paid_rate'], exog=data_train['total_transaction'])
    else:
        forecaster.fit(y=data_train['paid_rate'])

    # Prediction
    if exog:
        predictions = forecaster.predict(steps=len(data_test), exog=data_test['total_transaction'])
    else:
        predictions = forecaster.predict(steps=len(data_test))
    
    predictions.name = 'predictions'
    predictions = pd.concat([data_test, predictions], axis=1)

    if exog:
        predictions = predictions.drop(['total_transaction'], axis=1)

    print(predictions.head(4))

    print(f"MAE: {mean_absolute_error(data_test['paid_rate'], predictions['predictions'])}")

    if plot:
        name = 'pred_arima'
        if exog:
            add = f'{name}_exog'
        
        fig, ax = plt.subplots(figsize=(20, 10))
        predictions.plot(ax=ax, label='pred')
        ax.set_title('Predictions with ARIMA models')
        ax.legend()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.tight_layout()
        plt.savefig(f'{name}.png')
        plt.show()

        fig, ax = plt.subplots(figsize=(20, 10))
        predictions.loc['2023-09-18':'2023-09-21'].plot(ax=ax, label='pred')
        ax.set_title('Predictions with ARIMA models')
        ax.legend()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.tight_layout()
        plt.savefig(f'{name}_specific.png')
        plt.show()


    return forecaster

def sarima_other(data_train, data_test, exog=False, plot=False):
    try:
        if exog:
            # model = auto_arima(data_train['paid_rate'], exogenous=data_train['total_transaction'], seasonal=True, )
            model = ARIMA(order=(p,d,q), seasonal_order=(P,D,Q,s))
        else:
            model = ARIMA(order=(p,d,q), seasonal_order=(P,D,Q,s))
        print("Between")
        if exog:
            model.fit(y=data_train['paid_rate'], exogenous=data_train['total_transaction'])
        else:
            model.fit(y=data_train)
        print("After")
        print("Prediction pdmarima")
        predictions = model.predict(len(data_test))
        predictions.name = 'predictions'
        predictions = pd.concat([data_test, predictions], axis=1)
        print(predictions.head(4))

    except Exception as e:
        print("Error pdmarima can't fit the model", e)
        predictions = pd.DataFrame(index=data_test.index, columns=['predictions'])
        exit()

    if exog:
        predictions = predictions.drop(['total_transaction'], axis=1)

    print(predictions.head(4))

    print(f"MAE: {mean_absolute_error(data_test['paid_rate'], predictions['predictions'])}")

    if plot:
        name = 'pred_arima'
        if exog:
            name = f'{name}_exog'
        
        # fig, ax = plt.subplots(figsize=(20, 10))
        # predictions.plot(ax=ax, label='pred')
        # ax.set_title('Predictions with ARIMA models')
        # ax.legend()
        # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        # plt.autoscale(enable=True, axis='x', tight=True)
        # plt.tight_layout()
        # plt.savefig(f'{name}.png')
        # plt.show()


        # create a plot for every 3 days
        for i in range(1, 29, 3):
            fig, ax = plt.subplots(figsize=(20, 10))
            predictions.loc[f'2023-07-{i}':f'2023-07-{i+2}'].plot(ax=ax, label='pred')
            ax.set_title(f'Predictions with ARIMA models for the {i}th to {i+2}th july')
            ax.legend()
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.tight_layout()
            plt.savefig(f'{name}_{i}.png')
            plt.show()

        # fig, ax = plt.subplots(figsize=(20, 10))
        # predictions.loc['2023-09-18':'2023-09-21'].plot(ax=ax, label='pred')
        # ax.set_title('Predictions with ARIMA models')
        # ax.legend()
        # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        # plt.autoscale(enable=True, axis='x', tight=True)
        # plt.tight_layout()
        # plt.savefig(f'{name}_specific.png')
        # plt.show()


def sarima_statsmodels():
    # ARIMA model with statsmodels.Sarimax
    # ==============================================================================
    warnings.filterwarnings("ignore", category=UserWarning, message='Non-invertible|Non-stationary')
    model = SARIMAX(endog = data_train, order = (p,d,q), seasonal_order = (P,D,Q,s))
    model_res = model.fit(disp=0)
    warnings.filterwarnings("default")

    # print(model_res.summary())

    # Prediction
    # ==============================================================================
    predictions_statsmodels = model_res.get_forecast(steps=len(data_test)).predicted_mean
    predictions_statsmodels.name = 'predictions_statsmodels'
    # compare prediction to real data
    predictions_statsmodels = pd.concat([data_test, predictions_statsmodels], axis=1)
    print(predictions_statsmodels.head(4))

    # # Plot
    # # ==============================================================================
    # fig, ax = plt.subplots(figsize=(20, 10))
    # # data_train.plot(ax=ax, label='train')
    # data_test.plot(ax=ax, label='test')
    # predictions_statsmodels.predictions_statsmodels.plot(ax=ax, label='SARIMA')
    # ax.legend()
    # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    # plt.autoscale(enable=True, axis='x', tight=True)
    # plt.show()

    return predictions_statsmodels


def sarima_skforecast():
    # ARIMA model with skforecast.Sarimax
    # ==============================================================================
    warnings.filterwarnings("ignore", category=UserWarning, message='Non-invertible|Non-stationary')

    model = Sarimax(order=(p,d,q), seasonal_order=(P,D,Q,s))
    model.fit(y=data_train)

    warnings.filterwarnings("default")

    # Prediction
    # ==============================================================================
    predictions_skforecast = model.predict(steps=len(data_test))
    predictions_skforecast.columns = ['predictions_skforecast']
    # compare prediction to real data
    predictions_skforecast = pd.concat([data_test, predictions_skforecast], axis=1)
    print(predictions_skforecast.head(4))


    # # Plot
    # # ==============================================================================
    # fig, ax = plt.subplots(figsize=(20, 10))
    # # data_train.plot(ax=ax, label='train')
    # data_test.plot(ax=ax, label='test')
    # predictions_skforecast.skforecast.plot(ax=ax, label='SARIMA')
    # ax.legend()
    # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    # plt.autoscale(enable=True, axis='x', tight=True)
    # plt.show()

    return predictions_skforecast


def sarima_pdmarima():
    try:
        model = ARIMA(order=(p,d,q), seasonal_order=(P,D,Q,s))
        model.fit(y=data_train)
        print("Prediction pdmarima")
        predictions_pdmarima = model.predict(len(data_test))
        predictions_pdmarima.name = 'predictions_pdmarima'
        predictions_pdmarima = pd.concat([data_test, predictions_pdmarima], axis=1)
        print(predictions_pdmarima.head(4))

    except:
        print("Error pdmarima can't fit the model")
        predictions_pdmarima = pd.DataFrame(index=data_test.index, columns=['predictions_pdmarima'])

    return predictions_pdmarima


# pred_statsmodels = sarima_statsmodels()['predictions_statsmodels']
# pred_skforecast = sarima_skforecast()['predictions_skforecast']
# pred_pdmarima = sarima_pdmarima()['predictions_pdmarima']

# print(f"MAE skforecast: {mean_absolute_error(data_test, pred_skforecast)}")
# print(f"MAE pdmarima: {mean_absolute_error(data_test, pred_pdmarima)}")


def plot_predictions():
    # compare the different models predictions
    # predictions = pd.concat([data_test, pred_statsmodels, pred_skforecast, pred_pdmarima], axis=1)
    predictions = pd.concat([data_test, pred_skforecast, pred_pdmarima], axis=1)
    print(predictions.head(20))

    # Plot predictions
    # ==============================================================================
    fig, ax = plt.subplots(figsize=(20, 10))
    data_test.plot(ax=ax, label='test')
    # pred_statsmodels.plot(ax=ax, label='statsmodels', linestyle='dashdot')
    pred_skforecast.plot(ax=ax, label='skforecast', linestyle='dashed')
    pred_pdmarima.plot(ax=ax, label='pmdarima', linestyle='dotted')
    ax.set_title('Predictions with ARIMA models')
    ax.legend()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.tight_layout()
    plt.savefig('pred_arima.png')
    plt.show()

    # plot only data from the 18th to 20th september
    fig, ax = plt.subplots(figsize=(20, 10))
    data_test.loc['2023-09-18':'2023-09-21'].plot(ax=ax, label='test')
    # pred_statsmodels.loc['2023-09-18':'2023-09-21'].plot(ax=ax, label='statsmodels', linestyle='dashdot')
    pred_skforecast.loc['2023-09-18':'2023-09-21'].plot(ax=ax, label='skforecast', linestyle='dashed')
    pred_pdmarima.loc['2023-09-18':'2023-09-21'].plot(ax=ax, label='pmdarima', linestyle='dotted')
    ax.set_title('Predictions with ARIMA models')
    ax.legend()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.tight_layout()
    plt.savefig('pred_arima_specific.png')
    plt.show()


def sarima_plot_3days(data_train, data_test, month_name):
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
    try:
        model = ARIMA(order=(p,d,q), seasonal_order=(P,D,Q,s))
        model.fit(y=data_train)
        predictions = model.predict(len(data_test))
        predictions.name = 'predictions'
        predictions = pd.concat([data_test, predictions], axis=1)
        print(predictions.head(4))

    except Exception as e:
        print("Error pdmarima can't fit the model", e)
        predictions = pd.DataFrame(index=data_test.index, columns=['predictions'])
        exit()

    print(f"MAE: {mean_absolute_error(data_test['paid_rate'], predictions['predictions'])}")

    # save the data into a csv file
    # predictions.to_csv(f'./predictions_{month_name}.csv')
    
    name = 'pred_arima'
    days = days_in_month[months.index(month_name)] - 1

    # create a plot for every 3 days
    for i in range(1, days, 3):
        fig, ax = plt.subplots(figsize=(20, 10))
        predictions.loc[f'2023-{(months.index(month_name)+1):02}-{i}':f'2023-{(months.index(month_name)+1):02}-{i+2}'].plot(ax=ax, label='pred')
        ax.set_title(f'Predictions with ARIMA models for the {i}th to {i+2}th {month_name}')
        ax.legend()
        # set the y axis to be between 0 and 100
        # ax.set_ylim(0, 100)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.tight_layout()
        plt.savefig(f'{name}_{month_name}_{i}.png')
        plt.show()



def analyse_2years_data(fname):
    data = pd.read_csv(fname)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')
    data.index = pd.DatetimeIndex(data.index.values, freq=data.index.inferred_freq)
    
    data = data['percentage']
    
    data_diff_1 = data.diff().dropna()
    data_diff_2 = data_diff_1.diff().dropna()

    test_stationarity(data, data_diff_1, data_diff_2)
    plot_acf_pacf(data, data_diff_1, savefig=True)
    plot_decomposition(data, data_diff_1)
    adfuller_test(data)

def grid_search_2years(fname, the_p_value, output_folder, special=False):
    data = pd.read_csv(fname)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')
    data.index = pd.DatetimeIndex(data.index.values, freq=data.index.inferred_freq)
    
    data = data['percentage']

    data_train = data.loc['2021-12-01':'2023-07-31'] # 2021-12-01 -> 2023-07-31 = 608 days (1 year and 8 months)
    data_test = data.loc['2023-08-01':] # 2023-08-01 -> 2023-11-28 = 120 days (4 months and 28 days)

    # print(len(data_train))
    # print(len(data_test))

    backup_print = ""

    possibilities = [2,1,0] # [1, 0]

    for p in [the_p_value]: # should be changed after each run
        for d in [1,0]:
            for q in possibilities:
                for P in possibilities:
                    for D in [1,0]:
                        for Q in possibilities:
                            # if special:
                            #     if (q != 2 and P != 2 and  Q != 2):
                            #         continue
                            # else:
                            #     if (q == 2 and P != 2 and  Q != 2) or (q != 2 and P == 2 and  Q != 2) or (q != 2 and P != 2 and  Q == 2) or (q == 2 and P == 2 and  Q == 2):
                            #         continue
                            try:
                                start_time = time.time()
                                model = ARIMA(order=(p,d,q), seasonal_order=(P,D,Q,24))
                                model.fit(y=data_train)
                                predictions_pdmarima = model.predict(len(data_test))
                                predictions_pdmarima.name = 'predictions_pdmarima'
                                predictions_pdmarima = pd.concat([data_test, predictions_pdmarima], axis=1)

                                # print the MSE and AIC
                                the_text = f"p={p}, d={d}, q={q}, P={P}, D={D}, Q={Q}, AIC={model.aic()}, MSE={mean_absolute_error(data_test, predictions_pdmarima['predictions_pdmarima'])}, Time={time.time() - start_time:.2f}s"
                                print(the_text)
                                backup_print += the_text + "\n"

                            except Exception as e:
                                the_text = f"p={p}, d={d}, q={q}, P={P}, D={D}, Q={Q}, AIC={float('inf')}, MSE={float('inf')}, Time={2**31-1}s\n {e}"
                                print(the_text)
                                backup_print += the_text + "\n"

    # save the output to a file
    with open(f'gridsearch/{output_folder}/output_gridsearch_p{the_p_value}.txt', 'w') as f:
        f.write(backup_print)



def plot_by_month(pred, month_name):
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 28, 31] # november isn't complete in the dataset so we take 28 days
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
    
    month_nb = months.index(month_name) + 1
    days = days_in_month[month_nb - 1]
    
    plt.figure(figsize=(20, 10))
    plt.plot(pred.loc[f'2023-{month_nb:02}-01':f'2023-{month_nb:02}-{days:02}']['percentage'], label='test')
    plt.plot(pred.loc[f'2023-{month_nb:02}-01':f'2023-{month_nb:02}-{days:02}']['predictions'], label='pred')
    plt.title(f'Predictions with ARIMA models for the month of {month_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'pred_arima_{month_name}.png')
    plt.show()

def sarima_2years(fname):

    p,d,q,P,D,Q = 1,1,1,0,1,1

    data = pd.read_csv(fname)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')
    data.index = pd.DatetimeIndex(data.index.values, freq=data.index.inferred_freq)
    
    data = data['percentage']

    data_train = data.loc['2021-12-01':'2023-06-30'] # 2021-12-01 -> 2023-06-30 = 577 days (1 year and 7 months)
    data_test = data.loc['2023-07-01':'2023-11-28'] # 2023-07-01 -> 2023-11-28 = 151 days (5 months and 28 days)

    if os.path.isfile('model.pkl'):
        with open('model.pkl', 'rb') as pkl:
            model = pickle.load(pkl)
    else:
        timestart = time.time()    
        model = ARIMA(order=(p,d,q), seasonal_order=(P,D,Q,24))
        model.fit(y=data_train)
        print(f"Time to fit the model: {time.time() - timestart:.2f}s")
        with open('model.pkl', 'wb') as pkl:
            pickle.dump(model, pkl)

    predictions = model.predict(len(data_test))
    predictions.name = 'predictions'
    predictions = pd.concat([data_test, predictions], axis=1)

    mse = mean_absolute_error(data_test, predictions['predictions'])
    print(f"MAE: {mse}")

    # plot_by_month(predictions, 'august')
    # plot_by_month(predictions, 'september')
    # plot_by_month(predictions, 'october')
    # plot_by_month(predictions, 'november')

    # # create a plot for every 3 days
    # for i in range(1, 31, 3):
    #     fig, ax = plt.subplots(figsize=(20, 10))
    #     predictions.loc[f'2023-01-{i:02}':f'2023-01-{i+2:02}'].plot(ax=ax, label='pred')
    #     ax.set_title(f'Predictions with ARIMA models for the {i}th to {i+2}th january')
    #     ax.legend()
    #     plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    #     plt.autoscale(enable=True, axis='x', tight=True)
    #     plt.tight_layout()
    #     plt.savefig(f'pred_january_{i}.png')
    #     plt.show()

    return predictions



def compute_score(anomalies, predictions):
    # test if predictions is empty
    if len(predictions) == 0:
        return len(anomalies)

    # check the value 'predictions' of the predictions dataframe to verify that the point is an anomaly or not (check if -1 for anomalies))
    tp = []; fp = []; fn = []; tn = []
    score = 0
    # compute the true positive and false positive
    for index, row in anomalies.iterrows():
        try :
            if predictions.loc[index, 'predictions'] == -1:
                tp.append(index)
            else:
                fp.append(index)
        except:
            fp.append(index)
    
    # compute the score
    for fpfp in fp: # check if the false positive is between 1am and 6am -> count 2.5
        if fpfp.hour >= 1 and fpfp.hour <= 6:
            score += 2.5
        else:
            score += 1
    
    # compute the false negative and true negative
    for index, row in predictions.iterrows():
        if row['predictions'] == -1 and index not in tp:
            fn.append(index)
    score += len(fn)

    return score

def label_prediction_as_anomalies(predictions, threshold=-3.5):
    my_predictions = predictions.copy()
    my_predictions['residuals'] = my_predictions['percentage'] - my_predictions['predictions']
    my_predictions['predictions'] = my_predictions['residuals'].apply(lambda x: -1 if x < threshold else 1)

    # if the anomaly is between 1am and 6am -> set the value to 1
    for index, row in my_predictions.iterrows():
        if row['predictions'] == -1 and index.hour >= 1 and index.hour <= 6:
            # in 0.1% of the cases their is truly an anomaly between 1am and 6am else it's a false positive
            if not random.random() < 0.01:
                my_predictions.loc[index, 'predictions'] = 1

    print(my_predictions['predictions'].value_counts())
    # print(my_predictions[my_predictions['predictions'] == -1])

    # count the values that are considered as anomalies but that have a percentage > 70
    # print(my_predictions[(my_predictions['predictions'] == -1) & (my_predictions['percentage'] > 70)])

    return my_predictions



def open_file(file):
    # create a dataframe that will contains the data from the file
    data_pd = pd.DataFrame(columns=['p', 'd', 'q', 'P', 'D', 'Q', 'AIC', 'MSE', 'Time'])
    with open(file, 'r') as f:
        lines = f.readlines()
        # remove empty lines
        lines = [line for line in lines if line.strip()]
        # remove the '\n' at the end of each line
        lines = [line.strip() for line in lines]
        for line in lines:
            # example of a line :
            # p=0, d=0, q=2, P=1, D=0, Q=1, AIC=36352.21971120763, MSE=8.406316710696737, Time=15.73s
            line_arr = line.split(', ')
            # remove the name and the '='
            line_arr = [line.split('=')[1] for line in line_arr]
            # remove the 's' at the end of the time
            line_arr[-1] = line_arr[-1][:-1]
            print(line_arr)
            try:
                # add the line to the dataframe
                data_pd.loc[len(data_pd)] = line_arr
            except:
                print("Error with the line : ", line)
                exit()

    # convert the values to the correct type
    data_pd['p'] = data_pd['p'].astype(int)
    data_pd['d'] = data_pd['d'].astype(int)
    data_pd['q'] = data_pd['q'].astype(int)
    data_pd['P'] = data_pd['P'].astype(int)
    data_pd['D'] = data_pd['D'].astype(int)
    data_pd['Q'] = data_pd['Q'].astype(int)
    data_pd['AIC'] = data_pd['AIC'].astype(float)
    data_pd['MSE'] = data_pd['MSE'].astype(float)
    data_pd['Time'] = data_pd['Time'].astype(float)

    # return the dataframe
    return data_pd

def grid_search_analysis(folder):

    # files = os.listdir(folder)

    # global_data = pd.DataFrame(columns=['p', 'd', 'q', 'P', 'D', 'Q', 'AIC', 'MSE', 'Time', 'score'])
    # for f in files:
    #     data_spec = open_file(f'{folder}/{f}')
    #     global_data = pd.concat([global_data, data_spec])

    # # remove the duplicates that have the same p, d, q, P, D, Q
    # global_data = global_data.drop_duplicates(subset=['p', 'd', 'q', 'P', 'D', 'Q'])

    # # save the dataframe to a csv file
    # global_data.to_csv(f'{folder}/output_gridsearch.csv', index=False)

    data = pd.read_csv(f'{folder}/output_gridsearch.csv')
    data = data[data['AIC'] != -1]
    data = data.drop(['score'], axis=1)

    print(f"SCORE {folder}")

    data = data.sort_values(by=['AIC'])
    print("AIC")
    print(data.head(10))

    data = data.sort_values(by=['MSE'])
    print("MSE")
    print(data.head(10))

    weight_aic = 0.7
    weight_mse = 0.3

    # get the max aic that is not inf
    max_aic = data['AIC'].max() if data['AIC'].max() != float('inf') else data[data['AIC'] != float('inf')]['AIC'].max()
    max_mse = data['MSE'].max() if data['MSE'].max() != float('inf') else data[data['MSE'] != float('inf')]['MSE'].max()
    # normalize the data so the AIC and MSE to be between 0 and 1
    norm_aic = (data['AIC'] - data['AIC'].min()) / (max_aic - data['AIC'].min())
    norm_mse = (data['MSE'] - data['MSE'].min()) / (max_mse - data['MSE'].min())
    
    data['score'] = weight_aic * norm_aic + weight_mse * norm_mse

    data = data.sort_values(by=['score'])
    print(f"{weight_aic*100:00}% AIC {weight_mse*100:00}% MSE")
    print(data.head(10))
    
    ### RESULT ###
    #      p  d  q  P  D  Q           AIC       MSE    Time         score
    # 148  1  0  2  0  1  1  25629.983562  2.201917   54.37  12816.092740
    # 238  2  0  1  0  1  1  25631.249118  2.191278   55.40  12816.720198
    # 149  1  0  2  0  1  2  25631.299778  2.189466  116.34  12816.744622
    # 154  1  0  2  1  1  1  25631.384019  2.189653   68.57  12816.786836
    # 239  2  0  1  0  1  2  25632.707111  2.194291  138.97  12817.450701
    # 160  1  0  2  2  1  1  25633.072605  2.190785  141.71  12817.631695
    # 155  1  0  2  1  1  2  25633.254862  2.191324  124.73  12817.723093
    # 245  2  0  1  1  1  2  25634.087578  2.191055  168.59  12818.139316
    # 244  2  0  1  1  1  1  25634.105795  2.197561   63.90  12818.151678
    # 263  2  0  2  1  1  2  25634.272720  2.190599  181.97  12818.231660



def load_anomalies_nicely(fname):
    ANOMALIES_START_DATE = '2023-07-01 00:00:00'
    ANOMALIES_END_DATE = '2023-11-28 23:59:59'
    # Load anomalies dataframe
    anomalies = pd.read_csv(fname)
    anomalies['timestamp'] = pd.to_datetime(anomalies['timestamp'])
    anomalies = anomalies.set_index('timestamp')
    anomalies.index = pd.DatetimeIndex(anomalies.index.values, freq=anomalies.index.inferred_freq)
    anomalies = anomalies[ANOMALIES_START_DATE:ANOMALIES_END_DATE]
    anomalies = anomalies[anomalies['status'] == True]
    anomalies['status'] = -1
    # Create non-anomalies dataframe
    # non_anomalies = pd.DataFrame(pd.date_range(ANOMALIES_START_DATE, ANOMALIES_END_DATE, freq='H'), columns=['timestamp'])
    # non_anomalies = non_anomalies.set_index('timestamp')
    # non_anomalies.index = pd.DatetimeIndex(non_anomalies.index.values, freq=non_anomalies.index.inferred_freq)
    # non_anomalies = non_anomalies.drop(anomalies.index)
    # non_anomalies['status'] = False
    # # Concatenate anomalies and non-anomalies
    # anomalies = pd.concat([anomalies, non_anomalies])
    anomalies.sort_index(inplace=True)
    anomalies.rename(columns={'status': 'predictions'}, inplace=True)
    # print(anomalies['predictions'].value_counts())
    print(anomalies)

    return anomalies





# analyse_2years_data('./data/2years_ogone_noise_reduction.csv')
# analyse_2years_data('./data/2years_az_noise_reduction.csv')
# analyse_2years_data('./data/2years_datatrans_noise_reduction.csv')

# grid_search_2years('./data/2years_ogone_noise_reduction.csv', 0, special=True)
# grid_search_2years('./data/2years_ogone_noise_reduction.csv', 0)
# grid_search_2years('./data/2years_ogone_noise_reduction.csv', 1)
# grid_search_2years('./data/2years_ogone_noise_reduction.csv', 2)

# grid_search_2years('./data/2years_az_noise_reduction.csv', 0, '2years_az')
# grid_search_2years('./data/2years_az_noise_reduction.csv', 1, '2years_az')
# grid_search_2years('./data/2years_az_noise_reduction.csv', 2, '2years_az')

# grid_search_2years('./data/2years_datatrans_noise_reduction.csv', 0, '2years_datatrans')
# grid_search_2years('./data/2years_datatrans_noise_reduction.csv', 1, '2years_datatrans')
# grid_search_2years('./data/2years_datatrans_noise_reduction.csv', 2, '2years_datatrans')

anomalies = load_anomalies_nicely('./data/anomalies_ogone.csv')
pred = sarima_2years('./data/2years_ogone_noise_reduction.csv')


# start = 1
# end = 15
# scores = []
# for i in np.arange(start, end, 0.1):
#     the_pred = label_prediction_as_anomalies(pred, -i)
#     score = compute_score(anomalies, the_pred)
#     print(f"Threshold : {i} -> Score: {score}")
#     print(the_pred[the_pred['predictions'] == -1])
#     scores.append(score)
#     print()

# plt.figure(figsize=(20, 10))
# plt.plot(np.arange(start, end, 0.1), scores, label='test')
# plt.title(f'Score of the SARIMA model for different residual thresholds')
# plt.xlabel('Residual threshold')
# plt.ylabel('Score')
# plt.legend()
# plt.tight_layout()
# plt.savefig(f'score_sarima_by_threshold.png')
# plt.show()


the_pred = label_prediction_as_anomalies(pred, -6)
print(the_pred[the_pred['predictions'] == -1])
score = compute_score(anomalies, the_pred)
print(f"Threshold : -6 -> Score: {score}")


# grid_search_analysis('./gridsearch/2years_ogone')
# grid_search_analysis('./gridsearch/2years_az')
# grid_search_analysis('./gridsearch/2years_datatrans')

exit()



fn_global = './data/data_ogone_norm_global.csv'
fn_morning = './data/data_ogone_norm_morning.csv'

exog = False

data = load_data(fn_global, exog=exog)

data_train = data.loc[:'2023-04-30']
data_test = data.loc['2023-05-01':'2023-05-31']
# # print(data_train.head(10))

# data_diff_1 = data_train.diff().dropna()
# data_diff_2 = data_diff_1.diff().dropna()

# plot_acf_pacf(data_train, data_diff_1, savefig=True)
# exit()

# forecaster = sarima_forecaster(data_train, data_test, exog=exog, plot=True)
# sarima_other(data_train, data_test, exog=exog, plot=True)
# forecaster.save('sarima_model_skforecast.pkl')



# sarima_plot_3days(data_train, data_test, 'may')

# exit()

def backtesting():
    forecaster = ForecasterSarimax(
                    regressor=Sarimax(
                                    order=(p,d,q),
                                    seasonal_order=(P,D,Q,s),
                                    maxiter=200
                                )
                )

    data_series = data['paid_rate']
    data_series = data_series.squeeze()
    data_series_train = data_series.loc[:'2023-08-31']

    metric, predictions = backtesting_sarimax(
                            forecaster            = forecaster,
                            y                     = data_series,
                            initial_train_size    = len(data_series_train),
                            fixed_train_size      = False,
                            steps                 = 24,
                            metric                = 'mean_absolute_error',
                            refit                 = True,
                            n_jobs                = "auto",
                            suppress_warnings_fit = True,
                            verbose               = True,
                            show_progress         = True
                        )

    print(f"Metric (mean_absolute_error): {metric}")
    # print(predictions.head(4))

    # Metric (mean_absolute_error): 1.904687096398027

    # Plot backtest predictions
    # ==============================================================================
    fig, ax = plt.subplots(figsize=(6, 3))
    data.loc['2023-09-01':].plot(ax=ax, label='test')
    predictions.plot(ax=ax)
    ax.set_title('Backtest predictions with SARIMAX model')
    ax.legend()
    plt.tight_layout()
    plt.show()


end_train = '2023-06-30'
start_val = '2023-07-01'
end_val = '2023-08-31'
start_test = '2023-09-01'

def grid_search_bad():
    # data = data['paid_rate']
    # data = data.squeeze()

    # data_train = data.loc[:end_train]


    # print(
    #     f"Train dates      : {data.index.min()} --- {data.loc[:end_train].index.max()}  "
    #     f"(n={len(data.loc[:end_train])})"
    # )
    # print(
    #     f"Validation dates : {data.loc[end_train:].index.min()} --- {data.loc[:end_val].index.max()}  "
    #     f"(n={len(data.loc[end_train:end_val])})"
    # )
    # print(
    #     f"Test dates       : {data.loc[end_val:].index.min()} --- {data.index.max()}  "
    #     f"(n={len(data.loc[end_val:])})"
    # )

    # fig, ax = plt.subplots(figsize=(7, 3))
    # data.loc[:end_train].plot(ax=ax, label='train')
    # data.loc[end_train:end_val].plot(ax=ax, label='validation')
    # data.loc[end_val:].plot(ax=ax, label='test')
    # ax.set_title('Monthly fuel consumption in Spain')
    # ax.legend()
    # plt.tight_layout()
    # plt.show()

    # # Grid search based on backtesting
    # # ==============================================================================
    # forecaster = ForecasterSarimax(
    #                  regressor=Sarimax(order=(1, 1, 1), maxiter=500), # Placeholder replaced in the grid search
    #              )

    # param_grid = {
    #     'order': [(0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1)],
    #     'seasonal_order': [(0, 0, 0, 0), (0, 1, 0, 24), (1, 1, 1, 24)],
    #     'trend': [None, 'n', 'c']
    # }

    # results_grid = grid_search_sarimax(
    #                    forecaster            = forecaster,
    #                    y                     = data.loc[:end_val],
    #                    param_grid            = param_grid,
    #                    steps                 = 24,
    #                    refit                 = False,
    #                    metric                = 'mean_absolute_error',
    #                    initial_train_size    = len(data_train),
    #                    fixed_train_size      = False,
    #                    return_best           = False,
    #                    n_jobs                = 'auto',
    #                    suppress_warnings_fit = True,
    #                    verbose               = False,
    #                    show_progress         = True
    #                )

    # print(results_grid.head(5))

    ### RESULT ###
    #                                                params  mean_absolute_error      order seasonal_order trend
    # 42  {'order': (2, 1, 1), 'seasonal_order': (1, 1, ...             2.068286  (2, 1, 1)  (1, 1, 1, 24)  None
    # 43  {'order': (2, 1, 1), 'seasonal_order': (1, 1, ...             2.068286  (2, 1, 1)  (1, 1, 1, 24)     n
    # 34  {'order': (1, 1, 1), 'seasonal_order': (1, 1, ...             2.068829  (1, 1, 1)  (1, 1, 1, 24)     n
    # 33  {'order': (1, 1, 1), 'seasonal_order': (1, 1, ...             2.068829  (1, 1, 1)  (1, 1, 1, 24)  None
    # 35  {'order': (1, 1, 1), 'seasonal_order': (1, 1, ...             2.099228  (1, 1, 1)  (1, 1, 1, 24)     c


    # # Capture auto_arima trace in a pandas dataframe
    # # ==============================================================================
    # buffer = StringIO()
    # with contextlib.redirect_stdout(buffer):
    #     auto_arima(
    #             y                 = data.loc[:end_val],
    #             start_p           = 0,
    #             start_q           = 0,
    #             max_p             = 3,
    #             max_q             = 3,
    #             seasonal          = True,
    #             test              = 'adf',
    #             m                 = 24, # Seasonal period
    #             d                 = None, # The algorithm will determine 'd'
    #             D                 = None, # The algorithm will determine 'D'
    #             trace             = True,
    #             error_action      = 'ignore',
    #             suppress_warnings = True,
    #             stepwise          = True
    #         )
    # trace_autoarima = buffer.getvalue()
    # pattern = r'ARIMA\((\d+),(\d+),(\d+)\)\((\d+),(\d+),(\d+)\)\[(\d+)\]\s+(intercept)?\s+:\s+AIC=([\d\.]+), Time=([\d\.]+) sec'
    # matches = re.findall(pattern, trace_autoarima)
    # results = pd.DataFrame(matches, columns=['p', 'd', 'q', 'P', 'D', 'Q', 'm', 'intercept', 'AIC', 'Time'])
    # results['order'] = results[['p', 'd', 'q']].apply(lambda x: f"({x[0]},{x[1]},{x[2]})", axis=1)
    # results['seasonal_order'] = results[['P', 'D', 'Q', 'm']].apply(lambda x: f"({x[0]},{x[1]},{x[2]},{x[3]})", axis=1)
    # results = results[['order', 'seasonal_order', 'intercept', 'AIC', 'Time']]

    # results = sorted(results.values.tolist(), key=lambda x: x[3])
    # results = pd.DataFrame(results, columns=['order', 'seasonal_order', 'intercept', 'AIC', 'Time'])

    # print(results.head(15))

    ### RESULT ###
    #      order seasonal_order  intercept        AIC    Time
    # 0  (1,0,0)     (2,0,0,24)  intercept  30905.728  126.70
    # 1  (1,0,0)     (1,0,0,24)  intercept  30945.505   11.07
    # 2  (1,0,0)     (0,0,0,24)  intercept  31177.671    0.13
    # 3  (0,0,1)     (0,0,1,24)  intercept  31323.331    2.30
    # 4  (0,0,0)     (0,0,0,24)  intercept  32904.217    0.06
    # 5  (0,0,1)     (2,0,0,24)  intercept  39451.645   77.70
    # 6  (0,0,0)     (0,0,0,24)             67225.746    0.04
    # model = auto_arima(order=(p,d,q), seasonal_decompose=(P,D,Q,s), y=data_train)
    # model.fit(data_train)
    # forecast = model.predict(n_periods=len(data[start_val:end_val]))
    # forecast = pd.DataFrame(forecast,index = data[start_val:end_val].index,columns=['Prediction'])
    # print(model.aic())
    # 23496.291621331176

    ## Je comprends pas pourquoi il ne fait pas tout les modèles jusqu'a (3,3,3) 
    # Car avec order=(1,1,1) et seasonal_order=(1,1,1,24) on a un AIC de 23496.291621331176
    pass

def launch_grid_search(data):
    data_train = data.loc[:'2023-07-31'] # 7 months
    data_test = data.loc['2023-08-01':] # 2 months
    # create a grid search manually
    # for p in [2]:#range(0, 3):
    #     for d in range(0, 2):
    #         for q in range(0, 3):
    #             for P in range(0, 3):
    #                 for D in range(0, 2):
    #                     for Q in range(0, 3):
    #                         try:
    #                             start_time = time.time()
    #                             model = ARIMA(order=(p,d,q), seasonal_order=(P,D,Q,s))
    #                             model.fit(y=data_train)
    #                             predictions_pdmarima = model.predict(len(data_test))
    #                             predictions_pdmarima.name = 'predictions_pdmarima'
    #                             predictions_pdmarima = pd.concat([data_test, predictions_pdmarima], axis=1)

    #                             # print the MSE and AIC
    #                             print(f"p={p}, d={d}, q={q}, P={P}, D={D}, Q={Q} - AIC: {model.aic()} - MSE: {mean_absolute_error(data_test['paid_rate'], predictions_pdmarima['predictions_pdmarima'])} - Time: {time.time() - start_time:.2f}s")

    #                         except:
    #                             print(f"Error pdmarima can't fit the model. p={p}, d={d}, q={q}, P={P}, D={D}, Q={Q}")



# from sklearn.model_selection import TimeSeriesSplit
# # train a model with this 2 possible parameters
# # 1  0  2  0  1  1
# # 2  0  1  0  1  1

# tscv = TimeSeriesSplit(n_splits=5)

# for train_idx, test_idx in tscv.split(data['paid_rate']):
#     train_data = data.iloc[train_idx]['paid_rate']
#     test_data = data.iloc[test_idx]['paid_rate']

#     print(f"Train dates      : {train_data.index.min()} --- {train_data.index.max()}  "
#           f"(n={len(train_data)})")
#     print(f"Test dates       : {test_data.index.min()} --- {test_data.index.max()}  "
#             f"(n={len(test_data)})")

#     model = ARIMA(order=(1,1,1), seasonal_order=(1,1,1,24))
#     model.fit(y=train_data)

#     predictions = model.predict(len(test_data))

#     print(f"MAE: {mean_absolute_error(test_data, predictions)}")

