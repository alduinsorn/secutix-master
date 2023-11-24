# Libraries
# ======================================================================================
import numpy as np
import pandas as pd
from io import StringIO
import contextlib
import re
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
import time

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


def load_data(fn, exog=False):
    # data = pd.read_csv('./data_ogone_norm_global.csv')
    data = pd.read_csv('./data_ogone_norm_morning.csv')
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
p,d,q = 1,0,2
P,D,Q = 0,1,1
s = 24


def test_stationarity():
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


def plot_acf_pacf():
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


def plot_decomposition():
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

def adfuller_test():
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
            predictions.loc[f'2023-09-{i}':f'2023-09-{i+2}'].plot(ax=ax, label='pred')
            ax.set_title(f'Predictions with ARIMA models for the {i}th to {i+2}th september')
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


fn_global = './data_ogone_norm_global.csv'
fn_morning = './data_ogone_norm_morning.csv'

exog = False

data = load_data(fn_global, exog=exog)

data_train = data.loc[:'2023-08-31']
data_test = data.loc['2023-09-01':]
# print(data_train.head(10))

# data_diff_1 = data_train.diff().dropna()
# data_diff_2 = data_diff_1.diff().dropna()

# forecaster = sarima_forecaster(data_train, data_test, exog=exog, plot=True)
# sarima_other(data_train, data_test, exog=exog, plot=True)
# forecaster.save('sarima_model_skforecast.pkl')
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

def launch_grid_search():
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


def grid_search_analysis():

    data1 = open_file('output_gridsearch_p0.txt')
    data2 = open_file('output_gridsearch_p1.txt')
    data3 = open_file('output_gridsearch_p2.txt')
    data = pd.concat([data1, data2, data3])
    # save the dataframe to a csv file
    data.to_csv('output_gridsearch.csv', index=False)

    data = pd.read_csv('output_gridsearch.csv')

    data = data.sort_values(by=['AIC'])
    print("AIC")
    print(data.head(10))

    data = data.sort_values(by=['MSE'])
    print("MSE")
    print(data.head(10))


    # sort the data using 50% AIC and 50% MSE
    data['score'] = (data['AIC'] + data['MSE']) / 2
    data = data.sort_values(by=['score'])
    print("SCORE")
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


from sklearn.model_selection import TimeSeriesSplit
# train a model with this 2 possible parameters
# 1  0  2  0  1  1
# 2  0  1  0  1  1

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(data['paid_rate']):
    train_data = data.iloc[train_idx]['paid_rate']
    test_data = data.iloc[test_idx]['paid_rate']

    print(f"Train dates      : {train_data.index.min()} --- {train_data.index.max()}  "
          f"(n={len(train_data)})")
    print(f"Test dates       : {test_data.index.min()} --- {test_data.index.max()}  "
            f"(n={len(test_data)})")

    model = ARIMA(order=(1,1,1), seasonal_order=(1,1,1,24))
    model.fit(y=train_data)

    predictions = model.predict(len(test_data))

    print(f"MAE: {mean_absolute_error(test_data, predictions)}")

