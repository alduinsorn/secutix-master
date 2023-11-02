import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from mango import Tuner
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot as plt
from datetime import datetime
from pmdarima import auto_arima

MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'november', 'december']
DAYS_IN_MONTH = [31,28,31,30,31,30,31,31,30,31,30,31]

def sarimax_objective_function(args_list):
    global data_values
    global exog_values
    
    params_evaluated = []
    mse_results = []
    
    for params in args_list:
        print("Evaluating params: ", params)
        try:
            p, d, q = params['p'], params['d'], params['q']
            P, D, Q, s = params['P'], params['D'], params['Q'], params['s']
            trend = params['trend']
            
            # model = SARIMAX(data_values, exog=exog_values, order=(p, d, q), seasonal_order=(P, D, Q, s), trend=trend)
            # model = sm.tsa.SARIMAX(data_values, exog=exog_values, order=(p, d, q), seasonal_order=(P, D, Q, s), trend=trend)
            model = sm.tsa.statespace.SARIMAX(data_values, exog=exog_values, order=(p, d, q), seasonal_order=(P, D, Q, s), trend=trend)
            model_fit = model.fit(disp=False)

            mse = mean_squared_error(data_values, model_fit.fittedvalues)   
            params_evaluated.append(params)
            mse_results.append(mse)
        except Exception as e:
            params_evaluated.append(params)
            mse_results.append(1e5)
        
    return params_evaluated, mse_results

def arimax_objective_function(args_list):
    global data_values
    global exog_values
    
    params_evaluated = []
    mse_results = []
    
    for params in args_list:
        try:
            p, d, q = params['p'], params['d'], params['q']
            trend = params['trend']
            
            model = sm.tsa.ARIMAX(data_values, exog=exog_values, order=(p, d, q), trend=trend)
            model_fit = model.fit(disp=False)

            # mse = mean_squared_error(data_values, model_fit.fittedvalues)
            aic = model_fit.aic
            params_evaluated.append(params)
            # mse_results.append(mse)
            mse_results.append(aic)
        except Exception as e:
            params_evaluated.append(params)
            mse_results.append(1e5)

    return params_evaluated, mse_results

def evaluate_sarimax_models(data, exog_data, params_grid):
    models = []
    print(f"Number of models to evaluate: {len(params_grid)}")
    for i, params in enumerate(params_grid):
        print(f"Evaluating model {i+1}")

        # p,d,q,P,D,Q,s,t = params
        p,d,q,P,D,Q = params
        model = SARIMAX(data, exog=exog_data, order=(p, d, q), seasonal_order=(P, D, Q, 24))
        model_fit = model.fit(disp=False)

        mse = mean_squared_error(data, model_fit.fittedvalues)
        aic = model_fit.aic
        bic = model_fit.bic
        hqic = model_fit.hqic

        residuals = model_fit.resid
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)

        combined_metrics = (mse + aic + bic + hqic) / 4

        models.append({
            'params': params,
            'mse': mse,
            'aic': aic,
            'bic': bic,
            'hqic': hqic,
            'mean_residual': mean_residual,
            'std_residual': std_residual,
            'combined_metrics': combined_metrics
        })

    # get the 5 best models
    models = sorted(models, key=lambda x: x['combined_metrics'])[:5]

    return models

def seasonal_plots(data, fn, save=False):
    # check the seasonal decompose for our data
    result = seasonal_decompose(data['paid_rate'], model='additive', extrapolate_trend='freq')

    # Afficher les composantes : tendance, saisonnalité et résidus
    plt.figure(figsize=(16, 8))
    ax1 = plt.subplot(411)
    ax2 = plt.subplot(412)
    ax3 = plt.subplot(413)
    ax4 = plt.subplot(414)

    ax1.plot(result.observed, label='Observed', color='blue')
    ax1.legend()
    ax2.plot(result.trend, label='Trend', color='red')
    ax2.legend()
    ax3.plot(result.seasonal, label='Seasonal', color='green')
    ax3.legend()
    ax4.plot(result.resid, label='Residual', color='purple')
    ax4.legend()

    # create index for x axis
    # create a label for each x axis tick
    # labels = [f'{i+1}' for i in range(len(data))]

    # for ax in [ax1, ax2, ax3, ax4]:
    #     ax.set_xticks(x)
    #     ax.set_xticklabels(labels, rotation=90)

    # get index of the fn in MONTHS
    month_index = MONTHS.index(fn)
    for i in range(DAYS_IN_MONTH[month_index]):
        day = datetime(2023, month_index+1, i+1)
        ax1.axvline(x=day, color='black', linestyle='--')
        ax2.axvline(x=day, color='black', linestyle='--')
        ax3.axvline(x=day, color='black', linestyle='--')
        ax4.axvline(x=day, color='black', linestyle='--')

    plt.tight_layout()
    if save: plt.savefig(f'{fn}_seasonal_decompose.png')
    plt.show()

## Old data with multiple columns
# data_fn = '../database/data/month/data_ogone.csv'
# data_fn = '../database/data/data_ogone.csv'

data_fn = '../database/data/real_data_ogone.csv' # contains 3 columns: (timestamp, paid_rate, total_transaction_count)
# data_fn = '../database/data/real_data_ogone_incidents.csv' # contains 4 columns: (timestamp, paid_rate, total_transaction_count, incident)

data = pd.read_csv(data_fn)


print("convert timestamp to datetime")
data['timestamp'] = pd.to_datetime(data['timestamp'])
print("set timestamp as index")
data.set_index('timestamp', inplace=True)

# # keep only the september month
# data = data[data.index.month == 9]
# # keep only the 19th september
# data = data[data.index.day == 19]
# # save the data into a csv file
# data.to_csv('../database/data/real_data_ogone_september.csv')
# exit()

print("set frequency to hourly")
data = data.asfreq('H')
print("sort index")

# data = data.drop(columns=['timestamp'])
# data = data.drop(columns=['index'])

data_values = data['paid_rate'].values
exog_values = data['total_transaction_count'].values

# exog_values = data[data.columns[1:]].values
# exog_values = data[['total_transaction_count', 'incident']].values

print("Data loaded")

def mongo_search():
    param_space_arimax = dict(
        p=range(0, 30),
        d=range(0, 2),
        q=range(0, 30),
        # n = no trend, c = constant, t = linear, ct = constant with linear
        trend=['n', 'c', 't', 'ct']
    )

    # train_data = data[:-720]
    # test_data = data[-720:]

    # best_model = auto_arima(train_data['paid_rate'], exogenous=train_data['total_transaction_count'], stepwise=True, seasonal=True, trace=True, max_p=30, max_q=30, max_P=30, max_Q=30, m=24, max_iter=1000)

    # p, d, q = best_model.order
    # P, D, Q, s = best_model.seasonal_order

    # print(f'Meilleurs ordres ARIMA : ({p}, {d}, {q})')
    # print(f'Meilleurs ordres saisonniers SARIMA : ({P}, {D}, {Q}, {s})')

    conf_dict = dict(num_iteration=500)

    print("Starting tuning")
    # tuner = Tuner(param_space_sarimax, sarimax_objective_function, conf_dict)
    tuner = Tuner(param_space_arimax, arimax_objective_function, conf_dict)

    results = tuner.run()
    best_params = results['best_params']
    best_mse = results['best_objective']
    order = (best_params['p'], best_params['d'], best_params['q'])
    # seasonal_order = (best_params['P'], best_params['D'], best_params['Q'], best_params['s'])

    print(f"Best order: {order}")
    # print(f"Best seasonal order: {seasonal_order}")
    print(f"Best trend: {best_params['trend']}")

def sarimax_search():
    param_grid = [(p, d, q, P, D, Q, s, trend)
                  for p in range(30)
                  for d in range(2)
                  for q in range(30)
                  for P in range(30)
                  for D in range(2)
                  for Q in range(30)
                  for s in [6, 12, 18, 24]
                  for trend in ['n', 'c', 't', 'ct']]

    param_grid = [(p, d, q, P, D, Q)
                  for p in range(0, 30, 2)
                  for d in range(2)
                  for q in range(0, 30, 2)
                  for P in range(0, 30, 2)
                  for D in range(2)
                  for Q in range(0, 30, 2)]


    best_models = evaluate_sarimax_models(data_values, exog_values, param_grid)
    for i, best in enumerate(best_models):
        print(f'Model {i+1}')
        print(f'Params: {best["params"]}')
        print(f'Combined metrics: {best["combined_metrics"]}')
        # print the residuals values
        print(f'Mean residual: {best["mean_residual"]}')
        print(f'Std residual: {best["std_residual"]}')
        print('------------------')

# seasonal_plots(data[:24*31], 'january', True) # plot the first month
# seasonal_plots(data[-24*31:], 'september', True) # plot the last month



