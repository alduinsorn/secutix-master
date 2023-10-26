import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from mango import Tuner
import pandas as pd

def sarimax_objective_function(args_list):
    global data_values
    global exog_values
    
    params_evaluated = []
    mse_results = []
    
    for params in args_list:
        try:
            p, d, q = params['p'], params['d'], params['q']
            P, D, Q, s = params['P'], params['D'], params['Q'], params['s']
            trend = params['trend']
            
            model = SARIMAX(data_values, exog=exog_values, order=(p, d, q), seasonal_order=(P, D, Q, s), trend=trend)
            model_fit = model.fit(disp=False)

            mse = mean_squared_error(data_values, model_fit.fittedvalues)   
            params_evaluated.append(params)
            mse_results.append(mse)
        except Exception as e:
            params_evaluated.append(params)
            mse_results.append(1e5)
        
    return params_evaluated, mse_results

data_fn = '../database/data/data_ogone.csv'
data = pd.read_csv(data_fn)

data_values = data['paid_rate'].values
exog_values = data[data.columns[1:]].values

print("Data loaded")

param_space = dict(
    p=range(0, 30),
    d=range(0, 2),
    q=range(0, 30),
    P=range(0, 30),
    D=range(0, 2),
    Q=range(0, 30),
    s=[24],
    trend=['n', 'c', 't', 'ct']
)

conf_dict = dict(num_iteration=200)

tuner = Tuner(param_space, sarimax_objective_function, conf_dict)

results = tuner.run()
best_params = results['best_params']
best_mse = results['best_objective']
order = (best_params['p'], best_params['d'], best_params['q'])
seasonal_order = (best_params['P'], best_params['D'], best_params['Q'], best_params['s'])

print(f"Best order: {order}")
print(f"Best seasonal order: {seasonal_order}")
print(f"Best trend: {best_params['trend']}")