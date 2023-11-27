# Search for the best SARIMA(X) model
## Skforecast
Avec données 'data_ogone_norm_global.csv' et valeurs exogènes
> **MAE: 1.7829637890496262**

Avec données 'data_ogone_norm_global.csv'
> MAE: 1.9435190972587164

Avec données 'data_ogone_norm_morning.csv' et valeurs exogènes
> MAE: 1.9867365675440782

Avec données 'data_ogone_norm_morning.csv'
> MAE: 2.0626078138236954

## Pdmarima
Avec données 'data_ogone_norm_global.csv' et valeurs exogènes
> MAE: 1.8880328352139106

Avec données 'data_ogone_norm_global.csv'
> MAE: 1.8880328352139106

Avec données 'data_ogone_norm_morning.csv' et valeurs exogènes
> MAE: 1.8880328352139106

Avec données 'data_ogone_norm_morning.csv'
> MAE: 1.8880328352139106

**Avec ce modèle il n'y a pas de différence entre les données avec et sans valeurs exogènes, ainsi que si on utilise les données norm_global ou norm_morning**



## Cross validation
### Parameters: 1  0  2  0  1  1 24
Train dates      : 2023-01-01 00:00:00 --- 2023-02-15 11:00:00  (n=1092)
Test dates       : 2023-02-15 12:00:00 --- 2023-04-01 23:00:00  (n=1092)
MAE: 2.147815232381058
Train dates      : 2023-01-01 00:00:00 --- 2023-04-01 23:00:00  (n=2184)
Test dates       : 2023-04-02 00:00:00 --- 2023-05-17 11:00:00  (n=1092)
MAE: 2.3821576315583095
Train dates      : 2023-01-01 00:00:00 --- 2023-05-17 11:00:00  (n=3276)
Test dates       : 2023-05-17 12:00:00 --- 2023-07-01 23:00:00  (n=1092)
/home/romain/anaconda3/lib/python3.9/site-packages/statsmodels/base/model.py:604: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
MAE: 2.082892091969942
Train dates      : 2023-01-01 00:00:00 --- 2023-07-01 23:00:00  (n=4368)
Test dates       : 2023-07-02 00:00:00 --- 2023-08-16 11:00:00  (n=1092)
MAE: 2.030102778967258
Train dates      : 2023-01-01 00:00:00 --- 2023-08-16 11:00:00  (n=5460)
Test dates       : 2023-08-16 12:00:00 --- 2023-09-30 23:00:00  (n=1092)
MAE: 2.3552500577891773

### Parameters: 2 0 1 0 1 1 24
Train dates      : 2023-01-01 00:00:00 --- 2023-02-15 11:00:00  (n=1092)
Test dates       : 2023-02-15 12:00:00 --- 2023-04-01 23:00:00  (n=1092)
/home/romain/anaconda3/lib/python3.9/site-packages/statsmodels/tsa/statespace/sarimax.py:966: UserWarning: Non-stationary starting autoregressive parameters found. Using zeros as starting parameters.
  warn('Non-stationary starting autoregressive parameters'
/home/romain/anaconda3/lib/python3.9/site-packages/statsmodels/tsa/statespace/sarimax.py:978: UserWarning: Non-invertible starting MA parameters found. Using zeros as starting parameters.
  warn('Non-invertible starting MA parameters found.'
MAE: 2.147680226931859
Train dates      : 2023-01-01 00:00:00 --- 2023-04-01 23:00:00  (n=2184)
Test dates       : 2023-04-02 00:00:00 --- 2023-05-17 11:00:00  (n=1092)
MAE: 2.3821628915005166
Train dates      : 2023-01-01 00:00:00 --- 2023-05-17 11:00:00  (n=3276)
Test dates       : 2023-05-17 12:00:00 --- 2023-07-01 23:00:00  (n=1092)
MAE: 2.0855888096472763
Train dates      : 2023-01-01 00:00:00 --- 2023-07-01 23:00:00  (n=4368)
Test dates       : 2023-07-02 00:00:00 --- 2023-08-16 11:00:00  (n=1092)
/home/romain/anaconda3/lib/python3.9/site-packages/statsmodels/tsa/statespace/sarimax.py:966: UserWarning: Non-stationary starting autoregressive parameters found. Using zeros as starting parameters.
  warn('Non-stationary starting autoregressive parameters'
/home/romain/anaconda3/lib/python3.9/site-packages/statsmodels/tsa/statespace/sarimax.py:978: UserWarning: Non-invertible starting MA parameters found. Using zeros as starting parameters.
  warn('Non-invertible starting MA parameters found.'
/home/romain/anaconda3/lib/python3.9/site-packages/statsmodels/base/model.py:604: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
MAE: 2.0302851632722696
Train dates      : 2023-01-01 00:00:00 --- 2023-08-16 11:00:00  (n=5460)
Test dates       : 2023-08-16 12:00:00 --- 2023-09-30 23:00:00  (n=1092)
MAE: 2.3563759628156604
### Parameters: 1 1 2 0 1 1 24
**Pour tester d'éviter les problèmes de convergence on rajoute une différenciation**
Train dates      : 2023-01-01 00:00:00 --- 2023-02-15 11:00:00  (n=1092)
Test dates       : 2023-02-15 12:00:00 --- 2023-04-01 23:00:00  (n=1092)
MAE: 3.0350212153678267
Train dates      : 2023-01-01 00:00:00 --- 2023-04-01 23:00:00  (n=2184)
Test dates       : 2023-04-02 00:00:00 --- 2023-05-17 11:00:00  (n=1092)
MAE: 1.9185564444384113
Train dates      : 2023-01-01 00:00:00 --- 2023-05-17 11:00:00  (n=3276)
Test dates       : 2023-05-17 12:00:00 --- 2023-07-01 23:00:00  (n=1092)
MAE: 2.5767613218744927
Train dates      : 2023-01-01 00:00:00 --- 2023-07-01 23:00:00  (n=4368)
Test dates       : 2023-07-02 00:00:00 --- 2023-08-16 11:00:00  (n=1092)
/home/romain/anaconda3/lib/python3.9/site-packages/statsmodels/base/model.py:604: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
MAE: 2.0408024311935704
Train dates      : 2023-01-01 00:00:00 --- 2023-08-16 11:00:00  (n=5460)
Test dates       : 2023-08-16 12:00:00 --- 2023-09-30 23:00:00  (n=1092)
MAE: 3.615716751194311
### Parameters: 1 1 1 1 1 1 24 (modèle trouvé par analyse visuelle)
Train dates      : 2023-01-01 00:00:00 --- 2023-02-15 11:00:00  (n=1092)
Test dates       : 2023-02-15 12:00:00 --- 2023-04-01 23:00:00  (n=1092)
MAE: 3.0342471759350893
Train dates      : 2023-01-01 00:00:00 --- 2023-04-01 23:00:00  (n=2184)
Test dates       : 2023-04-02 00:00:00 --- 2023-05-17 11:00:00  (n=1092)
MAE: 2.2762863057436626
Train dates      : 2023-01-01 00:00:00 --- 2023-05-17 11:00:00  (n=3276)
Test dates       : 2023-05-17 12:00:00 --- 2023-07-01 23:00:00  (n=1092)
/home/romain/anaconda3/lib/python3.9/site-packages/statsmodels/base/model.py:604: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
MAE: 2.453032938335804
Train dates      : 2023-01-01 00:00:00 --- 2023-07-01 23:00:00  (n=4368)
Test dates       : 2023-07-02 00:00:00 --- 2023-08-16 11:00:00  (n=1092)
MAE: 2.0763422766887887
Train dates      : 2023-01-01 00:00:00 --- 2023-08-16 11:00:00  (n=5460)
Test dates       : 2023-08-16 12:00:00 --- 2023-09-30 23:00:00  (n=1092)
MAE: 3.5256194788099418

### Conclusion
> Rien de fou dans ces valeurs, les valeurs MAE varies pas mal donc on va partir sur le premier modèle qui est le plus simple et qui obtient des résultats corrects. Le modèle le plus simple trouvé par analyse du ACF et PACF ne donne pas des résultats impressionnants comparé aux modèles trouvés par le grid search.


# September 2023 problems analysis
Le 6 vers 7h, grosse différence
Le 13 vers 12h, chute du rate mais pas de beaucoup (~5%)
Le 17 vers 13h-14h, grosse chute (~10%)
Le 19 à 9h, problème général de Ogone -> chute de 15%

                     paid_rate  predictions  residuals   z-score  total_transaction
2023-09-01 08:00:00       72.9    77.501109  -4.601109 -1.447061                369
2023-09-04 16:00:00       71.7    76.647451  -4.947451 -1.612931                794
2023-09-06 09:00:00       68.5    78.023436  -9.523436 -3.804453                672
2023-09-17 15:00:00       67.8    77.075496  -9.275496 -3.685711                974
2023-09-17 16:00:00       71.3    76.523657  -5.223657 -1.745211                902
2023-09-19 09:00:00       57.8    77.899838 -20.099838 -8.869686                770
2023-09-21 08:00:00       72.1    76.896629  -4.796629 -1.540699                426