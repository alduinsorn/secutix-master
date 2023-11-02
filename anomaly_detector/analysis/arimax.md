# Real data: (paid rate, total transaction, incident) 

> Problème avec les incidents, c'est que sur 6500 lignes, il n'y a que 115 incidents. Donc je ne sais pas si cette information va vraiment aider le modèle.

## Yearly Data

### ARIMAX
*Sans incidents*
**(500 itérations)** (MSE)
> Best order: (21, 0, 29) - Best trend: c

**(1000 itérations)** (MSE)
> Best order: (29, 0, 9) - Best trend: c

**(1500 itérations)** (AIC)
> Best order: (17, 0, 28) - Best trend: c


*Avec incidents* 
**(500 itérations)** (MSE)
> Best order: (1, 1, 3) - Best trend: t

**(1000 itérations)** (MSE)
> Best order: (12, 1, 11) - Best trend: t

**(1500 itérations)** (AIC)
> Best order: (18, 1, 6) - Best trend: t


### SARIMAX
*Avec incidents*
**(200 itérations)**
Best order: (0, 1, 19)
Best seasonal order: (19, 0, 16, 18)
Best trend: t



### Sans incidents
#### 500 itérations
```
Size of training data: 5241 & size of testing data: 1311
Start training...
Training time: 94.94 seconds
Prediction time: 0.05 seconds
MAE: 3.16
RMSE: 4.79
AIC: 31580.37
BIC: 31928.28
HQIC: 31702.02
Mean Residuals: 0.06
Standard Deviation of Residuals: 4.79
```

#### 1000 itérations
```
Size of training data: 5241 & size of testing data: 1311
Start training...
Training time: 88.74 seconds
Prediction time: 0.04 seconds
MAE: 3.15
RMSE: 4.78
AIC: 31544.08
BIC: 31813.21
HQIC: 31638.18
Mean Residuals: 0.01
Standard Deviation of Residuals: 4.78
```

### Avec incidents
#### 500 itérations
```
Size of training data: 5241 & size of testing data: 1311
Start training...
Training time: 4.25 seconds
Prediction time: 0.02 seconds
MAE: 3.32
RMSE: 4.85
AIC: 31577.48
BIC: 31629.99
HQIC: 31595.84
Mean Residuals: 0.66
Standard Deviation of Residuals: 4.81
```

#### 1000 itérations
```
Training time: 24.46 seconds
Prediction time: 0.03 seconds
MAE: 3.09
RMSE: 4.75
AIC: 31646.19
BIC: 31823.42
HQIC: 31708.16
Mean Residuals: -0.25
Standard Deviation of Residuals: 4.74
```


