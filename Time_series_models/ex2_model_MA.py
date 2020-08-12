import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import pearsonr

from datetime import datetime, timedelta
register_matplotlib_converters()


## generate some data
errors = np.random.normal(0, 1, 400)
date_index = pd.date_range(start='9/1/2019', end='1/1/2020')
mu = 50
series = []
for t in range(1, len(date_index)+1):
    series.append(mu+0.4*errors[t-1]+0.3*errors[t-2]+errors[t])

series = pd.Series(series, index=date_index)
series.head()

# freq
series = series.asfreq(pd.infer_freq(series.index))
series.head()

# plot
plt.figure(figsize=(10, 4))
plt.plot(series)
plt.axhline(mu, linestyle='--', color='k')

## calculate correlation
def calc_corr(series, lag):
    return pearsonr(series[:-lag], series[lag:])

## ACF
num_lags = 10
acf_vals = acf(series, nlags=num_lags)
plt.bar(range(num_lags), acf_vals[:num_lags])
acf_vals

acf_vals1 = calc_corr(series, 1)
acf_vals1
acf_vals2 = calc_corr(series, 2)
acf_vals2


## PACF
pacf_vals = pacf(series, nlags=10)
plt.bar(range(num_lags), pacf_vals[:num_lags])
pacf_vals


## Get training and testing sets
train_end = datetime(2019, 12, 30)
test_end = datetime(2020,1,1)

train_data = series[:train_end]
test_data = series[train_end + timedelta(days=1):test_end]

## first ARIMA model
model = ARIMA(train_data, order=(0,0,2))

#fit
model_fit = model.fit()

#summary
print(model_fit.summary())

#predict
pred_start_date = test_data.index[0]
pred_end_date = test_data.index[-1]

predictions = model_fit.predict(start = pred_start_date, end=pred_end_date)

#residuals
residuals = test_data - predictions

#plot
plt.figure(figsize=(10,4))
plt.plot(series[-14:])
plt.plot(predictions)
plt.legend(('data', 'predictions'), fontsize=16)