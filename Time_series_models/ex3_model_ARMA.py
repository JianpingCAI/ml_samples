import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
# from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARMA
# register_matplotlib_converters()
from time import time

## Calfish sales data


def parser(s):
    return datetime.strptime(s, '%Y-%m-%d')


#read data
catfish_sales = pd.read_csv('catfish.csv', parse_dates=[
                            0], index_col=0, squeeze=True, date_parser=parser)

#freq
catfish_sales = catfish_sales.asfreq(pd.infer_freq(catfish_sales.index))

#plot


def plot_series(series, title):
    plt.figure(figsize=(10, 4))
    plt.plot(series)
    plt.title(title, fontsize=20)
    plt.ylabel('Sales', fontsize=16)
    for year in range(start_date.year, end_date.year):
        plt.axvline(pd.to_datetime(str(year)+'-01-01'),
                    color='k', linestyle='--', alpha=0.2)
    plt.axhline(series.mean(), color='r', alpha=0.2, linestyle='--')


start_date = datetime(2000, 1, 1)
end_date = datetime(2004, 1, 1)
lim_catfish_sales = catfish_sales[start_date:end_date]
plot_series(lim_catfish_sales, 'Catfish Sales in 1000s of Pounds')


## Remove trend: first diff
first_diff = lim_catfish_sales.diff().dropna()  # or, [1:]
plot_series(first_diff, 'First Difference of Catfish Sales')


## ACF
num_lags = 20
acf_vals = acf(first_diff, nlags=num_lags)
plt.bar(range(num_lags), acf_vals[:num_lags])
# ==>Based on ACF, we should start with a MA(1) process

## PACF
pacf_vals = pacf(first_diff, nlags=num_lags)
plt.bar(range(num_lags), pacf_vals[:num_lags])
# ==>Based on PACF, we should start with a AR(4) process

## get training and testing sets
train_end = datetime(2003, 7, 1)
test_end = datetime(2004, 1, 1)

train_data = first_diff[:train_end]
test_data = first_diff[train_end+timedelta(days=1):test_end]

## ARMA model
model = ARMA(train_data, order=(4,1))

start=time()
model_fit = model.fit()
end=time()
print('Time: ', end-start)

model_fit.summary()

# prediction
pred_start_date = test_data.index[0]
pred_end_date = test_data.index[-1]

predictions = model_fit.predict(start=pred_start_date, end=pred_end_date)
residuals = test_data - predictions

plt.figure(figsize=(10, 4))
plt.plot(residuals)
plt.title('Residuals from AR Model', fontsize=20)
plt.ylabel('Error', fontsize=16)
plt.axhline(0, color='r', alpha=0.2, linestyle='--')

plt.figure(figsize=(10,4))
plt.plot(test_data)
plt.plot(predictions)
plt.legend(('Data', 'Predictions'), fontsize=16)
plt.title('First Difference of Catfish Sales', fontsize=20)
plt.ylabel('Sales', fontsize=16)

print('Root Mean Squared Error:', np.sqrt(np.mean(residuals**2)))
