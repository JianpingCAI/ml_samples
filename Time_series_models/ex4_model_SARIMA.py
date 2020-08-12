import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
# from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
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


start_date = datetime(1996, 1, 1)
end_date = datetime(2000, 1, 1)
lim_catfish_sales = catfish_sales[start_date:end_date]
plot_series(lim_catfish_sales, 'Catfish Sales in 1000s of Pounds')


## Remove trend: first diff
first_diff = lim_catfish_sales.diff().dropna()  # or, [1:]
plot_series(first_diff, 'First Difference of Catfish Sales')


## ACF
num_lags = 20
acf_vals = acf(first_diff, nlags=num_lags)
plt.bar(range(num_lags), acf_vals[:num_lags])
# ==>Based on ACF, we should start with a seasonal MA process

## PACF
pacf_vals = pacf(first_diff, nlags=num_lags)
plt.bar(range(num_lags), pacf_vals[:num_lags])
# ==>Based on PACF, we should start with a seasonal AR process

## get training and testing sets
train_end = datetime(1999, 7, 1)
test_end = datetime(2000, 1, 1)

train_data = lim_catfish_sales[:train_end]
test_data = lim_catfish_sales[train_end+timedelta(days=1):test_end]
# train_data = first_diff[:train_end]
# test_data = first_diff[train_end+timedelta(days=1):test_end]

## ARMA model
my_order = (0,1,0) #remove trend 
my_seasonal_order = (1,0,1,12)
model = SARIMAX(train_data, order=my_order, seasonal_order=my_seasonal_order)

start=time()
model_fit = model.fit()
end=time()
print('Time: ', end-start)

model_fit.summary()

# prediction
# pred_start_date = test_data.index[0]
# pred_end_date = test_data.index[-1]

# predictions = model_fit.predict(start=pred_start_date, end=pred_end_date)
predictions = model_fit.forecast(len(test_data))
predictions = pd.Series(predictions, index=test_data.index)
residuals = test_data - predictions

plt.figure(figsize=(10,4))
plt.plot(residuals)
plt.axhline(0, linestyle='--', color='k')
plt.title('Residuals from SARIMA Model', fontsize=20)
plt.ylabel('Error', fontsize=16)

plt.figure(figsize=(10,4))
plt.plot(lim_catfish_sales)
plt.plot(predictions)
plt.legend(('Data', 'Predictions'), fontsize=16)
plt.title('Catfish Sales in 1000s of Pounds', fontsize=20)
plt.ylabel('Production', fontsize=16)
for year in range(start_date.year,end_date.year):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)

print('Mean Absolute Percent Error:', round(np.mean(abs(residuals/test_data)),4))
print('Root Mean Squared Error:', np.sqrt(np.mean(residuals**2)))

###########################
## Using the rolling forecast origin
rolling_preditions = test_data.copy()
rolling_preditions.head()

my_order = (0,1,0) #remove trend 
my_seasonal_order = (1,0,1,12)
for train_end in test_data.index:
    # train_data = lim_catfish_sales[:train_end] #!!! incorrect boundary
    train_data = lim_catfish_sales[:train_end-timedelta(days=1)] #!!! 
    model = SARIMAX(train_data, order=my_order, seasonal_order=my_seasonal_order)
    model_fit = model.fit()
    pred = model_fit.forecast()
    rolling_preditions[train_end]=pred

rolling_residuals = test_data - rolling_preditions

plt.figure(figsize=(10,4))
plt.plot(rolling_residuals)
plt.axhline(0, linestyle='--', color='k')
plt.title('Rolling Forecast Residuals from SARIMA Model', fontsize=20)
plt.ylabel('Error', fontsize=16)

plt.figure(figsize=(10,4))
plt.plot(lim_catfish_sales)
plt.plot(rolling_preditions)
plt.legend(('Data', 'Predictions'), fontsize=16)
plt.title('Catfish Sales in 1000s of Pounds', fontsize=20)
plt.ylabel('Production', fontsize=16)
for year in range(start_date.year,end_date.year):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)

print('Mean Absolute Percent Error:', round(np.mean(abs(rolling_residuals/test_data)),4))
print('Root Mean Squared Error:', np.sqrt(np.mean(rolling_residuals**2)))
