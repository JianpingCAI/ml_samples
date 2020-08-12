import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import yfinance as yf

## Ticker
ticker = 'SPY'

# get data
tickerdata = yf.Ticker(ticker)

# get historical prices
tickerDf = tickerdata.history(period='1d', start='2015-1-1', end='2020-1-1')
tickerDf.head()

# only the 'close' price
tickerDf = tickerDf[['Close']]
tickerDf.head()

## plot
plt.figure(figsize=(10,4))
plt.plot(tickerDf.Close)
plt.title('Stock price %s'%ticker, fontsize=20)
plt.ylabel('Price', fontsize=16)
for year in range(2015,2021):
    plt.axvline(pd.to_datetime(str(year)+'-1-1'), color='k', linestyle='--', alpha=0.2)


## Process: stationarity - take first difference
# first diff --> remove trend
first_diff = tickerDf.Close.diff().dropna()
first_diff.head()
# # method2
# first_diff = tickerDf.Close.values[1:] - tickerDf.Close.values[:-1]
# first_diff=np.concatenate([first_diff, [0]])
# tickerDf['FirstDiff'] = first_diff

# plot
plt.figure(figsize=(10,4))
plt.plot(first_diff)
plt.title('First Difference over Time (%s)'%ticker, fontsize=20)
plt.ylabel('Price Difference', fontsize=16)
for year in range(2015,2021):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)

## ACF
acf_plot = plot_acf(first_diff)
# --> does not tell much 

## PACF
pacf_plot = plot_pacf(first_diff)