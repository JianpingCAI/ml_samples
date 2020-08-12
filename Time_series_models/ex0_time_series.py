import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

## Ice cream production data
df_ice_cream = pd.read_csv('ice_cream.csv')
df_ice_cream.head()

# rename colums
df_ice_cream.rename(
    columns={'DATE': 'date', 'IPN31152N': 'production'}, inplace=True)

# convert datetime
df_ice_cream['date'] = pd.to_datetime(df_ice_cream.date)
df_ice_cream.head()

# set index
df_ice_cream.set_index('date', inplace=True)

# just get partial data(from 2010 onwards)
start_date = pd.to_datetime('2010-01-01')
df_ice_cream = df_ice_cream[start_date:]
df_ice_cream.head()

# plot
plt.figure(figsize=(10, 4))
plt.plot(df_ice_cream.production)
plt.title('Ice Create Production', fontsize=20)
plt.ylabel('Production', fontsize=16)
for year in range(2011, 2021):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'),
                color='k', linestyle='--', alpha=0.2)

## ACF
acf_plot = plot_acf(df_ice_cream.production, lags=100)

#==>Based on decaying ACF, we are likely dealing with an Auto Regressive process

## PACF
pacf_plot = plot_pacf(df_ice_cream.production)
#==>Based on PACF, we should start with an Auto Regressive model with lags 1, 2, 3, 10, 13
