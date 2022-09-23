# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 12:32:11 2021

@author: joshe
"""




from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from fmp_python.fmp import FMP
plt.rcParams.update({'figure.figsize':(10,7), 'figure.dpi':120})
import yfinance as yf
yfin.pdr_override()
import datetime

# #API

# fmp = FMP(api_key='378dccb401f0fbea5da1add85b69b1c0')


# ######Get daily stock data
# hist_chart = fmp.get_historical_chart('1hour','^GSPC')
# print(hist_chart)

def get_adj_close(ticker):
    info=yf.download(ticker,data_source='yahoo',start=datetime(2020,6,1), end=datetime(2021,6,1))['Adj Close']
    return pd.DataFrame(info)

hist_chart = get_adj_close('gspc^')



#Reverse order and reindex
stockdata = pd.DataFrame(hist_chart)
stockdata = stockdata[::-1].reset_index(drop=False)
print(stockdata.head)



#Quick plot
plt.plot(stockdata['close'])
plt.show()


#Find trend and cut df 
stockdata = stockdata[0:].reset_index(drop=True)

#Plot again

ax = plt.axes()
plt.plot(stockdata.date, stockdata.close, color="maroon") #(y and x)
ax.xaxis.set_major_locator(plt.MaxNLocator(15))                 # set x label axis ticks
plt.xticks(rotation=45)                                          #rotate lables
plt.show()


#ARIMA(p,d,q)

# ADF test for stationarity(d)  H0: the time series is non-stationary

from statsmodels.tsa.stattools import adfuller
result = adfuller(stockdata.close.dropna())
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

#p>0.5 so we difference


# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(stockdata.close); axes[0, 0].set_title('Original Series')
plot_acf(stockdata.close, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(stockdata.close.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(stockdata.close.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(stockdata.close.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(stockdata.close.diff().diff().dropna(), ax=axes[2, 1])

plt.show()

#Use d=1 as acf quick drop to 1 indicates overdifferenced

from pmdarima.arima.utils import ndiffs

y = stockdata.close
## Adf Test
ndiffs(y, test='adf')  # 1

# KPSS test
ndiffs(y, test='kpss')  # 1

# PP test:
ndiffs(y, test='pp')  # 1



##Find p through the Partial Autocorrelation (PACF) plot

plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})
plt.figure(figsize=(9,3), dpi=120)
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(stockdata.close.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(stockdata.close.diff().dropna(), ax=axes[1], lags=20, zero=False)


plt.show()


#Better view
plot_pacf(stockdata.close.diff().dropna(), zero=False)
plt.show()

# Looks like 7, might use 0. p = 7 or 0

## Find q MA component looking at ACF plot

plot_acf(stockdata.close.diff().dropna())
plt.show()

#Looks like q = 7



#Fit ARIMA Model
#from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA

# 7,1,7 ARIMA Model summary(mles dont converge)
model = ARIMA(stockdata.close, order=(6,1,6))    #seasonal_order=(7,1,7,40)
model_fit = model.fit(disp=0)
print(model_fit.summary())



# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()


#real vs predicted
model_fit.plot_predict(dynamic=False)
plt.show()

#Out-of-Time Cross validation self forecast

# Create Training and Test
train = stockdata.close[:350]
test = stockdata.close[350:]



# Build Model
  ###8 datapoints = 1 trading day
model = ARIMA(train, order=(6, 1, 6))  
fitted = model.fit(disp=-1)  
print(fitted.summary())

# Forecast
fc, se, conf = fitted.forecast(len(test), alpha=0.1)  # 90% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='Trained')
plt.plot(test, label='Actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('NVDA Forecast vs Actual')
plt.legend(loc='upper left', fontsize=8)
plt.show()


#All Accuracy tests

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
  #  acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 
            'corr':corr, 'minmax':minmax})

forecast_accuracy(fc, test.values)

#MAPE shows 99% accurate

#Forcast
# Forecast
model = ARIMA(stockdata.close, order=(6, 1, 6))
fitted = model.fit(disp=-1) 
print(fitted.summary())

#fc, confint = model.predict(fitted, start=len(stockdata.close), end=len(stockdata.close)+40) #n_periods = 24
fc, se, confint = fitted.forecast(40, alpha=0.1)  # 90% conf



index_of_fc = np.arange(len(stockdata.close), len(stockdata.close)+40)



# make series for plotting confidence intervals
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)
#print(fc_series)
#print(lower_series)
# Plot


plt.plot(stockdata.close, label = "Real")
plt.plot(fc_series, color='limegreen', label = "predicted")
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15, label="Confidence interval")

plt.title("Week Forecast for NVDA ")
plt.legend(loc='upper left', fontsize=8)
plt.show()

#Plot forcast


predictions = fc_series.reset_index(drop=True)

plt.figure(figsize=(12,8), dpi=85)
x_ticks = np.arange(0, 40, 8)
plt.xticks(x_ticks)
plt.plot(predictions, color="limegreen")
plt.fill_between((lower_series.reset_index(drop=True)).index, 
                 lower_series.reset_index(drop=True), 
                 upper_series.reset_index(drop=True), 
                 color='k', alpha=.15)
plt.title("Week Forecast for NVDA 8 hours = 1 day")
plt.xlabel("Trading hours from now")
plt.ylabel("Price ($)")
plt.show()

##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
# Auto ARIMA

import pmdarima as pm


automodel = pm.auto_arima(stockdata.close, start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=7, max_q=7, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(automodel.summary())


automodel.plot_diagnostics(figsize=(7,5))
plt.show()


# Forecast
n_periods = 40
fc1, confint1 = automodel.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc1 = np.arange(len(stockdata.close), len(stockdata.close)+n_periods)

# make series for plotting purpose
fc_series1 = pd.Series(fc1, index=index_of_fc1)
lower_series1 = pd.Series(confint1[:, 0], index=index_of_fc1)
upper_series1 = pd.Series(confint1[:, 1], index=index_of_fc1)

# Plot
plt.plot(stockdata.close, label="Real")
plt.plot(fc_series1, color='limegreen', label = "predicted")
plt.fill_between(lower_series1.index, 
                 lower_series1, 
                 upper_series1, 
                 color='k', alpha=.15, label="C.I.")

plt.title("Week autoforcast for NVDA")
plt.show()





#########For seasonal and exogenous variable go to bottom of https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/






