#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 16:34:01 2022

@author: josh
"""

# import requests
import pandas as pd
import fredapi as fa


key = 'bb9719b0329be10db656675822c9813b'


fred = fa.Fred(key)


#Get SP500 data

sp500 = fred.get_series('SP500')

sp500.name = 'SP500'

sp500.head()


#https://fred.stlouisfed.org/


# data = requests.get(f'https://api.stlouisfed.org/fred/series/observations?series_id=SP500&api_key={key}&file_type=json').json()

# data = pd.DataFrame(data=data)

# data.tail()


#Get US GDP data

gdp = fred.get_series('GDP')

gdp.name = 'GDP'

gdp.tail()

#Name columns and convert to pandas

sp500.columns = ['SP500']
sp500 = pd.DataFrame(data=sp500)
sp500.info()

gdp.columns = ['GDP']
gdp = pd.DataFrame(data=gdp)

sp500.tail()


#Join datasets

df = sp500.join(gdp)

df.tail()
df.head()

#Get CPI data

cpi = fred.get_series('CPIAUCSL')
cpi.name = 'CPI'
cpi = pd.DataFrame(data=cpi)

cpi.head()

df = df.join(cpi)


#Get unemployment rate


unrate = fred.get_series('UNRATE')
unrate.name = 'Unemployment'
unrate = pd.DataFrame(data=unrate)

unrate.head()

df = df.join(unrate)


#Get interest rate


interest = fred.get_series('DFF')
interest.name = 'Interest'
interest = pd.DataFrame(data=interest)

interest.head()

df = df.join(interest)

#Rename

df.rename(columns = {'Interest rate':'Interest'}, inplace = True)

df



#Get retail sales



retail = fred.get_series('RSXFS')
retail.name = 'Retail_Sales'
retail = pd.DataFrame(data=retail)

retail.head()

df = df.join(retail)

#Get Industrial production 


indpro = fred.get_series('INDPRO')
indpro.name = 'Industrial_production'
indpro = pd.DataFrame(data=indpro)

indpro.head()

df = df.join(indpro)



#Personal saving rate



savings = fred.get_series('PSAVERT')
savings.name = 'Personal_Savings'
savings = pd.DataFrame(data=savings)

savings.head()

df = df.join(savings)

df.info()



#New houses started



housing = fred.get_series('HOUST')
housing.name = 'Houses_Started'
housing = pd.DataFrame(data=housing)

housing.head()

df = df.join(housing)

df.info()


#Oil price

oil = fred.get_series('DCOILWTICO')
oil.name = 'Oil_Price'
oil = pd.DataFrame(data=oil)

oil.head()

df = df.join(oil)

df.info()



#Real GDP Nowcast

gdpn = fred.get_series('STLENI')
gdpn.name = 'GDPnow'
gdpn = pd.DataFrame(data=gdpn)

gdpn.head()

df = df.join(gdpn)




import yfinance as yf
dxy = yf.download('DX-Y.NYB', start_date='2012-06-15', end_date='2022-06-14')['Adj Close']
dxy.name = 'DXY'
dxy = pd.DataFrame(data=dxy)

dxy.head()

df = df.join(dxy)

df.info()




#Commodity prices

comm = fred.get_series('PPIACO')
comm.name = 'Commodity_Price'
comm = pd.DataFrame(data=comm)

comm.head()

df = df.join(comm)

df.info()

#Fill nulls

df1 = df

df1.fillna(method='ffill', inplace=True)
df1.fillna(method='bfill', inplace=True)

df1.info()

df1


#df1['1mchange'] = df1['SP500'].diff(periods=30 )
df1['1mchange'] = df1['SP500'].pct_change(periods=30)


#1 day pct change for first 30 days
df1['1mchange'][:30] = df1['SP500'][:30].pct_change()

df1.info()
df1.head()


#    Forex 

# EURESD key = 'DEXUSEU'

#Idea: create model that looks at USD index+ other exchanges to predict movement of EURUSD in 1 Hour. Shift y 1 hour back

##---------------------------------------------------------------------------

#SVR Model



df2 = df1
df2 = df2.dropna()
x = df2.iloc[:, 1:-1].values   #
y = df2.iloc[:, 0].values     #




y = y.reshape(len(y),1)# reshape y into a 2D array (len(y) rows and 1 column)

#Feature sacling

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler() #create scaler object for x and y(different means and sd)
sc_y = StandardScaler()
x = sc_x.fit_transform(x)#fit x using x values, can also scale test data using original(same(dont include test values)) x fit to make predictions.
y = sc_y.fit_transform(y)# scale salary using original salary data( we can inverse at end to see predictions using sc_y object)


#Train data

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')#create model  # depends on data, Gaussian Radial Basis Function.(Most common)
regressor.fit(x, y) #train model




import matplotlib.pyplot as plt

plt.plot(range(len(y)), sc_y.inverse_transform(y), color = 'red')
plt.plot(range(len(y)), sc_y.inverse_transform(regressor.predict(x)), color = 'blue')
plt.title('SP500 Predicted(Blue) vs Actual(Red)')
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()

len(sc_y.inverse_transform(regressor.predict(x)))



(((sc_y.inverse_transform(regressor.predict(x))[-1])-(sc_y.inverse_transform(y)[-1]))/(sc_y.inverse_transform(y)[-1]))*100

regressor.score(x,y)

##---------------------------------------------------------------------------


#Random forest model


from sklearn.ensemble import RandomForestRegressor
regressor2 = RandomForestRegressor(n_estimators = 100, random_state = 0, max_depth=7)
regressor2.fit(x, y)


plt.figure(figsize=(12,5), dpi=100)
plt.plot(df2.index[-100:], sc_y.inverse_transform(y)[-100:], color = 'limegreen', label="Actual")
plt.plot(df2.index[-100:], sc_y.inverse_transform(regressor2.predict(x))[-100:], color = 'orange', label="Forecast")
plt.title('SP500 Forest Predicted(Orange) vs Actual(Green)')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()



(((sc_y.inverse_transform(regressor2.predict(x))[-1])-(sc_y.inverse_transform(y)[-1]))/(sc_y.inverse_transform(y)[-1]))*100

regressor2.score(x,y)

regressor2.feature_importances_

plt.barh(df2.columns[1:-1], regressor2.feature_importances_)




plt.matshow(df2.corr(), label=df2.columns[1:-1])
plt.show()


df2.corr()






# Test Train Slpit


x_train = x[0:-10]
y_train = y[0:-10]
x_test = x[-100:]
y_test = y[-100:]

#SVR Model


regressor3 = SVR(kernel = 'rbf', gamma='auto', epsilon=0.1)#create model  # depends on data, Gaussian Radial Basis Function.(Most common)
regressor3.fit(x_train, y_train)


plt.figure(figsize=(12,5), dpi=100)
plt.plot(df2.index[-100:], sc_y.inverse_transform(y)[-100:], color = 'limegreen', label="Actual")
plt.plot(df2.index[-100:], sc_y.inverse_transform(regressor3.predict(x_test))[-100:], color = 'orange', label="Forecast")
plt.title('SP500 Forest Predicted(Orange) vs Actual(Green)')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()



(((sc_y.inverse_transform(regressor2.predict(x))[-1])-(sc_y.inverse_transform(y)[-1]))/(sc_y.inverse_transform(y)[-1]))*100

regressor3.score(x_test,y_test)





#XGBOOST

from xgboost import XGBRegressor
classifier = XGBRegressor()
regressor2.fit(x_train, y_train)




plt.figure(figsize=(12,5), dpi=100)
plt.plot(df2.index[-100:], sc_y.inverse_transform(y)[-100:], color = 'limegreen', label="Actual")
plt.plot(df2.index[-100:], sc_y.inverse_transform(regressor2.predict(x))[-100:], color = 'orange', label="Forecast")
plt.title('SP500 Forest Predicted(Orange) vs Actual(Green)')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()





sc_y.inverse_transform(y)[-5:]





corr_df = df.corr(method='pearson')
corr_df

import seaborn
seaborn.heatmap(corr_df, cmap='RdYlGn',vmax=1.0,vmin=-1.0,linewidths=2.1)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()







#RNN


