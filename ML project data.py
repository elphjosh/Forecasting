# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 22:12:44 2022

@author: joshe
"""

# import requests
import pandas as pd
import fredapi as fa


key = 'bb9719b0329be10db656675822c9813b'


fred = fa.Fred(key)



#Get SP500 data

avehouse = fred.get_series('ASPUS')

avehouse.name = 'Average house price'

avehouse.tail()


#Get US GDP data

gdp = fred.get_series('GDP')

gdp.name = 'GDP'

gdp.tail()

#Name columns and convert to pandas

avehouse.columns = ['Average house price']
avehouse = pd.DataFrame(data=avehouse)
avehouse.info()

gdp.columns = ['GDP']
gdp = pd.DataFrame(data=gdp)




#Join datasets

df = avehouse.join(gdp)

df.tail()
df.head()



#CPI

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





#Get personal savings rate


sav = fred.get_series('PSAVERT')
sav.name = 'Personal savings rate'
sav = pd.DataFrame(data=sav)

sav.head()

df = df.join(sav)

df






#Get personal savings rate


er = fred.get_series('DFF')
er.name = 'Effective rate'
er = pd.DataFrame(data=er)

sav.head()

df = df.join(er)

df




#convert to csv

df.to_csv('AverageHousePrice.csv')


#Cut dataframe

df = df[:-10]


#RNN Model

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt

train_df = df[:-20]
test_df = df[-20:]


#Feature sacling

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler() #create scaler object for x and y(different means and sd)
sc_y = StandardScaler()
scaledx = sc_x.fit_transform(train_df)#fit x using x values, can also scale test data using original(same(dont include test values)) x fit to make predictions.
scaledy = sc_y.fit_transform(train_df['Average house price'].values.reshape(-1,1))
#y = sc_y.fit_transform(y)# scale salary using original salary data( we can inverse at end to see predictions using sc_y object)

scaledx

x_train = []
y_train = []




for x in range(8, len(scaledx)):
    x_train.append(scaledx[x-8:x])
    y_train.append(scaledy[x, 0])
    
    
    
   
    
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 6))


x_train.shape[1]
x_train
y_train


#Build model

model = Sequential()

model.add(LSTM(units=200, return_sequences=True, input_shape=(x_train.shape[1], 6)))
model.add(Dropout(0.1))
model.add(LSTM(units=200, return_sequences=True))
model.add(Dropout(0.05))
model.add(LSTM(units=250, return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(units=250))
model.add(Dropout(0.05))        
model.add(Dense(1))   #Prediction of next house price

model.compile(optimizer='adam', loss='mean_squared_error')
#model.fit(x_train, y_train, epochs=25, batch_size=32)
model.fit(x_train, y_train, epochs=25, batch_size=50)



#Test on test data:
    

from sklearn.preprocessing import StandardScaler

scaledx = sc_x.transform(df)#fit x using x values, can also scale test data using original(same(dont include test values)) x fit to make predictions.
scaledy = sc_y.transform(df['Average house price'].values.reshape(-1,1))
#y = sc_y.fit_transform(y)# scale salary using original salary data( we can inverse at end to see predictions using sc_y object)



x_test = []
y_test = []




for x in range(8, len(scaledx)):
    x_test.append(scaledx[x-8:x])
    y_test.append(scaledy[x, 0])
    
    
    
   
    
x_test, y_test = np.array(x_test), np.array(y_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 6))



#Make perdictions

predicted_prices = model.predict(x_test)
predicted_prices = sc_y.inverse_transform(predicted_prices)


#Plot 

x_data = range(len(predicted_prices))
x_data

plt.figure(figsize=(12,8), dpi=85)
plt.plot(df.index, sc_y.inverse_transform(scaledy), color='black', label='Actual Price')
plt.plot(df.index[8:-20], predicted_prices[:-20], color='orange', label='Train Predicted Price')
plt.plot(df.index[-20:], predicted_prices[-20:], color='limegreen', label='Test Predicted Price')
plt.legend()
plt.show()




#Accuracy

import sklearn.metrics as sm



sm.r2_score(sc_y.inverse_transform(scaledy)[-20:], predicted_prices[-20:])




sm.r2_score(sc_y.inverse_transform(scaledy)[8:-20], predicted_prices[:-20])








