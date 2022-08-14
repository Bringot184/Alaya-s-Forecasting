#Imports
from cProfile import label
from pydoc import plain
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.metrics import mean_squared_error

#Config
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Vars
col = 'cantidad_vendida'
iteraciones = 3 

#Functions
def Prediction (train, test, model, x,y,z):
    if model == 'ARMA':
        ARIMAmodel = SARIMAX (train[col], order = (x,y,z))
    elif model == 'ARIMA':
        ARIMAmodel = ARIMA(train[col], order = (x, y, z))
    ARIMAmodel = ARIMAmodel.fit()
    y_pred = ARIMAmodel.get_forecast(len(test.index))
    y_pred_df = y_pred.conf_int(alpha = 0.05) 
    y_pred_df[col] = ARIMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
    y_pred_df.index = test.index
    return y_pred_df   

#Data
data = pd.read_csv("data/data.csv")
data.index = pd.to_datetime(data['fecha'], format='%Y-%m-%d')
dataTest = data[data[col] == 'test']
data = data[data[col] != 'test']
data[col] = pd.to_numeric(data[col], errors='ignore', downcast='integer')

#Split Data
sku1 = data[data['sku']=='articulo1']
sku2 = data[data['sku']=='articulo2']
sku1Test = dataTest[dataTest['sku']=='articulo1']
sku2Test = dataTest[dataTest['sku']=='articulo2']
del data['fecha']
del dataTest['fecha']

#Statistics
mean_sku1 = sku1.rolling(12).mean()
var_sku1 = sku1.rolling(12).std()
mean_sku2 = sku2.rolling(12).mean()
var_sku2 = sku2.rolling(12).std()

#Plotting
sns.set()
plt.ylabel('Cantidad Vendida')
plt.xlabel('Fecha')
plt.xticks(rotation=45)

#SKU1
plt.subplot(2, 2, 1)
plt.plot(sku1.index,sku1[col],c='g')
plt.plot(mean_sku1.index,mean_sku1[col],c='blue')
plt.plot(var_sku1.index,var_sku1[col],c='black')
plt.legend(['Articulo 1','Mean','Var'])
#SKU2
plt.subplot(2, 2, 2)
plt.plot(sku2.index,sku2[col],c='r')
plt.plot(mean_sku2.index,mean_sku2[col],c='yellow')
plt.plot(var_sku2.index,var_sku2[col],c='purple')
plt.legend(['Articulo 2','Mean','Var'])
#Predict sku 1 
prediction1 = Prediction(sku1,sku1Test,'ARIMA',0,3,3)
plt.subplot(2, 2, 3)
plt.plot(sku1.index,sku1[col],c='green',label='Train')
plt.plot(sku1Test.index,prediction1[col], color='Yellow', label = 'ARIMA Predictions')
plt.legend()

#Predict sku 2
prediction2 = Prediction(sku2,sku2Test,'ARIMA',0,3,3)
plt.subplot(2, 2, 4)
plt.plot(sku2.index,sku2[col],c='red',label='Train')
plt.plot(sku2Test.index, prediction2[col], color='Yellow', label = 'ARIMA Predictions')
plt.legend()

plt.show()

del prediction1['lower cantidad_vendida']
del prediction1['upper cantidad_vendida']
del prediction2['lower cantidad_vendida']
del prediction2['upper cantidad_vendida']
prediction1['sku'] = 'Articulo1'
prediction2['sku'] = 'Articulo2'
result = pd.concat([prediction1,prediction2])
result.to_csv("prediction.csv")