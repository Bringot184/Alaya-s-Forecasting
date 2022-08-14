#Prediction
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