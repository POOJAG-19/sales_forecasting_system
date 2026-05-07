from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_arima(series):
    model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,7))
    return model.fit(disp=False)