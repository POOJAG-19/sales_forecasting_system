def forecast_arima_8weeks(model):
    return model.forecast(steps=8).tolist()