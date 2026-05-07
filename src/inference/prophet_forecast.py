def forecast_prophet_8weeks(model):

    future = model.make_future_dataframe(periods=8, freq="W")
    forecast = model.predict(future)

    return forecast["yhat"].tail(8).tolist()