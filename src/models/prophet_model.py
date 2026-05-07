from prophet import Prophet

def train_prophet(df):
    df = df.rename(columns={"Date":"ds","Total":"y"})
    model = Prophet()
    model.fit(df)
    return model