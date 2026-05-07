import pandas as pd
from src.features import create_features

def forecast_xgb_8weeks(model, df, state):

    data = df[df["State"] == state].copy()
    data = data.sort_values("Date")

    preds = []

    for _ in range(8):

        feat = create_features(data)
        last = feat.iloc[-1]

        X = last[
            ["lag_1","lag_7","lag_30",
             "rolling_mean_7","rolling_std_7",
             "day_of_week","month"]
        ].values.reshape(1, -1)

        pred = model.predict(X)[0]
        preds.append(pred)

        new_row = pd.DataFrame([{
            "Date": data["Date"].max() + pd.Timedelta(days=7),
            "State": state,
            "Total": pred
        }])

        data = pd.concat([data, new_row], ignore_index=True)

    return preds