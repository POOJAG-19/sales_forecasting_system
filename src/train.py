import os
import pickle
import pandas as pd

from src.preprocessing import load_data, fill_missing_dates
from src.features import create_features

from src.models.xgboost_model import train_xgb
from src.models.arima_model import train_arima
from src.models.prophet_model import train_prophet

from src.evaluate import evaluate
from src.inference.forecast_8weeks import forecast_8_weeks


def main():

    print("Loading data...")
    df = load_data("data/raw.xlsx")
    df = fill_missing_dates(df)
    df = create_features(df)

    states = df["State"].unique()
    print("States found:", states)

    best_models = {}
    scores_global = {}

    models_dict = {
        "xgb": {},
        "arima": {},
        "prophet": {}
    }

    feature_cols = None

    print("\nTraining models...\n")

    for state in states:

        df_s = df[df["State"] == state].copy()

        y = df_s["Total"]
        X = df_s.drop(["Total", "Date", "State"], axis=1).select_dtypes("number")

        # store feature schema once
        if feature_cols is None:
            feature_cols = X.columns.tolist()

        split = int(len(df_s) * 0.8)

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # ---------------- XGBOOST ----------------
        xgb = train_xgb(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        xgb_score = evaluate(y_test, xgb_pred)
        models_dict["xgb"][state] = xgb

        # ---------------- ARIMA ----------------
        arima = train_arima(y_train)
        arima_pred = arima.forecast(len(y_test))
        arima_score = evaluate(y_test, arima_pred)
        models_dict["arima"][state] = arima

        # ---------------- PROPHET ----------------
        prophet = train_prophet(df_s[["Date", "Total"]])
        future = prophet.make_future_dataframe(periods=len(y_test))
        forecast = prophet.predict(future)
        prophet_pred = forecast["yhat"].tail(len(y_test)).values
        prophet_score = evaluate(y_test, prophet_pred)
        models_dict["prophet"][state] = prophet

        # ---------------- SELECT BEST ----------------
        scores = {
            "xgb": xgb_score,
            "arima": arima_score,
            "prophet": prophet_score
        }

        best = min(scores, key=scores.get)

        best_models[state] = best
        scores_global[state] = scores

        print(f"{state} → {best}")

    # ---------------- SAVE ARTIFACTS ----------------
    os.makedirs("saved_models", exist_ok=True)

    pickle.dump(best_models, open("saved_models/best_models.pkl", "wb"))
    pickle.dump(models_dict, open("saved_models/models_dict.pkl", "wb"))
    pickle.dump(feature_cols, open("saved_models/feature_cols.pkl", "wb"))

    # ---------------- 8 WEEK FORECAST ----------------
    print("\nGenerating 8-week forecasts...\n")

    forecast_results = forecast_8_weeks(df, models_dict, best_models)

    pickle.dump(forecast_results, open("saved_models/forecast_8weeks.pkl", "wb"))

    for state, vals in forecast_results.items():
        print(state, "→", vals)


if __name__ == "__main__":
    main()