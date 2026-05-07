import pickle
import numpy as np
import pandas as pd


def forecast_8_weeks(df, models_dict, best_models):

    feature_cols = pickle.load(open("saved_models/feature_cols.pkl", "rb"))

    results = {}

    states = df["State"].unique()

    for state in states:

        df_state = df[df["State"] == state].copy().sort_values("Date")

        model_type = best_models[state]
        model = models_dict[model_type][state]

        temp_df = df_state.copy()
        predictions = []

        for _ in range(8):

            from src.features import create_features

            feat_df = create_features(temp_df)

            # 🔥 FINAL SAFETY FIX (IMPORTANT)
            if feat_df.empty:
                print(f"⚠️ Not enough data for state: {state}")
                break

            last_row = feat_df.iloc[-1]

            X = last_row.drop(["Total", "Date", "State"], errors="ignore")

            X = X.reindex(feature_cols)

            X = np.array(X).reshape(1, -1)

            pred = model.predict(X)[0]
            predictions.append(float(pred))

            # append prediction for next step
            new_row = pd.DataFrame([{
                "Date": temp_df["Date"].max() + pd.Timedelta(days=7),
                "State": state,
                "Total": pred
            }])

            temp_df = pd.concat([temp_df, new_row], ignore_index=True)

        results[state] = predictions

    return results