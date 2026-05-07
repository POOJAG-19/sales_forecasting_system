import pandas as pd

def create_features(df):

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["State", "Date"])

    # lag features
    df["lag_1"] = df.groupby("State")["Total"].shift(1)
    df["lag_7"] = df.groupby("State")["Total"].shift(7)
    df["lag_30"] = df.groupby("State")["Total"].shift(30)

    # rolling features
    df["rolling_mean_7"] = (
        df.groupby("State")["Total"]
        .shift(1)
        .rolling(7)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["rolling_std_7"] = (
        df.groupby("State")["Total"]
        .shift(1)
        .rolling(7)
        .std()
        .reset_index(level=0, drop=True)
    )

    # time features
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month

    # 🔥 SAFE CLEANING (IMPORTANT FIX)
    df = df.dropna(subset=["lag_1", "lag_7", "lag_30"])

    return df