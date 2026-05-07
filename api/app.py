import os
import pickle
import pandas as pd
from fastapi import FastAPI

from src.inference.forecast_8weeks import forecast_8_weeks
from src.features import create_features

app = FastAPI(title="8-Week Sales Forecasting API")

# -----------------------------
# LOAD ARTIFACTS
# -----------------------------
best_models = pickle.load(open("saved_models/best_models.pkl", "rb"))
models_dict = pickle.load(open("saved_models/models_dict.pkl", "rb"))

# -----------------------------
# LOAD DATA (SAFE INIT)
# -----------------------------
df = pd.read_excel("data/raw.xlsx")

# 🔥 GLOBAL FIX: ENSURE CONSISTENCY
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["State", "Date"])


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "8-Week Forecasting API is running 🚀",
        "available_states": list(best_models.keys())
    }


# -----------------------------
# FORECAST SINGLE STATE
# -----------------------------
@app.get("/forecast/{state}")
def forecast_state(state: str):

    state = state.strip()

    if state not in best_models:
        return {"error": f"State '{state}' not found in trained models"}

    df_state = df[df["State"] == state].copy()

    if df_state.empty:
        return {"error": f"No data found for state: {state}"}

    # 🔥 ensure datetime safety
    df_state["Date"] = pd.to_datetime(df_state["Date"])

    # feature engineering
    df_state = create_features(df_state)

    result = forecast_8_weeks(df_state, models_dict, best_models)

    return {
        "state": state,
        "best_model": best_models[state],
        "forecast_8_weeks": result[state]
    }


# -----------------------------
# FORECAST ALL STATES
# -----------------------------
@app.get("/forecast_all")
def forecast_all():

    df_all = df.copy()

    # 🔥 ensure datetime safety
    df_all["Date"] = pd.to_datetime(df_all["Date"])

    # feature engineering
    df_feat = create_features(df_all)

    result = forecast_8_weeks(df_feat, models_dict, best_models)

    return {
        "forecast_8_weeks_all_states": result
    }