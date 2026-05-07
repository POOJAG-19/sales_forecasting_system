# End-to-End Time Series Forecasting System with FastAPI

## Overview

This project is a production-style time series forecasting system developed to predict the next 8 weeks of sales for each state using historical sales data.

The system:
- trains multiple forecasting models
- compares model performance
- automatically selects the best model
- exposes predictions through REST APIs using FastAPI

---

## Features

- Multi-model forecasting pipeline
- SARIMA forecasting
- Facebook Prophet forecasting
- XGBoost with lag-based feature engineering
- LSTM deep learning forecasting
- Automatic best model selection using MAE
- Recursive 8-week forecasting
- Multi-state forecasting support
- FastAPI deployment
- Production-style modular architecture

---

## Tech Stack

- Python
- Pandas
- NumPy
- XGBoost
- Statsmodels
- Prophet
- TensorFlow / Keras
- FastAPI
- Uvicorn
- Scikit-learn

---

## Feature Engineering

Implemented:
- lag_1
- lag_7
- lag_30
- rolling mean
- rolling standard deviation
- day_of_week
- month features

---

## Model Evaluation

Models were evaluated using:
- Mean Absolute Error (MAE)

The model with the lowest MAE was automatically selected for each state.


---

## Installation

### Clone repository

```bash
git clone <your_repo_url>
cd forecasting-system
```

### Create environment

```bash
conda create -n forecasting-env python=3.11
conda activate forecasting-env
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training Models

```bash
python -m src.train
```

---

## Running API

```bash
uvicorn api.app:app --reload
```

Swagger docs:

```text
http://127.0.0.1:8000/docs
```

---

## Author

Pooja Golla