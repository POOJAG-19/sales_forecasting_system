from xgboost import XGBRegressor

def train_xgb(X_train, y_train):
    model = XGBRegressor(n_estimators=200)
    model.fit(X_train, y_train)
    return model