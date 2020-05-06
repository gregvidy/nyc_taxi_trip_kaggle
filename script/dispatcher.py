import xgboost as xgb

MODELS = {
    "xgboost": xgb.XGBRegressor(n_estimators=100,
                                n_jobs=-1,
                                verbose=2,
                                booster='gbtree',
                                reg_lambda=2,
                                reg_alpha=2,
                                objective='reg:squarederror',
                                random_state=123)
}