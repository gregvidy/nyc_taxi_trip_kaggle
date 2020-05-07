import xgboost as xgb

MODELS = {
    "xgboost": xgb.XGBRegressor(n_estimators=1500,
                                n_jobs=-1,
                                verbose=2,
                                max_depth=12,
                                subsample=0.99,
                                min_child_weight=0.5,
                                booster='gbtree',
                                objective='reg:squarederror',
                                random_state=123)
}