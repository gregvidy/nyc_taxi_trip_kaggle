import os
import pandas as pd
from sklearn import ensemble
import joblib
from .cross_validation import CrossValidation
from .metrics import RegressionMetrics
import math

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    # calling the data
    df = pd.read_csv(TRAINING_DATA)

    # cross validation
    df["kfold"] = -1
    cv = CrossValidation(df,
                         shuffle=True,
                         target_cols=["trip_duration"],
                         problem_type="single_col_regression",
                         random_state=1337)
    df_split = cv.split()

    train_df = df_split[df_split.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
    valid_df = df_split[df_split.kfold==FOLD].reset_index(drop=True)
    ytrain = train_df["trip_duration"].values
    yvalid = valid_df["trip_duration"].values
    train_df = train_df.drop(["trip_duration", "kfold"], axis=1)
    valid_df = valid_df.drop(["trip_duration", "kfold"], axis=1)
    valid_df = valid_df[train_df.columns]

    # data is ready to train
    reg = dispatcher.MODELS[MODEL]
    reg.fit(train_df, ytrain, eval_set=[(train_df, ytrain), (valid_df, yvalid)], eval_metric="rmse")
    preds = reg.predict(valid_df)
    print(f"RMSE for {MODEL}_{FOLD} is: ", math.sqrt(RegressionMetrics()("mse", yvalid, preds)))
    print(f"RMSLE for {MODEL}_{FOLD} is: ", RegressionMetrics()("rmsle", yvalid, preds))

    joblib.dump(reg, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")
    print(f"{MODEL}_{FOLD} model is saved!")