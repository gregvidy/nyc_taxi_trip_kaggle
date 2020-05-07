import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
import numpy as np
import math

from . import dispatcher

TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")

def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df["id"].values
    prediction = None

    for FOLD in range(5):
        print(FOLD)
        df = pd.read_csv(TEST_DATA)
        cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))        
           
        # predict the data
        reg = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}.pkl"))
        
        df = df[cols]
        preds = reg.predict(df)

        if FOLD == 0:
            prediction = preds
        else:
            prediction += preds
     
    prediction /= 5
    
    sub = pd.DataFrame(np.column_stack((test_idx, prediction)),
                                        columns=["ID", "PREDS"])
    sub["id"] = sub["id"].astype(int)
    return sub

if __name__ == "__main__":
    print("==============================")
    print("Start predicting ...")
    submission = predict()
    submission.to_csv(f"models/{MODEL}.csv", index=False)
    print("Predictions saved!")
