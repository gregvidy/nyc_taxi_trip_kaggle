import numpy as np
from sklearn import metrics as skmetrics
import math

class ClassificationMetrics:
    def __init__(self):
        self.metrics = {
            "accuracy": self._accuracy,
            "f1": self._f1,
            "precision": self._precision,
            "recall": self._recall,
            "auc": self._auc,
            "logloss": self._logloss,
            "gini": self._gini,
            "gini_normalized": self._gini_normalized
        }
    
    def __call__(self, metric, y_true, y_pred, y_proba=None):
        if metric not in self.metrics:
            raise Exception("Metric is not supported!")
        if metric == "auc":
            if y_proba is not None:
                return self._auc(y_true=y_true, y_pred=y_proba)
            else:
                raise Exception("y_proba cannot be None for AUC")
        elif metric == "logloss":
            if y_proba is not None:
                return self._logloss(y_true=y_true, y_pred=y_proba)
            else:
                raise Exception("y_proba cannot be None for LogLoss")                
        else:
            return self.metrics[metric](y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _auc(y_true, y_pred):
        return skmetrics.roc_auc_score(y_true=y_true, y_score=y_pred)

    @staticmethod
    def _gini(y_true, y_pred):
        auc = skmetrics.roc_auc_score(y_true=y_true, y_score=y_pred)
        return 2 * auc - 1

    @staticmethod
    def _gini_normalized(y_true, y_pred):
        return gini(y_true=y_true, y_pred=y_pred) / gini(y_true=y_true, y_pred=y_true)

    @staticmethod
    def _logloss(y_true, y_pred):
        return skmetrics.log_loss(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _accuracy(y_true, y_pred):
        return skmetrics.accuracy_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _f1(y_true, y_pred):
        return skmetrics.f1_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _precision(y_true, y_pred):
        return skmetrics.precision_score(y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _recall(y_true, y_pred):
        return skmetrics.recall_score(y_true=y_true, y_pred=y_pred)


class RegressionMetrics:
    def __init__(self):
        self.metrics = {
            "mse": self._mse,
            "rmse": self._rmse,
            "rmsle": self._rmsle
        }

    def __call__(self, metric, y_true, y_pred):
        if metric not in self.metrics:
            raise Exception("Metric is not supported!")
        else:
            return self.metrics[metric](y_true=y_true, y_pred=y_pred)

    @staticmethod
    def _mse(y_true, y_pred):
        return skmetrics.mean_squared_error(y_true=y_true, y_pred=y_pred)

    def _rmse(y_true, y_pred):
        return math.sqrt(skmetrics.mean_squared_error(y_true=y_true, y_pred=y_pred))

    @staticmethod
    def _rmsle(y_true, y_pred) : 
        assert len(y_true) == len(y_pred)
        return np.sqrt(np.mean((np.log(1+np.array(y_pred)) - np.log(1+np.array(y_true)))**2))