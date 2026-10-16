import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y, p):
    return {"r2": float(r2_score(y, p)), "mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p)))}
