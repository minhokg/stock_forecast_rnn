import logging
from typing import Dict

import numpy as np
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
)


def evaluate_single_step_model(
    model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    scalers: Dict,
    target_col: str = "Close",
) -> tuple[float, float]:
    """
    Evaluate a single-step forecasting model.

    :param model:
        Trained model.
    :param x_test:
        Test features.
    :param y_test:
        Test targets.
    :param scalers:
        Dictionary containing fitted scalers.
    :param target_col:
        Target column name.
    :return:
        MSE and MAPE.
    """
    pred = model.predict(x_test).reshape(-1, 1)
    logging.info(scalers.keys())
    scaler = scalers["Close"]

    pred = scaler.inverse_transform(pred)
    actual = scaler.inverse_transform(y_test)

    mse = mean_squared_error(actual, pred)
    mape = mean_absolute_percentage_error(actual, pred)

    return mse, mape
