from typing import Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from stock_forecast_rnn.utils.preprocessing.split_series import split_series


def prepare_dataset(
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_past: int,
    n_future: int,
    target_col: int | list[int],
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Prepare training and test datasets for time series forecasting.

    This function converts the scaled training and test DataFrames into
    supervised learning sequences using a sliding window approach.

    To generate test sequences correctly, the last ``n_past`` observations
    from the training dataset are prepended to the test dataset. This ensures
    that sufficient historical observations are available to predict the
    first test target.

    :param train:
        Training dataset containing historical observations.
    :type train: pd.DataFrame

    :param test:
        Test dataset containing future observations.
    :type test: pd.DataFrame

    :param n_past:
        Number of historical time steps used as input features.
    :type n_past: int

    :param n_future:
        Number of future time steps to predict.
    :type n_future: int

    :param target_col:
        Target column index or list of column indices to predict.
        For single-target forecasting, this can be an integer.
        For multi-target forecasting, this can be a list of integers.
    :type target_col: Optional[list[int] | int]

    :return:
        A tuple containing:

        - **x_train** (*NDArray[np.float64]*) -- Training input sequences.
        - **y_train** (*NDArray[np.float64]*) -- Training target sequences.
        - **x_test** (*NDArray[np.float64]*) -- Test input sequences.
        - **y_test** (*NDArray[np.float64]*) -- Test target sequences.
    :rtype:
        Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64]
        ]
    """
    # Generate training sequences.
    x_train, y_train = split_series(
        series=train.values,
        n_past=n_past,
        n_future=n_future,
        target_col=target_col,
    )

    # Append the final n_past observations from the training dataset
    # to the beginning of the test dataset. This allows the first test
    # prediction to use the required historical context.
    test_data = pd.concat([train.iloc[-n_past:], test])

    # Generate test sequences.
    x_test, y_test = split_series(
        series=test_data.values,
        n_past=n_past,
        n_future=n_future,
        target_col=target_col,
    )

    return x_train, y_train, x_test, y_test
