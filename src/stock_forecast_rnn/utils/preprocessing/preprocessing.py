from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def split_series(
    series: NDArray[np.float64],
    n_past: int,
    n_future: int,
    target_col: Optional[list[int]] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Split a multivariate time series into input-output sequences for supervised learning.

    Each sample consists of:
        - ``n_past`` historical observations as input features.
        - ``n_future`` future observations from the selected target columns
          as prediction targets.

    For example, if ``n_past=24`` and ``n_future=6``, the function creates
    training samples using the previous 24 time steps to predict the next
    6 time steps.

    :param series:
        Input time series array with shape
        ``(n_timesteps, n_features)``.
    :type series: NDArray[np.float64]

    :param n_past:
        Number of historical time steps to include in each input sample.
    :type n_past: int

    :param n_future:
        Number of future time steps to predict.
    :type n_future: int

    :param target_col:
        Indices of the target columns to be predicted. If ``None``,
        all columns are used as prediction targets.
    :type target_col: Optional[list[int]]

    :return:
        A tuple containing:

        - **X** (*NDArray[np.float64]*) -- Input sequences with shape
          ``(n_samples, n_past, n_features)``.
        - **y** (*NDArray[np.float64]*) -- Target sequences with shape
          ``(n_samples, n_future, n_target_features)``.
    :rtype: Tuple[NDArray[np.float64], NDArray[np.float64]]
    """
    # Store generated input and target sequences.
    x, y = [], []

    # Iterate through the time series using a sliding window.
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future

        # Stop when there are not enough future observations.
        if future_end > len(series):
            break

        # Historical window used as model input.
        past = series[window_start:past_end, :]

        # Future window used as prediction target.
        future = series[past_end:future_end, target_col]

        x.append(past)
        y.append(future)

    return np.array(x), np.array(y)
