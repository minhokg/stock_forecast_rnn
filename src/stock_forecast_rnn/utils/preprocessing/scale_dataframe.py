from typing import Dict, Tuple

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def scale_dataframe(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, MinMaxScaler]]:
    """
    Scale training and test datasets using a separate ``MinMaxScaler`` for each feature column.

    The scaler is fitted on the training data only and then applied to
    both the training and test datasets to avoid data leakage.

    All features are scaled to the range ``[-1, 1]``.

    :param train_df:
        Training dataset containing the features to be scaled.
    :type train_df: pd.DataFrame

    :param test_df:
        Test dataset containing the features to be scaled.
    :type test_df: pd.DataFrame

    :return:
        A tuple containing:

        - **train_scaled** (*pd.DataFrame*) -- Scaled training dataset.
        - **test_scaled** (*pd.DataFrame*) -- Scaled test dataset.
        - **scalers** (*Dict[str, MinMaxScaler]*) -- Dictionary mapping
          each column name to its fitted scaler.
    :rtype:
        Tuple[
            pd.DataFrame,
            pd.DataFrame,
            Dict[str, MinMaxScaler]
        ]
    """
    # Create copies to avoid modifying the original DataFrames.
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()

    # Store the fitted scaler for each column.
    scalers = {}

    # Scale each feature independently.
    for col in train_df.columns:
        scaler = MinMaxScaler(feature_range=(-1, 1))

        # Fit the scaler on the training data.
        train_scaled[col] = scaler.fit_transform(train_df[[col]])

        # Apply the fitted scaler to the test data.
        test_scaled[col] = scaler.transform(test_df[[col]])

        # Save the fitted scaler for future inverse transformations.
        scalers[col] = scaler

    return train_scaled, test_scaled, scalers
