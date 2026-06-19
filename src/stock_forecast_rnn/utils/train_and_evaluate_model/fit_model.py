from typing import Any

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping


def fit_model(
    grid_search: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    search_epochs: int = 50,
    retrain_epochs: int = 50,
    batch_size: int = 20,
    validation_split: float = 0.2,
    patience: int = 5,
):
    """
    Perform hyperparameter tuning and retrain the best model using EarlyStopping.

    :param grid_search:
        Configured RandomizedSearchCV object.
    :type grid_search: RandomizedSearchCV

    :param x_train:
        Training input data.
    :type x_train: NDArray

    :param y_train:
        Training target data.
    :type y_train: NDArray

    :param search_epochs:
        Number of epochs used during hyperparameter search.
    :type search_epochs: int

    :param retrain_epochs:
        Maximum number of epochs used when retraining the best model.
    :type retrain_epochs: int

    :param batch_size:
        Mini-batch size.
    :type batch_size: int

    :param validation_split:
        Fraction of training data reserved for validation.
    :type validation_split: float

    :param patience:
        Number of epochs without improvement before EarlyStopping.
    :type patience: int

    :return:
        Tuple containing the best model and its training history.
    :rtype: Tuple[Any, tensorflow.keras.callbacks.History]
    """
    # Hyperparameter search.
    grid_search.fit(
        x_train,
        y_train,
        epochs=search_epochs,
        batch_size=batch_size,
        verbose=0,
    )

    # Extract the best model.
    best_model = grid_search.best_estimator_

    # Configure early stopping.
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=patience,
    )

    # Retrain the best model.
    history = best_model.fit(
        x_train,
        y_train,
        epochs=retrain_epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=0,
    )

    return best_model, history
