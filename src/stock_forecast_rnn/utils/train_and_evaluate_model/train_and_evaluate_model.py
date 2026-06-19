from typing import Any, Callable

from tensorflow.keras import Model

from stock_forecast_rnn.utils.create_plots import plot_training_history
from stock_forecast_rnn.utils.hyperparameter_tuning.create_random_search import (
    create_random_search,
)
from stock_forecast_rnn.utils.train_and_evaluate_model.evaluate_model import (
    evaluate_single_step_model,
)
from stock_forecast_rnn.utils.train_and_evaluate_model.fit_model import fit_model


def train_and_evaluate_model(
    build_fn: Callable[..., Model],
    param_grid: dict[str, list[Any]],
    x_train: Any,
    y_train: Any,
    x_test: Any,
    y_test: Any,
    scalers: dict[str, Any],
    kfold: Any,
) -> tuple[Any, float, float]:
    """
    Train a forecasting model using hyperparameter tuning and evaluate its predictive performance on a test dataset.

    This function performs the following steps:

        1. Creates a ``RandomizedSearchCV`` object.
        2. Searches for the best hyperparameter combination.
        3. Retrains the best model using Early Stopping.
        4. Visualizes training and validation MSE.
        5. Evaluates the model on the test dataset using MSE and MAPE.

    :param build_fn:
        Function that builds and returns a compiled Keras model.
    :type build_fn: Callable[..., Model]

    :param param_grid:
        Hyperparameter search space used in
        ``RandomizedSearchCV``.
    :type param_grid: dict[str, list[Any]]

    :param x_train:
        Training feature dataset.
    :type x_train: Any

    :param y_train:
        Training target dataset.
    :type y_train: Any

    :param x_test:
        Test feature dataset.
    :type x_test: Any

    :param y_test:
        Test target dataset.
    :type y_test: Any

    :param scalers:
        Dictionary containing fitted scalers used to inverse-transform
        predictions and actual values.
    :type scalers: dict[str, Any]

    :param kfold:
        Cross-validation strategy used during hyperparameter tuning.
    :type kfold: Any

    :return:
        A tuple containing:

        - **best_model** -- Best-performing trained model.
        - **mse** (*float*) -- Mean Squared Error on the test dataset.
        - **mape** (*float*) -- Mean Absolute Percentage Error on the test dataset.
    :rtype: tuple[Any, float, float]
    """
    # Create a hyperparameter search object.
    grid = create_random_search(
        build_fn=build_fn,
        param_grid=param_grid,
        kfold=kfold,
    )

    # Fit the search object and retrain the best model.
    best_model, history = fit_model(
        grid_search=grid,
        x_train=x_train,
        y_train=y_train,
    )

    # Visualize training and validation MSE.
    plot_training_history(
        train_mse=history.history["mse"],
        valid_mse=history.history["val_mse"],
    )

    # Evaluate the model on the test dataset.
    mse, mape = evaluate_single_step_model(
        model=best_model,
        x_test=x_test,
        y_test=y_test,
        scalers=scalers,
    )

    return best_model, mse, mape
