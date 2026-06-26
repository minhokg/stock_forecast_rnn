import logging
from functools import partial
from typing import Any

import pandas as pd
from sklearn.model_selection import KFold

from stock_forecast_rnn.models.build_models import (
    build_lstm,
    build_rnn,
)
from stock_forecast_rnn.utils.preprocessing.prepare_dataset import (
    prepare_dataset,
)
from stock_forecast_rnn.utils.train_and_evaluate_model.train_and_evaluate_model import (
    train_and_evaluate_model,
)


class StockForecastExperiment:
    """
    Manage forecasting experiments for multiple deep-learning models.

    This class centralizes dataset preparation, hyperparameter tuning,
    model training, and evaluation.

    :param train:
        Scaled training dataset.
    :type train: pd.DataFrame

    :param test:
        Scaled test dataset.
    :type test: pd.DataFrame

    :param scalers:
        Dictionary of fitted scalers.
    :type scalers: dict

    :param kfold:
        Cross-validation strategy.
    :type kfold: KFold
    """

    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        scalers: dict[str, Any],
        kfold: KFold,
    ) -> None:
        """
        Initialize the experiment configuration and datasets.

        :param train:
            Training dataset containing features and target values.
        :type train: pandas.DataFrame

        :param test:
            Test dataset used for model evaluation.
        :type test: pandas.DataFrame

        :param scalers:
            Dictionary containing fitted scaler objects used for data
            normalization and inverse transformations.
        :type scalers: dict[str, Any]

        :param kfold:
            K-Fold cross-validator used to generate training and validation
            splits during model training.
        :type kfold: sklearn.model_selection.KFold
        """
        self.train = train
        self.test = test
        self.scalers = scalers
        self.kfold = kfold

        self.results: dict[str, dict[str, Any]] = {}

    def run_rnn(
        self,
        n_past: int = 22,
        n_future: int = 1,
    ) -> None:
        """
        Train and evaluate a Simple RNN model.

        :param n_past:
            Number of historical observations.
        :type n_past: int

        :param n_future:
            Number of future observations to predict.
        :type n_future: int
        """
        logging.info("Start RNN experiment")
        x_train, y_train, x_test, y_test = prepare_dataset(
            train=self.train,
            test=self.test,
            n_past=n_past,
            n_future=n_future,
            target_col=3,  # Close
        )

        param_grid = {"model__n_neurons": [10, 20, 30]}

        build_fn = partial(
            build_rnn,
            n_past=n_past,
            n_features=self.train.shape[1],
        )

        best_model, mse, mape = train_and_evaluate_model(
            build_fn=build_fn,
            param_grid=param_grid,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            scalers=self.scalers,
            kfold=self.kfold,
        )

        self.results["rnn"] = {
            "model": best_model,
            "mse": mse,
            "mape": mape,
        }

    def run_lstm(
        self,
        n_past: int = 22,
        n_future: int = 1,
    ) -> None:
        """
        Train and evaluate an LSTM model.

        :param n_past:
            Number of historical observations.
        :type n_past: int

        :param n_future:
            Number of future observations to predict.
        :type n_future: int
        """
        logging.info("Start LSTM experiment")
        x_train, y_train, x_test, y_test = prepare_dataset(
            train=self.train,
            test=self.test,
            n_past=n_past,
            n_future=n_future,
            target_col=3,
        )

        param_grid = {"model__n_neurons": [10, 20, 30]}
        build_fn = partial(
            build_lstm,
            n_past=n_past,
            n_features=self.train.shape[1],
        )

        best_model, mse, mape = train_and_evaluate_model(
            build_fn=build_fn,
            param_grid=param_grid,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            scalers=self.scalers,
            kfold=self.kfold,
        )

        self.results["lstm"] = {
            "model": best_model,
            "mse": mse,
            "mape": mape,
        }

    def summary(self) -> pd.DataFrame:
        """
        Summarize experiment results.

        :return:
            DataFrame containing evaluation metrics.
        :rtype: pd.DataFrame
        """
        rows = []

        for name, result in self.results.items():
            rows.append(
                {
                    "model": name,
                    "mse": result["mse"],
                    "mape": result["mape"],
                }
            )

        return pd.DataFrame(rows)
