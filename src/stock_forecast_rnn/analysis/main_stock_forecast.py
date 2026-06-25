import os
import warnings

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging

import yfinance as yf
from sklearn.model_selection import KFold

from stock_forecast_rnn.experiments.stock_forecast_experiment import (
    StockForecastExperiment,
)
from stock_forecast_rnn.utils.helper.setup_logging import setup_logging
from stock_forecast_rnn.utils.preprocessing.scale_dataframe import scale_dataframe


def main() -> None:
    """
    Run forecasting experiments on Google stock data.

    The workflow consists of:

        1. Download historical Google stock data.
        2. Split the data into training and test sets.
        3. Scale all features using MinMaxScaler.
        4. Train and evaluate RNN, LSTM, and Seq2Seq models.
        5. Display a summary of model performance.
    """
    setup_logging()
    warnings.filterwarnings("ignore")

    # Download six years of Google stock data.
    goog = yf.Ticker("GOOGL")
    goog_6y = goog.history(period="6y")

    # Select Open, High, Low, Close, and Volume.
    goog = goog_6y.iloc[:, 0:5]

    # Check for missing values.
    logging.info("Missing values:")
    logging.info(goog.isnull().sum())

    # Train-test split.
    train_ratio = 0.7
    train_len = int(train_ratio * len(goog))

    train_df = goog.iloc[:train_len]
    test_df = goog.iloc[train_len:]

    # Scale data.
    train, test, scalers = scale_dataframe(
        train_df=train_df,
        test_df=test_df,
    )

    # Create experiment manager.
    experiment = StockForecastExperiment(
        train=train,
        test=test,
        scalers=scalers,
        kfold=KFold(
            n_splits=3,
            shuffle=True,
            random_state=42,
        ),
    )

    # Run experiments.
    logging.info("Training RNN...")
    experiment.run_rnn()

    logging.info("Training LSTM...")
    experiment.run_lstm()

    logging.info("Training Seq2Seq...")
    experiment.run_seq2seq()

    # Show final results.
    logging.info("\nExperiment Summary")
    logging.info("=" * 60)
    logging.info(experiment.summary())


if __name__ == "__main__":
    main()
