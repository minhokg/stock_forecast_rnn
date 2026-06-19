from typing import Optional

from tensorflow.keras import Model, optimizers
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Dense,
    Input,
    RepeatVector,
    SimpleRNN,
    TimeDistributed,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_rnn(
    n_past: int,
    n_features: int,
    n_neurons: Optional[int] = 20,
) -> Model:
    """
    Build and compile a simple Recurrent Neural Network (RNN) model for time series forecasting.

    The model consists of:
        - One SimpleRNN layer with ``n_neurons`` hidden units.
        - One fully connected output layer with a single neuron.
        - RMSprop optimizer and Mean Squared Error (MSE) loss.

    :param n_past:
        Number of historical time steps provided as input.
    :type n_past: int

    :param n_features:
        Number of features at each time step.
    :type n_features: int

    :param n_neurons:
        Number of hidden units in the SimpleRNN layer.
    :type n_neurons: int

    :return:
        Compiled RNN model ready for training.
    :rtype: Model
    """
    # Initialize a sequential neural network.
    model = Sequential()

    # Add a SimpleRNN layer to capture temporal patterns
    # from the input time series.
    model.add(
        SimpleRNN(
            units=n_neurons,
            activation="tanh",
            input_shape=(n_past, n_features),
        )
    )

    # Add a dense output layer for prediction.
    model.add(Dense(1, activation="tanh"))

    # Compile the model using MSE loss and RMSprop optimizer.
    model.compile(
        loss="mse",
        optimizer=optimizers.RMSprop(learning_rate=0.001),
        metrics=["mse"],
    )

    return model


def build_lstm(
    n_past: int,
    n_features: int,
    n_neurons: Optional[int] = 20,
) -> Model:
    """
    Build and compile a single-layer LSTM model for time series forecasting.

    The model consists of:
        - One LSTM layer with ``n_neurons`` hidden units.
        - One fully connected output layer with a single neuron.
        - RMSprop optimizer and Mean Squared Error (MSE) loss.

    :param n_past:
        Number of historical time steps provided as input.
    :type n_past: int

    :param n_features:
        Number of features at each time step.
    :type n_features: int

    :param n_neurons:
        Number of hidden units in the LSTM layer.
    :type n_neurons: int

    :return:
        Compiled LSTM model ready for training.
    :rtype: Model
    """
    # Initialize a sequential neural network.
    model = Sequential()

    # Add an LSTM layer to learn temporal dependencies
    # from the input time series.
    model.add(
        LSTM(
            units=n_neurons,
            activation="tanh",
            input_shape=(n_past, n_features),
        )
    )

    # Add a dense output layer for prediction.
    model.add(Dense(1, activation="tanh"))

    # Compile the model using MSE loss and RMSprop optimizer.
    model.compile(
        loss="mse",
        optimizer=optimizers.RMSprop(learning_rate=0.001),
        metrics=["mse"],
    )

    return model


def build_seq2seq_e1d1(
    n_past: int,
    n_future: int,
    n_features: int,
    activation: str = "elu",
    latent_dim: int = 100,
    learning_rate: float = 0.01,
) -> Model:
    """
    Build and compile an Encoder-Decoder (E1D1) Seq2Seq model for multistep time series forecasting.

    The architecture consists of:
        - One encoder LSTM layer.
        - Batch normalization on encoder hidden and cell states.
        - RepeatVector to replicate the encoded context.
        - One decoder LSTM layer.
        - TimeDistributed Dense output layer.

    :param n_past:
        Number of historical time steps used as input.
    :type n_past: int

    :param n_future:
        Number of future time steps to predict.
    :type n_future: int

    :param n_features:
        Number of features in the input time series.
    :type n_features: int

    :param activation:
        Activation function used in the encoder and decoder LSTM layers.
        Typical values are ``"elu"`` and ``"relu"``.
    :type activation: str

    :param latent_dim:
        Number of hidden units in the encoder and decoder LSTM layers.
    :type latent_dim: int

    :param learning_rate:
        Learning rate for the Adam optimizer.
    :type learning_rate: float

    :return:
        Compiled Seq2Seq model.
    :rtype: Model
    """
    # Define encoder input.
    encoder_input = Input(shape=(n_past, n_features))

    # Encoder.
    encoder_hidden, encoder_h, encoder_c = LSTM(
        latent_dim,
        activation=activation,
        return_sequences=False,
        return_state=True,
    )(encoder_input)

    # Normalize encoder states.
    encoder_h = BatchNormalization(momentum=0.6)(encoder_h)
    encoder_c = BatchNormalization(momentum=0.6)(encoder_c)

    # Repeat encoded context for each forecast step.
    decoder_input = RepeatVector(n_future)(encoder_h)

    # Decoder.
    decoder_output = LSTM(
        latent_dim,
        activation=activation,
        return_sequences=True,
    )(
        decoder_input,
        initial_state=[encoder_h, encoder_c],
    )

    # Forecast output.
    output = TimeDistributed(Dense(n_features))(decoder_output)

    # Build model.
    model = Model(
        inputs=encoder_input,
        outputs=output,
    )

    # Compile model.
    model.compile(
        loss="mse",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["mse"],
    )

    return model
