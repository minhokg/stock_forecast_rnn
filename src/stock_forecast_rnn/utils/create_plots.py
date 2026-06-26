from collections.abc import Sequence

import matplotlib.pyplot as plt


def plot_training_history(
    train_mse: Sequence[float],
    valid_mse: Sequence[float],
) -> None:
    """
    Plot training and validation Mean Squared Error (MSE) over epochs.

    :param train_mse:
        Training MSE values recorded at each epoch.
    :type train_mse: Sequence[float]

    :param valid_mse:
        Validation MSE values recorded at each epoch.
    :type valid_mse: Sequence[float]

    :return:
        None.
    :rtype: None
    """
    # Create a new figure.
    plt.figure(figsize=(8, 5))

    # Plot training and validation metrics.
    plt.plot(train_mse, label="Train MSE")
    plt.plot(valid_mse, label="Validation MSE")

    # Configure axes and legend.
    plt.ylabel("MSE")
    plt.xlabel("Epoch")

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fancybox=True,
        shadow=False,
        ncol=2,
    )

    plt.tight_layout()
    plt.show()
