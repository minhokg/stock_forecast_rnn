from typing import Any, Callable

from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras import Model


def create_random_search(
    build_fn: Callable[..., Model],
    param_grid: dict,
    kfold: Any,
) -> RandomizedSearchCV:
    """
    Create a RandomizedSearchCV object for a Keras model.

    :param build_fn:
        Function that builds and returns a compiled Keras model.
    :type build_fn: callable

    :param param_grid:
        Hyperparameter search space.
    :type param_grid: dict

    :param kfold:
        Cross-validation splitter.
    :type kfold:
        sklearn.model_selection.BaseCrossValidator

    :return:
        Configured RandomizedSearchCV instance.
    :rtype:
        RandomizedSearchCV
    """
    model = KerasRegressor(build_fn=build_fn)

    return RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        cv=kfold,
    )
