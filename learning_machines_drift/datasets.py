"""Datasets module with functions for generating example data.
"""

# pylint: disable=invalid-name

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from scipy.special import expit


def logistic_model(
    x_mu: NDArray[np.float64] = np.array([0.0, 0.0, 0.0]),
    x_scale: NDArray[np.float64] = np.array([1.0, 1.0, 1.0]),
    x_corr: NDArray[np.float64] = np.array(
        [[1.0, 0.4, 0.0], [0.4, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ),
    alpha: float = 0.5,
    beta: NDArray[np.float64] = np.array(
        [
            1.0,
            0.5,
            0.0,
        ]
    ),
    size: int = 50,
    seed: Optional[int] = None,
    return_latents: bool = False,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]]:
    """Generate synthetic features, labels and latents.

    Features are generated from a multivariate normal distribution, where the
    mean vector, scale vector and correlation matrix can be specified, allowing
    users to simulate covariate drift.

    Labels are generated with a logistic regression model. The regression
    parameters are controlled with the `beta` parameter, allowing simulation of
    concept drift.

    Latents are a single feature as characterizing the Bernoulli probability
    generated by the model.

    Args:
        x_mu (NDArray[np.float64]): Mean vector of features.
            Defaults to `np.array([0.0, 0.0, 0.0])`.
        x_scale (NDArray[np.float64]): Scale of features.
            Defaults to `np.array([1.0, 1.0, 1.0])`.
        x_corr (NDArray[np.float64]): Correlation matrix giving the correlation
            between features. Defaults to
            `np.array([[1.0, 0.4, 0.0], [0.4, 1.0, 0.0], [0.0, 0.0, 1.0]])`.
        alpha (float): Regression alpha parameter. Defaults to 0.5.
        beta (NDArray[np.float64]): Regression beta parameters . Defaults to
            `np.array([1.0, 0.5, 0.0])`.
        size (int): Number of samples to draw from model. Defaults to 50.
        return_latents (bool): Return underlying prediction value before
            thresholding as 'latent' data. Defaults to False.


    Returns:
        Tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.float64]]]:
            Tuple of features, labels and (optional) latents generated.
    """

    # pylint: disable=too-many-instance-attributes

    if seed:
        np.random.seed(seed)

    # Sample features from variate normal
    X_diag: NDArray[np.float64] = np.diag(x_scale)
    X_covariance = np.dot(np.dot(X_diag, x_corr), X_diag)
    X = stats.multivariate_normal.rvs(x_mu, X_covariance, size=size)

    # Regression parameters
    alpha = 0.5
    beta = np.array(
        [
            1.0,
            0.5,
            0.0,
        ]
    )

    # Sample from bernoulli distribution
    theta = expit(alpha + np.dot(X, beta))
    Y = stats.bernoulli.rvs(theta)

    # If not return_latents
    if not return_latents:
        return (X, Y, None)

    return (X, Y, theta)


def example_dataset(
    n_rows: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:

    """Generates data and returns features, labels and latents.

    Args:
        n_rows (int): Number of rows/samples.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame]: A dataset tuple of
            generated features, labels and latents.

    """

    features, labels, latents = logistic_model(
        x_mu=np.array([0.0, 0.0, 0.0]), size=n_rows, return_latents=True
    )

    features_df: pd.DataFrame = pd.DataFrame(
        {"age": features[:, 0], "height": features[:, 1], "ground-truth-label": labels}
    )

    predictions_series: pd.Series = pd.Series(labels, name="predicted-label")
    latents_df: pd.DataFrame = pd.DataFrame({"latents": latents})
    return (features_df, predictions_series, latents_df)
