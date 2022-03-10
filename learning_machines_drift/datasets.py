# pylint: disable=C0103

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.special import expit


def logistic_model(
    X_mu: NDArray[np.float64] = np.array([0.0, 0.0, 0.0]),
    X_scale: NDArray[np.float64] = np.array([1.0, 1.0, 1.0]),
    X_corr: NDArray[np.float64] = np.array(
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
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate synthetic features and labels.

    Features are generated from a multivariate normal distribution,
    where the mean vector, scale vector and correlation matrix can be specified,
    allowing users can simulate covariate drift.
    Labels are generated with a logistic regression model.
    The regression parameters are controlled with the `beta` parameter, allowing simulation of concept drift.

    Args:
        X_mu: Mean vector of features. Defaults to np.array([0.0, 0.0, 0.0]).
        X_scale: Scale of features. Defaults to np.array([1.0, 1.0, 1.0]).
        X_corr: Correlation matrix giving the correlation between features Defaults to np.array( [[1.0, 0.4, 0.0], [0.4, 1.0, 0.0], [0.0, 0.0, 1.0]] ).
        alpha: Regression alpha parameter. Defaults to 0.5.
        beta: Regression beta parameters . Defaults to np.array([1.0, 0.5, 0.0,]).
        size: Number of samples to draw from model. Defaults to 50.
        seed: Optionally set the numpy.random seed. Defaults to None.

    Returns:
        Tuple[NDArray[np.float64], NDArray[np.float64]]: _description_
    """
    if seed:
        np.random.seed(seed)

    # Sample features from variate normal
    X_diag: NDArray[np.float64] = np.diag(X_scale)
    X_covariance = np.dot(np.dot(X_diag, X_corr), X_diag)
    X = stats.multivariate_normal.rvs(X_mu, X_covariance, size=size)

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

    return (X, Y)
