"""TODO PEP 257"""
# pylint: disable=C0103
# pylint: disable=W0621
# pylint: disable=R0913

from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.special import expit

# pylint: disable=too-many-instance-attributes


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
) -> Union[
    Tuple[NDArray[np.float64], NDArray[np.float64]],
    Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
]:
    # pylint: disable=too-many-instance-attributes
    """Generate synthetic features and labels.

    Features are generated from a multivariate normal distribution,
    where the mean vector, scale vector and correlation matrix can
     be specified,
    allowing users can simulate covariate drift.
    Labels are generated with a logistic regression model.
    The regression parameters are controlled with the `beta` parameter,
    allowing simulation of concept drift.

    Args:
        x_mu: Mean vector of features. Defaults to np.array([0.0, 0.0, 0.0]).
        x_scale: Scale of features. Defaults to np.array([1.0, 1.0, 1.0]).
        x_corr: Correlation matrix giving the correlation between features
        Defaults to np.array( [[1.0, 0.4, 0.0], [0.4, 1.0, 0.0], [0.0, 0.0, 1.0]] ).
        alpha: Regression alpha parameter. Defaults to 0.5.
        beta: Regression beta parameters . Defaults to np.array([1.0, 0.5, 0.0,]).
        size: Number of samples to draw from model. Defaults to 50.
        return_latents (bool): Return underlying prediction value before thresholding
            as 'latent' data. Defaults to False.


    Returns:
        Tuple[NDArray[np.float64], NDArray[np.float64]]: _description_
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
        return (X, Y)

    return (X, Y, theta)
