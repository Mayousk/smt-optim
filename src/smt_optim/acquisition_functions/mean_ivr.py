import numpy as np


def mean_ivr(mu: float, ivr: float) -> np.ndarray:
    """
    Bi-objective mean / Integrated Variance Reduction (IVR) acquisition function.

    Parameters
    ----------
    mu: float
        Mean prediction.
    ivr: float
        Integrated Variance Reduction.

    Returns
    -------
    np.ndarray
        Values (mu, -ivr) of shape (2,), both to be minimized.
    """

    return np.array([mu, -ivr])


def vec_mean_ivr(mu: np.ndarray, ivr: np.ndarray) -> np.ndarray:
    """
    Vectorized bi-objective mean / Integrated Variance Reduction (IVR) acquisition function.

    Parameters
    ----------
    mu: np.ndarray
        Mean prediction of shape (num_points, 1).
    ivr: np.ndarray
        Integrated Variance Reduction of shape (num_points, 1).

    Returns
    -------
    np.ndarray
        Values (mu, -ivr) of shape (num_points, 2).
    """

    return np.hstack((mu.reshape(-1, 1), -ivr.reshape(-1, 1)))
