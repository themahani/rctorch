"""Normalization helpers for time series."""

import numpy as np


def z_transform(signal: np.ndarray):
    """Standardize each column to zero mean and unit variance.

    Parameters
    ----------
    signal : np.ndarray of shape (nt, n_features)
        Time series with time along the first axis.

    Returns
    -------
    np.ndarray of shape (nt, n_features)
        ``(signal - mean) / std``, with the mean and standard deviation of each column taken over time.
    """
    return (signal - signal.mean(axis=0)) / signal.std(axis=0)


def minmax_transform(signal, zero_mean: bool = False):
    """Rescale each column to the range [0, 1].

    Parameters
    ----------
    signal : np.ndarray of shape (nt, n_features)
        Time series with time along the first axis.
    zero_mean : bool, default False
        If True, subtract the mean of each rescaled column so the result is centered on zero.

    Returns
    -------
    np.ndarray of shape (nt, n_features)
        The rescaled signal.
    """
    min = signal.min(axis=0)
    max = signal.max(axis=0)
    minmaxed = (signal - min) / (max - min)
    if zero_mean:
        minmaxed -= minmaxed.mean(axis=0)
    return minmaxed
