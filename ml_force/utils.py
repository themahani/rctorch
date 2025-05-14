import numpy as np


def z_transform(signal: np.ndarray):
    return (signal - signal.mean(axis=0)) / signal.std(axis=0)


def minmax_transform(signal, zero_mean: bool = False):
    min = signal.min(axis=0)
    max = signal.max(axis=0)
    minmaxed = (signal - min) / (max - min)
    if zero_mean:
        minmaxed -= minmaxed.mean(axis=0)
    return minmaxed
