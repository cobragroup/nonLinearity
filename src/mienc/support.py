import multiprocessing as mp
import os
import warnings
from contextlib import contextmanager
from ctypes import POINTER, c_double, c_int, c_bool
from typing import Union
from .bindings import (
    c_pair_mutual_information,
    c_total_mutual_information,
    c_statistics,
    c_correct_vector,
    c_quantile_vector,
)

import numpy as np
from scipy.stats import norm

el_path = os.path.abspath(os.path.dirname(__file__))
warnings.simplefilter("once", lineno=94, append=True)
warnings.simplefilter("once", category=RuntimeWarning, append=True)


def normalise(vec):
    rv = norm(0, 1)
    return rv.ppf((np.argsort(np.argsort(vec)) + 0.5) / len(vec))


def bin_loc(vec, bins):
    q = np.sort(vec)
    k = len(q) / bins
    loc = [int(k * b + 0.5) for b in range(0, bins + 1)]
    loc[-1] -= 1

    return q[loc]


def single_iter(data):
    means, corre, nsamples, nbins = data
    points = np.random.multivariate_normal(means, corre, nsamples).T.copy()

    return pair_mutual_information(points[0], points[1], nbins)


def pair_mutual_information(x, y, binNo):
    assert len(x) == len(y), "x and y must have the same length"
    _nsamples = len(x)
    x = np.require(x, np.float64, "FA")
    y = np.require(y, np.float64, "FA")

    return c_pair_mutual_information(
        x.ctypes.data_as(POINTER(c_double)),
        y.ctypes.data_as(POINTER(c_double)),
        c_int(_nsamples),
        c_int(binNo),
    )


def total_mutual_information(data, binNo=None):
    if binNo is None:
        data, binNo = data
    data = np.require(data, np.float64, "FA")
    times, regions = data.shape
    totPairs = int(regions * (regions - 1) / 2)
    out = np.zeros(totPairs, dtype=np.float64)
    c_total_mutual_information(
        data.ctypes.data_as(POINTER(c_double)),
        c_int(times),
        c_int(regions),
        c_int(binNo),
        out.ctypes.data_as(POINTER(c_double)),
    )
    return out


def correct_vector(data: np.ndarray, estim: np.ndarray, actual: np.ndarray):
    data = np.require(data, np.float64, "FA")
    bins = len(estim)
    totPairs = data.size
    out = np.zeros_like(data, dtype=np.float64)
    c_correct_vector(
        data.ctypes.data_as(POINTER(c_double)),
        c_int(totPairs),
        estim.ctypes.data_as(POINTER(c_double)),
        actual.ctypes.data_as(POINTER(c_double)),
        c_int(bins),
        out.ctypes.data_as(POINTER(c_double)),
    )
    return out


def statistics(
    data: np.ndarray,
    estim: np.ndarray,
    actual: np.ndarray,
    numThreads: int,
    extended_stats: bool,
):
    numPairs, numSurrogatesPU = data.shape
    bins = len(estim)
    tmp = c_statistics(
        data.ctypes.data_as(POINTER(c_double)),
        c_int(numPairs),
        c_int(numSurrogatesPU - 1),
        estim.ctypes.data_as(POINTER(c_double)),
        actual.ctypes.data_as(POINTER(c_double)),
        c_int(bins),
        c_int(numThreads),
        c_bool(extended_stats),
    )
    return {f[0]: getattr(tmp, f[0]) for f in tmp._fields_}


def quantile_vector(data: np.ndarray, quantile: Union[float, np.ndarray]):
    if not isinstance(quantile, np.ndarray):
        if not hasattr(quantile, "__len__"):
            quantile = [quantile]
        quantile = np.array(quantile)
    numPairs, numSurrogatesPU = data.shape
    out = np.zeros((numPairs, len(quantile)))
    c_quantile_vector(
        data.ctypes.data_as(POINTER(c_double)),
        c_int(numPairs),
        c_int(numSurrogatesPU - 1),
        quantile.ctypes.data_as(POINTER(c_double)),
        c_int(len(quantile)),
        out.ctypes.data_as(POINTER(c_double)),
    )
    return out


def surrogate(
    x: np.ndarray,
    multivariate: bool = True,
    extension: int = 1,
    random_state: Union[np.random.Generator, int] = None,
) -> np.ndarray:
    """Generate a common angle surrogate of a N-D array (surrogates the first axis).
    Input:
    x: an N-dimensional np.Array with time along the first axis (time x whatever x ... x whatever).
    multivariate: if True (default) applies the same random phases to all the series.
    extension: create a longer surrogate by joining extension many.
    Output:
    np.array containing the surrogate time series such that output shape matches input shape.
    """
    rng = np.random.default_rng(random_state)
    if x.shape[1] > x.shape[0]:
        warnings.warn(
            "It looks you have more series than samples, or maybe you should transpose the input.",
            RuntimeWarning,
        )
    fft = np.fft.rfft(x, axis=0)
    fftX1 = []
    extra_shape = [1] if multivariate else x.shape[1:]

    for i in range(extension):
        rpha = np.exp(
            2 * np.pi * rng.random([int(x.shape[0] / 2 + 1), *extra_shape]) * 1.0j
        )
        fftX1.append(fft * rpha)

    xs = np.concatenate([np.fft.irfft(tmp, n=x.shape[0], axis=0) for tmp in fftX1], 0)
    return xs


def task_producer(
    session,
    Nsurrogates,
    multivariate=True,
    random_state: Union[np.random.Generator, int] = None,
):
    """Normalise the distribution of data and yields original and shadows"""
    _session = np.zeros_like(session)
    for i in range(session.shape[1]):
        _session[:, i] = normalise(session[:, i])

    if multivariate:
        yield _session
    for i in range(Nsurrogates):
        yield surrogate(_session, multivariate, random_state=random_state)


def adjust_jitter(jitter):
    if isinstance(jitter, str):
        try:
            jitter = float(jitter)
        except:
            jitter = bool(jitter)
    if isinstance(jitter, bool):
        if jitter:
            jitter = 1e-3
    else:
        if np.isclose(jitter, 0):
            jitter = False
    return jitter


class a_normal_map:
    def __init__(self, *args) -> None:
        pass

    def imap(self, *args):
        return map(*args)

    def map(self, *args):
        return list(map(*args))

    def close(self, *args):
        pass


@contextmanager
def get_pool(workers, **kwargs):
    if workers == 1:
        pool = a_normal_map()
    else:
        pool = mp.Pool(workers)

    try:
        yield pool
    finally:
        pool.close()
