import multiprocessing as mp
import os
import warnings
from contextlib import contextmanager
from ctypes import POINTER, Structure, c_bool, c_double, c_int, cdll
from typing import Union

import numpy as np
from scipy.stats import norm

el_path = os.path.abspath(os.path.dirname(__file__))
_libMI = cdll.LoadLibrary(os.path.join(el_path, "../bin/libpmi.so"))
warnings.simplefilter("once", lineno=94, append=True)
warnings.simplefilter("once", category=RuntimeWarning, append=True)


class returnStats(Structure):
    _fields_ = [
        ("ratio95control", c_double),
        ("ratio99control", c_double),
        ("ratio05", c_double),
        ("ratio95", c_double),
        ("ratio99", c_double),
        ("totalMI", c_double),
        ("gaussMI", c_double),
        ("sigmaGaussMI", c_double),
    ]


_libMI.pair_mutual_information.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    c_int,
    c_int,
)
_libMI.pair_mutual_information.restype = c_double
_libMI.total_mutual_information.argtypes = (
    POINTER(c_double),
    c_int,
    c_int,
    c_int,
    POINTER(c_double),
)
_libMI.total_mutual_information.restype = None
_libMI.statistics.argtypes = (
    POINTER(c_double),
    c_int,
    c_int,
    POINTER(c_double),
    POINTER(c_double),
    c_int,
    c_int,
    c_bool,
)
_libMI.statistics.restype = returnStats
_libMI.correct_vector.argtypes = (
    POINTER(c_double),
    c_int,
    POINTER(c_double),
    POINTER(c_double),
    c_int,
    POINTER(c_double),
)
_libMI.correct_vector.restype = None
_libMI.quantile_vector.argtypes = (
    POINTER(c_double),
    c_int,
    c_int,
    POINTER(c_double),
    c_int,
    POINTER(c_double),
)
_libMI.quantile_vector.restype = None


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

    return _libMI.pair_mutual_information(
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
    _libMI.total_mutual_information(
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
    _libMI.correct_vector(
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
    tmp = _libMI.statistics(
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
    _libMI.quantile_vector(
        data.ctypes.data_as(POINTER(c_double)),
        c_int(numPairs),
        c_int(numSurrogatesPU - 1),
        quantile.ctypes.data_as(POINTER(c_double)),
        c_int(len(quantile)),
        out.ctypes.data_as(POINTER(c_double)),
    )
    return out


def surrogate(x: np.ndarray, multivariate: bool = True, extension: int = 1) -> np.ndarray:
    """Generate a common angle surrogate of a N-D array (surrogates the first axis).
    Input:
    x: an N-dimensional np.Array with time along the first axis (time x whatever x ... x whatever).
    multivariate: if True (default) applies the same random phases to all the series.
    extension: create a longer surrogate by joining extension many.
    Output:
    np.array containing the surrogate time series such that output shape matches input shape.
    """
    if x.shape[1] > x.shape[0]:
        warnings.warn(
            "It looks you have more series than timepoints, or maybe you should transpose the input.",
            RuntimeWarning,
        )
    fft = np.fft.rfft(x, axis=0)
    fftX1 = []
    extra_shape = [1] if multivariate else x.shape[1:]

    for i in range(extension):
        rpha = np.exp(2 * np.pi * np.random.rand(int(x.shape[0] / 2 + 1), *extra_shape) * 1.0j)
        fftX1.append(fft * rpha)

    xs = np.concatenate([np.fft.irfft(tmp, n=x.shape[0], axis=0) for tmp in fftX1], 0)
    return xs


def task_producer(patient, Nsurrogates, multivariate=True):
    """Normalise the distribution of data and yields original and shadows"""
    _patient = np.zeros_like(patient)
    for i in range(patient.shape[1]):
        _patient[:, i] = normalise(patient[:, i])

    if multivariate:
        yield _patient
    for i in range(Nsurrogates):
        yield surrogate(_patient, multivariate)


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
    def __init__(self) -> None:
        pass

    def imap(self, *args):
        return list(map(*args))

    def map(self, *args):
        return list(map(*args))


@contextmanager
def get_pool(workers, **kwargs):
    try:
        if workers == 1:
            yield a_normal_map()
        else:
            yield mp.Pool(workers)
    finally:
        pass
