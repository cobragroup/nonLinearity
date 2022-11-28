import numpy as np
from scipy.stats import norm
from scipy.stats import entropy
import warnings
import os
from ctypes import Structure, addressof, byref, c_char, c_char_p, c_int, c_long, c_uint16, c_uint32, c_ushort, c_void_p, cdll, c_ulong, POINTER, cast, c_char_p, c_double
el_path = os.path.abspath(os.path.dirname(__file__))
_libMI = cdll.LoadLibrary(os.path.join(el_path, 'bin/libpmi.so'))
_libMI.pair_mutual_information.argtypes = (POINTER(c_double), POINTER(c_double), c_int, c_int)
_libMI.pair_mutual_information.restype = c_double

def normalise(vec):
    rv = norm(0, 1)
    return rv.ppf((np.argsort(np.argsort(vec)) + 0.5) / len(vec))


def bin_loc(vec, bins):
    q = np.sort(vec)
    k = len(q) / bins
    loc = [int(k * b + 0.5) for b in range(0, bins + 1)]
    loc[-1] -= 1

    return q[loc]


#%%
def single_iter(data):
    means, corre, nsamples, nbins = data
    points = np.random.multivariate_normal(means, corre, nsamples).T

    return pair_mutual_information(points[0], points[1], nbins)


def pair_mutual_information(x, y, binNo):
    assert len(x) == len(y), "x and y must have the same length"
    _nsamples = len(x)
    
    return _libMI.pair_mutual_information(x.ctypes.data_as(POINTER(c_double)), y.ctypes.data_as(POINTER(c_double)), c_int(_nsamples), c_int(binNo))


def surrogate(x, multivariate=True):
    """Generate a common angle surrogate of a N-D array (surrogates the first axis).
        Input:
        x: an N-dimensional np.Array with time along the first axis (time x whatever x ... x whatever).
        multivariate: if True (default) applies the same random phases to all the series.
        Output:
        np.array containing the surrogate time series such that output shape matches input shape."""
    if x.shape[1] > x.shape[0]:
        warnings.warn("It looks you have more series than timepoints, or maybe you should transpose the input.")
    if multivariate:
        rpha = np.exp(2 * np.pi * np.random.rand(int(x.shape[0] / 2 + 1)) * 1.0j)
        fftX1 = np.fft.rfft(x, axis=0).T * rpha
    else:
        rpha = np.exp(
            2 * np.pi * np.random.rand(int(x.shape[0] / 2 + 1), *x.shape[1:]) * 1.0j
        )
        fftX1 = (np.fft.rfft(x, axis=0) * rpha).T
    xs = np.fft.irfft(fftX1, n=x.shape[0]).T
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
