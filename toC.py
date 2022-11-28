import numpy as np
from scipy.stats import norm
#from scipy.stats import entropy
import warnings
from typing import Union
from numba import jit, float64, int32
@jit(float64[:](float64[:]), nopython=True, nogil=True, cache=True)
def entr (x):
    y = np.zeros_like(x)
    for i, t in enumerate(x):
        if t>0:
            y[i]=-t*np.log(t)
    return y

@jit(float64(float64[:]), nopython=True, nogil=True, cache=True)
def entropy(pk) -> Union[np.number, np.ndarray]:
    pk = np.asarray(pk)
    pk = 1.0*pk / np.sum(pk, axis=0)
    vec = entr(pk)
    S = np.sum(vec, axis=0)
    return S

@jit(float64[:](float64[:], int32), nopython=True, nogil=True, cache=True)
def bin_loc(vec, bins):
    q = np.sort(vec)
    k = len(q) / bins
    loc = [int(k * b + 0.5) for b in range(0, bins + 1)]
    loc[-1] -= 1

    ret=np.zeros(len(loc))
    for i, l in enumerate(loc):
        ret[i] = q[l]
    return ret

@jit(float64(float64[:], float64[:], int32), nopython=True, nogil=True, cache=True)
def pair_mutual_information(x, y, binNo):
    assert len(x) == len(y), "x and y must have the same length"

    _nsamples = len(x)

    xbins = bin_loc(x, binNo)
    ybins = bin_loc(y, binNo)
    sepx = np.searchsorted(xbins, x)-1
    sepy = np.searchsorted(ybins, y)-1
    sepxy = (binNo*sepy)+sepx
    pxy = np.bincount(sepxy) / _nsamples
    px = np.bincount(sepx) / _nsamples
    py = np.bincount(sepy) / _nsamples

    hx = entropy(px)
    hy = entropy(py)
    hxy = entropy(pxy.flatten())

    return hx + hy - hxy