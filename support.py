
import numpy as np
from scipy.stats import norm
from scipy.stats import entropy

# %%
def normalise(vec):
    rv = norm(0, 1)
    return rv.ppf((np.argsort(np.argsort(vec))+0.5)/len(vec))


def bin_loc(vec, bins):
    q = np.sort(vec)
    k = len(q)/bins
    loc = [int(k*b+0.5) for b in range(0, bins+1)]
    loc[-1] -= 1

    return q[loc]


def single_iter(data):
    means, corre, nsamples, nbins = data
    localI = 0
    points = np.random.multivariate_normal(means, corre, nsamples).T

    return pair_mutual_information(points[0], points[1], nbins)


def pair_mutual_information(x, y, binNo):
    assert len(x) == len(y), "x and y must have the same length"

    _nsamples = len(x)

    xbins = bin_loc(x, binNo)
    ybins = bin_loc(y, binNo)
    bins = np.array([xbins, ybins])
    pxy = np.histogram2d(x, y, bins)[0]/_nsamples
    px = np.histogram(x, xbins)[0]/_nsamples
    py = np.histogram(y, ybins)[0]/_nsamples

    hx = entropy(px)
    hy = entropy(py)
    hxy = entropy(pxy.flatten())

    return hx + hy - hxy


def surrogate(x, multivariate = True):
    """Generate a common angle surrogate of a 2D array (surrogates the first axis)."""
    if multivariate:
        rpha = np.exp(2*np.pi*np.random.rand(int(x.shape[0]/2+1))*1.0j)
        fftX1 = (np.fft.rfft(x, axis=0).T*rpha)
    else:
        rpha = np.exp(2*np.pi*np.random.rand(int(x.shape[0]/2+1), x.shape[1])*1.0j)
        fftX1 = (np.fft.rfft(x, axis=0)*rpha).T
    xs = np.fft.irfft(fftX1).T
    return xs


def task_producer(patient, Nsurrogates, multivariate = True):
    """Normalise the distribution of data and yields original and shadows"""
    _patient = np.zeros_like(patient)
    for i in range(patient.shape[1]):
        _patient[:, i] = normalise(patient[:, i])

    if multivariate:
        yield _patient
    for i in range(Nsurrogates):
        yield surrogate(_patient, multivariate)
