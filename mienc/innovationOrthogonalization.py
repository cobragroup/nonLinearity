# Script implementing the innovation orthogonalisation technique by
# Pascual-Marqui et al., 2017, http://dx.doi.org/10.1101/178657
# This script is provided WITHOUT ANY WARRANTY included the one of doing what
# stated in this header.
# Written and distributed by Giulio Tani Raffaelli under CC-BY 4.0 or later.
# tani[at]cs.cas.cz

import numpy as np
import numpy.typing as npt
import warnings

try:
    from statsmodels.tsa.api import VAR
    __loaded = True
except ModuleNotFoundError:
    warnings.warn(
        "'statsmodels' module missing, impossible to fit the VAR, 'innor' won't work.")
    __loaded = False
except ImportError as e:
    warnings.warn(
        "'statsmodels' failed to load, impossible to fit the VAR, 'innor' won't work.\n"+e.msg)
    __loaded = False


def innOr(Y: npt.NDArray, verbose: bool = False, all_matrices: bool = False, **kwargs) -> npt.NDArray:
    """Applies innovation orthogonalisation (Pascual-Marqui et al., 2017) to input data.

    Parameters
    ----------
    Y : {ndarray, None}
        The input data with shape (T, S, [N]) corresponding to S time series with T samples each. The optional third dimension is the number of subjects. With 3D input all_matrices is always False.
    verbose : {bool, False}
        Outputs optional information about the optimisation.
    all_matrices : {bool, False}
        Whether to return also the VAR model coefficients and residuals, and the unmixing matrix.
    Returns
    -------
    X : ndarray
        Unmixed time series (Same shape as Y)
    coefs : ndarray
        Coefficients of the VAR model, shape: (VAR_order, S, S). Only if all_matrices == True
    residuals : ndarray
        Residuals of the VAR model, shape: (T, S). Only if all_matrices == True
    M : ndarray
        Unmixing matrix, shape: (S, S). Only if all_matrices == True

    Notes
    -----
    Pascual-Marqui et al., 2017, p. 3-5, 20, http://dx.doi.org/10.1101/178657

    """
    if not __loaded:
        warnings.warn(
            "'statsmodels' module missing, impossible to fit the VAR, returning original input.")
        return Y
    if len(Y.shape) > 2:
        ortho = np.empty_like(Y)
        for i in range(Y.shape[2]):
            ortho[:, :, i] = __ortho(Y[:, :, i], verbose, False)
        return ortho
    else:
        return __ortho(Y, verbose, all_matrices)


def __ortho(Y, verbose, all_matrices):
    VAR_mod = VAR(Y)
    res = VAR_mod.fit(ic="aic", verbose=verbose, trend='n')
    eta = res.resid

    n = 1
    old_D = np.ones((Y.shape[1], Y.shape[1]))
    D = np.identity(Y.shape[1])
    while np.linalg.norm(old_D-D) > 1e-9:
        if not n % 10 and verbose:
            print("Optimising: ", n, f"({np.linalg.norm(old_D-D)})")
        old_D = D
        L, GAMMA, Rh = np.linalg.svd(eta@D, full_matrices=False)
        V = L@Rh
        DELTA = V.T@eta
        D = np.diag(np.diag(DELTA))
        n += 1
    if verbose:
        print("Optimised: ", n, f"({np.linalg.norm(old_D-D)})")

    M = np.linalg.inv(D)@V.T@eta

    X = Y@np.linalg.inv(M)

    if all_matrices:
        return X, res.coefs, eta, M
    else:
        return X
