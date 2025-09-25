from functools import partial
from ctypes import POINTER, c_double, c_int, c_bool
from .bindings import c_statistics
import numpy as np
from .corrector import Corrector


def f(A, B):
    return np.linalg.norm(A - B, 2)


def statistics(
    data: np.ndarray,
    extended_stats: bool,
    isEC: bool,
    pool,
    corrector: Corrector = None,
):
    if isEC:
        numSurrogatesPU = data.shape[-1]
        diff = []
        if corrector is None:
            _data = data
        else:
            _data = corrector.correct(data)
        g = partial(f, _data[:, :, 0])
        diff = pool.map(g, [_data[:, :, i] for i in range(1, numSurrogatesPU)])

        return {"mean": np.mean(diff), "std": np.std(diff)}
    else:
        numPairs, numSurrogatesPU = data.shape
        tmp = c_statistics(
            data.ctypes.data_as(POINTER(c_double)),
            c_int(numPairs),
            c_int(numSurrogatesPU - 1),
            corrector.correction.ctypes.data_as(POINTER(c_double)),
            corrector.true_value.ctypes.data_as(POINTER(c_double)),
            c_int(corrector.steps),
            c_int(pool._processes),
            c_bool(extended_stats),
        )
        return {f[0]: getattr(tmp, f[0]) for f in tmp._fields_}
