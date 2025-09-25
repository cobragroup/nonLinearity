from ctypes import POINTER, Structure, c_bool, c_double, c_int, CDLL
import importlib.resources as resources
import sys


def load_lib():
    """Locate and load the bundled C++ shared library with ctypes."""
    lib_dir = resources.files("mienc._libs")

    # Platform-specific names
    candidates = []
    if sys.platform.startswith("win"):
        candidates.append("pmi.dll")
    elif sys.platform == "darwin":
        candidates.append("libpmi.dylib")
    else:
        candidates.append("libpmi.so")

    for name in candidates:
        lib_path = lib_dir / name
        if lib_path.exists():
            return CDLL(str(lib_path))

    raise ImportError(
        f"Could not find compiled library. Looked for: {candidates} in {lib_dir}"
    )


_libMI = load_lib()


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


_libMI.binning_pair_mutual_information.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    c_int,
    c_int,
)
_libMI.binning_pair_mutual_information.restype = c_double
_libMI.binning_total_mutual_information.argtypes = (
    POINTER(c_double),
    c_int,
    c_int,
    c_int,
    POINTER(c_double),
)
_libMI.binning_total_mutual_information.restype = None
_libMI.pair_Chatterjee.argtypes = (
    POINTER(c_double),
    POINTER(c_double),
    c_int,
    c_int,
    c_int,
    c_bool,
)
_libMI.pair_Chatterjee.restype = c_double
_libMI.total_Chatterjee.argtypes = (
    POINTER(c_double),
    c_int,
    c_int,
    c_int,
    c_bool,
    POINTER(c_double),
)
_libMI.total_Chatterjee.restype = None
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

c_binning_pair_mutual_information = _libMI.binning_pair_mutual_information
c_binning_total_mutual_information = _libMI.binning_total_mutual_information
c_pair_Chatterjee = _libMI.pair_Chatterjee
c_total_Chatterjee = _libMI.total_Chatterjee
c_statistics = _libMI.statistics
c_correct_vector = _libMI.correct_vector
c_quantile_vector = _libMI.quantile_vector
