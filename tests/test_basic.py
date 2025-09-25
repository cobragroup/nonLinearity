import mienc
import mienc.estimators
import numpy as np

A = np.random.normal(0, 1, [27, 10])
B = np.random.normal(0, 1, [27, 10, 2])


def test_version_matches():
    # __version__ must be a non-empty string
    assert isinstance(mienc.__version__, str)
    assert mienc.__version__ != ""


def test_binning():
    est = mienc.estimators.BinEstimator()
    est.parameter = 3
    assert est.total_mutual_information(A).shape == (45,)


def test_knn():
    est = mienc.estimators.KNNEstimator(0)
    est.parameter = 3
    assert est.total_mutual_information(A).shape == (45,)


def test_knn_EC():
    est = mienc.estimators.KNNEstimator(1)
    est.parameter = 3
    assert est.total_mutual_information(A).shape == (10, 10)


def test_Chatterje():
    est = mienc.estimators.ChatterjeEstimator(0)
    assert est.total_mutual_information(A).shape == (45,)


def test_Chatterje_EC():
    est = mienc.estimators.ChatterjeEstimator(1)
    assert est.total_mutual_information(A).shape == (10, 10)


def test_Chatterje_dist():
    est = mienc.estimators.ChatterjeEstimator(1)
    assert est.total_mutual_information(A).shape == (10, 10)


def test_NLE_binning():
    nle = mienc.NonLinearEstimator(
        estimator="bin",
        parameter=0,
        surrogates=2,
        save_out="in_memory",
        verbose=False,
        retrieve=False,
        EC=0,
    )
    assert isinstance(
        nle.estimate(B, no_correction=True, extended_stats=True, compute_shadow=True),
        dict,
    )


def test_NLE_knn():
    nle = mienc.NonLinearEstimator(
        estimator="knn",
        parameter=0,
        surrogates=2,
        save_out="in_memory",
        verbose=False,
        retrieve=False,
        EC=0,
    )
    assert isinstance(
        nle.estimate(B, no_correction=True, extended_stats=True, compute_shadow=True),
        dict,
    )


def test_NLE_knn_EC():
    nle = mienc.NonLinearEstimator(
        estimator="knn",
        parameter=0,
        surrogates=2,
        save_out="in_memory",
        verbose=False,
        retrieve=False,
        EC=1,
    )
    assert isinstance(
        nle.estimate(B, no_correction=True, extended_stats=True, compute_shadow=True),
        dict,
    )


def test_NLE_Chatterjee():
    nle = mienc.NonLinearEstimator(
        estimator="chatt",
        parameter=0,
        surrogates=2,
        save_out="in_memory",
        verbose=False,
        retrieve=False,
        EC=0,
    )
    assert isinstance(
        nle.estimate(B, no_correction=True, extended_stats=True, compute_shadow=True),
        dict,
    )


def test_NLE_Chatterjee_EC():
    nle = mienc.NonLinearEstimator(
        estimator="chatt",
        parameter=0,
        surrogates=2,
        save_out="in_memory",
        verbose=False,
        retrieve=False,
        EC=1,
    )
    assert isinstance(
        nle.estimate(B, no_correction=True, extended_stats=True, compute_shadow=True),
        dict,
    )


def test_NLE_Chatterjee_dist():
    nle = mienc.NonLinearEstimator(
        estimator="dchatt",
        parameter=0,
        surrogates=2,
        save_out="in_memory",
        verbose=False,
        retrieve=False,
        EC=1,
    )
    assert isinstance(
        nle.estimate(B, no_correction=True, extended_stats=True, compute_shadow=True),
        dict,
    )
