from pathlib import Path
import re
from tkinter import N
import importlib

mienc = importlib.import_module("mienc")
import mienc.estimators
import pytest
import configparser, os, socket, sys, subprocess
import numpy as np
from scipy.io import savemat

DATA_DIR = Path(__file__).parent / "data"
B = np.random.normal(0, 1, [27, 10, 2])


@pytest.fixture
def results_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("results")


@pytest.fixture
def cache_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("cache")


@pytest.fixture
def config_file(results_dir):
    config = configparser.ConfigParser()
    config.read(DATA_DIR / "test_config.ini")
    config.set("global", "output_folder", str(results_dir))
    config.set("testing", "file_path", str(results_dir))
    this_host = socket.gethostname()
    config.add_section(this_host)
    config.set(this_host, "output_folder", str(results_dir))
    config.set(this_host, "workers", config.get("global", "workers"))
    A = np.random.normal(0, 1, [27, 10, 5])
    savemat(results_dir / "test_data_A.mat", {"data": A})
    config.write(open(results_dir / "config.ini", "w"))
    return str(results_dir / "config.ini")


def test_cli_version():
    # run "mienc --version"
    result = subprocess.run(
        [sys.executable, "-m", "coverage", "run", "-p", "-m", "mienc", "--version"],
        capture_output=True,
        text=True,
    )

    # check that it exits successfully
    assert result.returncode == 0

    # check that the output matches __version__
    from mienc._version import __version__

    assert __version__ in result.stdout


def test_bin_complete(config_file, cache_dir):
    nle = mienc.NonLinearEstimator(
        config_file=config_file,
        estimator="bin",
        parameter=0,
        surrogates=51,
        cache=cache_dir,
        save_out=True,
        dataset="testing",
        verbose=True,
        EC=0,
        random_state=42,
    )
    nle.estimate(
        display=False, dataset_sub="A", extended_stats=True, compute_shadow=True
    )
    nle.estimate(
        display=False,
        dataset_sub="A",
        extended_stats=True,
        compute_shadow=True,
        retrieve=True,
        parameter=3,
    )
    assert os.path.isfile(os.path.join(nle.folder_name, "session01_3.npy"))


def test_bin_fractionalParameter(config_file, cache_dir):
    nle = mienc.NonLinearEstimator(
        config_file=config_file,
        estimator="bin",
        parameter=0.1,
        surrogates=1,
        cache=cache_dir,
        save_out=3,
        dataset="testing",
        verbose=True,
        EC=0,
        random_state=42,
    )
    nle.estimate(
        display=False, dataset_sub="A", extended_stats=True, compute_shadow=True
    )
    nle2 = mienc.NonLinearEstimator(
        config_file=config_file,
        estimator="bin",
        parameter=2,
        surrogates=1,
        cache=cache_dir,
        save_out=True,
        dataset="testing",
        verbose=True,
        EC=0,
        random_state=42,
    )
    nle2.estimate(
        display=False, dataset_sub="A", extended_stats=True, compute_shadow=False
    )
    assert os.path.isfile(os.path.join(nle.folder_name, "session01_2_sha.npy"))


def test_knn_ortho(config_file, cache_dir, results_dir):
    pytest.importorskip("tigramite")
    pytest.importorskip("numba")
    config = configparser.ConfigParser()
    config.read(config_file)
    config.remove_option("testing", "field_name")
    config.write(open(results_dir / "config2.ini", "w"))

    nle = mienc.NonLinearEstimator(
        config_file=str(results_dir / "config2.ini"),
        estimator="knn",
        parameter=0,
        surrogates=2,
        cache=cache_dir,
        save_out=3,
        dataset="testing",
        verbose=True,
        EC=0,
        random_state=42,
        ortho=True,
        truncate_input=[5, 20],
    )
    with pytest.warns(RuntimeWarning, match="heuristics"):
        nle.estimate(
            display=False,
            dataset_sub="A",
            extended_stats=True,
            compute_shadow="extend",
            no_correction=True,
            save_out=False,
        )
    assert nle.folder_name is None


def test_Chatterjee_suffix(config_file, cache_dir):
    nle = mienc.NonLinearEstimator(
        config_file=config_file,
        estimator="Chatterjee",
        parameter=3,
        surrogates=2,
        cache=cache_dir,
        save_out=True,
        dataset="testing",
        verbose=True,
        EC=0,
        random_state=42,
        ortho=True,
        truncate_input=[20],
        suffix="try",
    )
    nle.estimate(
        display=False,
        dataset_sub="A",
        extended_stats=True,
        compute_shadow=True,
        no_correction=True,
        jitter=0.0,
    )
    assert os.path.isfile(os.path.join(nle.folder_name, "shape.json"))


def test_Chatterjee_Nones(config_file, cache_dir):
    nle = mienc.NonLinearEstimator(
        config_file=config_file,
        estimator="Chatterjee",
        parameter=None,
        surrogates=None,
        cache=cache_dir,
        save_out=True,
        verbose=True,
        EC=0,
        workers=None,
        ortho=None,
        jitter=None,
    )
    nle.estimate(
        display=False,
        dataset="testing",
        dataset_sub="A",
        extended_stats=True,
        compute_shadow=True,
        no_correction=True,
        truncate_input=[20],
        ortho=False,
        suffix="try",
    )
    assert os.path.isfile(os.path.join(nle.folder_name, "shape.json"))


def test_Chatterjee_wrongTruncate(config_file, cache_dir):
    nle = mienc.NonLinearEstimator(
        config_file=config_file,
        estimator="Chatterjee",
        parameter=0,
        surrogates=2,
        cache=cache_dir,
        save_out=True,
        dataset="testing",
        verbose=True,
        EC=0,
        random_state=42,
        ortho=True,
        truncate_input=[5, 20, 3],
        suffix="try",
    )
    with pytest.raises(ValueError, match="truncate_input"):
        nle.estimate(
            display=False,
            dataset_sub="A",
            extended_stats=True,
            compute_shadow=True,
            no_correction=True,
            save_out=False,
        )


def test_KNN_correct(config_file, cache_dir):
    pytest.importorskip("tigramite")
    pytest.importorskip("numba")
    nle = mienc.NonLinearEstimator(
        config_file=config_file,
        estimator="knn",
        parameter=0,
        surrogates=2,
        cache=cache_dir,
        save_out=True,
        dataset="testing",
        verbose=True,
        EC=0,
        random_state=42,
        truncate_input=10,
    )
    nle.estimate(
        display=False,
        dataset_sub="A",
        extended_stats=True,
        compute_shadow=True,
    )
    assert os.path.isfile(os.path.join(nle.folder_name, "shape.json"))


def test_dChatterjee_correct(config_file, cache_dir):
    with pytest.raises(AssertionError, match="Chatterjee"):
        nle = mienc.NonLinearEstimator(
            config_file=config_file,
            estimator="dchatt",
            parameter=0,
            surrogates=2,
            cache=cache_dir,
            save_out=True,
            dataset="testing",
            verbose=True,
            EC=0,
            random_state=42,
            truncate_input=10,
        )
    nle = mienc.NonLinearEstimator(
        config_file=config_file,
        estimator="dchatt",
        parameter=0,
        surrogates=2,
        cache=cache_dir,
        save_out=True,
        dataset="testing",
        verbose=True,
        EC=2,
        random_state=42,
        truncate_input=10,
    )
    nle.estimate(
        display=False,
        dataset_sub="A",
        extended_stats=True,
        compute_shadow=False,
    )
    assert os.path.isfile(os.path.join(nle.folder_name, "shape.json"))


def test_Corrector_bin(cache_dir):
    assert os.path.isfile(str(DATA_DIR / "test_config.ini"))
    assert os.path.isdir(str(cache_dir))
    est = mienc.estimators.BinEstimator()
    est.parameter = 4
    corrector = mienc.Corrector(
        samples=64,
        estimator=est,
        duration=64,
        workers=1,
        ensure_monotonic=False,
        steps=10,
        iterations=20,
        cache_dir=str(cache_dir),
        verbose=True,
    )
    corrector.compute_correction()
    corrector2 = mienc.Corrector(
        samples=64,
        estimator=est,
        duration=64,
        workers=1,
        ensure_monotonic=False,
        steps=10,
        iterations=20,
        cache_dir=str(cache_dir),
        verbose=False,
    )
    corrector2.compute_correction()
    assert corrector.correction is not None
    assert corrector2.correction is not None


def test_Corrector_Chatterjee(cache_dir):
    est = mienc.estimators.ChatterjeEstimator(0)
    with pytest.warns(RuntimeWarning, match="different"):
        corrector = mienc.Corrector(
            samples=25,
            estimator=est,
            duration=30,
            workers=1,
            ensure_monotonic=False,
            config=str(DATA_DIR / "test_config.ini"),
            cache_dir=str(cache_dir),
            verbose=True,
        )
    corrector.compute_correction()
    assert corrector.correction is not None


def test_Corrector_Nones(cache_dir):
    est = mienc.estimators.ChatterjeEstimator(0)
    corrector = mienc.Corrector(
        samples=30,
        estimator=est,
        duration=30,
        workers=1,
        steps=5,
        ensure_monotonic=False,
        config=None,
        cache_dir=None,
        verbose=False,
        iterations=10,
    )
    corrector.compute_correction()
    assert corrector.correction is not None


def test_NLE_Chatterjee_dist_randomOut():
    nle = mienc.NonLinearEstimator(
        estimator="dchatt",
        parameter=0,
        surrogates=2,
        save_out=True,
        verbose=False,
        retrieve=False,
        EC=1,
    )
    assert isinstance(
        nle.estimate(B, no_correction=True, extended_stats=True, compute_shadow=False),
        dict,
    )


def test_NLE_Chatterjee_partialRandomOut():
    nle = mienc.NonLinearEstimator(
        estimator="chatt",
        parameter=0,
        surrogates=2,
        save_out=1,
        verbose=False,
        retrieve=False,
        EC=1,
    )
    assert isinstance(
        nle.estimate(B, no_correction=True, extended_stats=True, compute_shadow=False),
        dict,
    )
