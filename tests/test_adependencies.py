import pytest, os, socket, sys
import builtins, configparser
from unittest import mock
import numpy as np
from scipy.io import savemat
from pathlib import Path
import importlib

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


@pytest.fixture
def no_statsmodels(monkeypatch):
    original_modules = sys.modules.copy()
    for key in list(sys.modules.keys()):
        if key.startswith("statsmodels"):
            sys.modules.pop(key)
    sys.modules.pop("mienc.innovationOrthogonalization", None)
    sys.modules.pop("mienc.innOr", None)
    # Block them so they cannot be imported
    for mod in ["statsmodels", "statsmodels.tsa", "statsmodels.tsa.api"]:
        monkeypatch.setitem(sys.modules, mod, None)

    yield

    sys.modules.clear()
    sys.modules.update(original_modules)


def test_bin_FailOrtho(config_file, cache_dir, no_statsmodels):

    mienc = importlib.import_module("mienc")

    nle = mienc.NonLinearEstimator(
        config_file=config_file,
        estimator="bin",
        parameter=0,
        surrogates=1,
        cache=cache_dir,
        save_out=True,
        dataset="testing",
        verbose=True,
        EC=0,
        random_state=42,
        ortho=True,
    )
    with pytest.warns(RuntimeWarning, match="statsmodels"):
        nle.estimate(
            display=False,
            dataset_sub="A",
            extended_stats=True,
            compute_shadow=True,
        )
    with pytest.warns(RuntimeWarning, match="missing"):
        nle.estimate(
            display=False,
            dataset_sub="A",
            extended_stats=True,
            compute_shadow=True,
        )

    assert os.path.isfile(os.path.join(nle.folder_name, "session01_3.npy"))


def test_knn_Fail(config_file, cache_dir, monkeypatch):
    sys.modules.pop("tigramite", None)

    # Patch sys.modules so any attempt to import tigramite fails
    monkeypatch.setitem(sys.modules, "tigramite", None)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tigramite":
            raise ImportError("No module named 'tigramite'")
        return real_import(name, *args, **kwargs)

    # with mock.patch.dict(sys.modules, {"tigramite": None}):
    with mock.patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(ImportError, match="tigramite"):
            mienc = importlib.import_module("mienc")

            nle = mienc.NonLinearEstimator(
                config_file=config_file,
                estimator="knn",
                parameter=0,
                surrogates=1,
                cache=cache_dir,
                save_out=True,
                dataset="testing",
                verbose=True,
                EC=0,
                random_state=42,
                ortho=True,
            )
