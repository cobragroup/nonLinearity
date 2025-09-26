import configparser
import glob
import json
import os
import socket
from typing import Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .support import correct_vector, get_pool
from .estimators import GenericEstimator


class Corrector:
    def __init__(
        self,
        estimator: GenericEstimator,
        duration: int,
        steps: int = None,
        iterations: int = None,
        samples: int = None,
        folder_name: str = None,
        cache_dir: Union[str, bytes, os.PathLike] = "cache",
        workers: int = 1,
        ensure_monotonic: bool = True,
        display: bool = False,
        retrieve: bool = True,
        config: Union[str, configparser.ConfigParser] = None,
        verbose: bool = False,
        no_correction: bool = False,
        **kwargs,
    ):
        self.old_correct = np.vectorize(self._correct)
        if no_correction:
            self.no_correction = True
            self.steps = 0
            return
        self.no_correction = False

        self.estimator = estimator
        if self.estimator.name == "knn":
            print(
                "Computing correction for KNN estimator is very slow and often unnecessary.\n"
                "Consider calling the NonLinearEstimator.estimate with `no_correction=True`\n"
            )
        self.verbose = verbose

        if config is not None:
            if isinstance(config, str):
                found = []
                try:
                    self.config = configparser.ConfigParser()
                    found = self.config.read(config)
                except:
                    self.config = None
                    found = []
                if found == []:
                    warn("Unable to read config file.\n", RuntimeWarning)
                    self.config = None
            elif isinstance(config, configparser.ConfigParser):
                self.config = config
            else:
                self.config = None
        else:
            self.config = None

        if self.config is not None:
            self.steps = self.config.getint("correction", "steps", fallback=200)
            self.iterations = self.config.getint("correction", "iters", fallback=10000)
            self.samples = self.config.getint("correction", "nsamples", fallback=0)
            self.cache_dir = self.config.getint("correction", "cacheDir", fallback=None)

            this_host = socket.gethostname()
            if self.config.has_section(this_host):
                self.cache_dir = self.config.get(
                    this_host, "cacheDir", fallback=self.cache_dir
                )
        else:
            self.steps = None
            self.iterations = None
            self.samples = None
            self.cache_dir = None

        if cache_dir is not None:
            self.cache_dir = cache_dir

        path = os.path.dirname(os.path.realpath(__file__))
        if (
            self.cache_dir is not None
            and not os.path.isdir(self.cache_dir)
            and os.path.isdir(os.path.join(path, self.cache_dir))
        ):
            self.cache_dir = os.path.join(path, self.cache_dir)

        if steps is not None:
            self.steps = steps
        if iterations is not None:
            self.iterations = iterations

        if samples is not None:
            self.samples = samples

        if self.samples == 0:
            self.samples = duration
        if duration != self.samples:
            warn(
                f"Acquisition duration ({duration}) is different from the set number of samples for correction ({self.samples}).",
                RuntimeWarning,
            )
        self.folder_name = folder_name

        self.ensure_monotonic = ensure_monotonic
        self.display = display
        self.workers = workers

        self.correction = None
        self.true_value = None

        self.out_file = f"correction_{self.samples}_{self.estimator.name}_{self.estimator.parameter}.npy"

        self.__retrieve(retrieve)
        assert (
            self.steps is not None
        ), "The number of quantization steps of the correction has to be specified in the config file or in the call to Corrector."

        assert (
            self.iterations is not None
        ), "The number of iterations in the sampling of Gaussian distributions for the correction has to be specified in the config file or in the call to Corrector."

    def __retrieve(self, retrieve):
        self.earlyResultsPath = None

        if self.folder_name is not None and os.path.isfile(
            os.path.join(self.folder_name, self.out_file)
        ):
            self.earlyResultsPath = os.path.join(self.folder_name, self.out_file)
            return

        if self.cache_dir is not None and os.path.isfile(
            os.path.join(self.cache_dir, self.out_file)
        ):
            self.earlyResultsPath = os.path.join(self.cache_dir, self.out_file)
            return

        if not retrieve or self.folder_name is None:
            return

        for fold in glob.glob(
            os.path.abspath(
                os.path.join(
                    self.folder_name, os.pardir, f"*{self.estimator.get_suffix()}"
                )
            )
        ):
            if os.path.isfile(os.path.join(fold, "shape.json")):
                with open(os.path.join(fold, "shape.json")) as fp:
                    shape = json.load(fp)
                    if shape[0] == self.samples:
                        if os.path.isfile(os.path.join(fold, self.out_file)):
                            if self.verbose:
                                print("Retrieving correction from: ", fold)
                            self.earlyResultsPath = os.path.join(fold, self.out_file)
                            return

    def compute_correction(self):
        """Computes the correction lookup table or loads the cached values."""
        if self.no_correction:
            self.true_value = np.full(10, np.nan)
            self.correction = np.full_like(self.true_value, np.nan)
            return

        increment = 1 / self.steps

        self.true_value = -0.5 * np.log(1 - (np.arange(self.steps) / self.steps) ** 2)
        correction = np.zeros(self.steps)
        if self.earlyResultsPath is None:
            if self.verbose:
                print(
                    f"Computing correction for estimator {self.estimator.name} with {self.samples} samples and parameter {self.estimator.parameter}."
                )
            with get_pool(self.workers) as pool:
                for i in tqdm(range(self.steps), desc="Step", disable=not self.verbose):
                    means = 0, 0
                    correlation_matrix = [[1, i * increment], [i * increment, 1]]
                    I = pool.map(
                        self.estimator.single_iter,
                        (
                            (means, correlation_matrix, self.samples)
                            for __ in range(self.iterations)
                        ),
                    )
                    correction[i] = np.average(I)

            last_decreasing = np.argmax(np.cumsum(np.diff(correction) < 0))
            if self.ensure_monotonic and last_decreasing > 0:
                fit_order = 1 if last_decreasing < 10 else 2
                fit = np.polyfit(
                    self.true_value[: last_decreasing + 3],
                    correction[: last_decreasing + 3],
                    fit_order,
                )

                self.correction = correction.copy()
                self.correction[: last_decreasing + 3] = np.polyval(
                    fit, self.true_value[: last_decreasing + 3]
                )
                if self.folder_name or self.display:
                    corr_fig = plt.figure(figsize=(4, 4))
                    try:
                        plt.plot(
                            correction[: last_decreasing + 3],
                            self.true_value[: last_decreasing + 3],
                            ".-",
                            label="Original",
                            lw=1,
                        )
                        plt.plot(
                            self.correction[: last_decreasing + 3],
                            self.true_value[: last_decreasing + 3],
                            ".-",
                            label="Monotonic",
                            lw=1,
                        )
                        plt.ylabel("True MI")
                        plt.xlabel("Estimated MI")
                        plt.legend()
                        if self.folder_name:
                            plt.savefig(
                                f"{self.folder_name}/monotonisationMap_{self.estimator.name}_{self.estimator.parameter}.pdf",
                                bbox_inches="tight",
                            )
                        if self.display:
                            plt.show(corr_fig.number)
                        else:
                            plt.close(corr_fig.number)
                    except:
                        pass
            else:
                self.correction = correction
        else:
            if self.verbose:
                print(
                    f"Loading correction for estimator {self.estimator.name} with {self.samples} samples and parameter {self.estimator.parameter}."
                )
            self.correction = np.load(self.earlyResultsPath)
            correction = self.correction

        if (
            self.cache_dir
            and not self.cache_dir == self.earlyResultsPath
            and not os.path.isfile(os.path.join(self.cache_dir, self.out_file))
        ):
            np.save(os.path.join(self.cache_dir, self.out_file), self.correction)

        if (
            self.folder_name
            and not self.folder_name == self.earlyResultsPath
            and self.folder_name != "in_memory"
            and not os.path.isfile(os.path.join(self.folder_name, self.out_file))
        ):
            np.save(os.path.join(self.folder_name, self.out_file), self.correction)

        if self.folder_name or self.display:
            # this is needed to get an estimate of the size of the bias we are correcting
            weights = np.zeros_like(self.true_value)
            weights[:-1] += 0.5 * (self.true_value[1:] - self.true_value[:-1])
            weights[1:] += weights[:-1]
            deviation = np.sqrt(
                np.average(
                    np.square(correction[:] - self.true_value[:]), weights=weights
                )
            )

            corr_fig2 = plt.figure(figsize=(4, 4))
            plt.title(f"RMS correction: {deviation:.4}")
            plt.plot(correction, self.true_value)
            plt.plot(self.correction, self.true_value)
            plt.plot(
                [min(self.true_value), max(self.true_value)],
                [min(self.true_value), max(self.true_value)],
                ":k",
            )
            plt.ylabel("True MI")
            plt.xlabel("Estimated MI")
            if (
                self.folder_name
                and self.folder_name != "in_memory"
                and not os.path.isfile(
                    f"{self.folder_name}/correctionMap_{self.estimator.name}_{self.estimator.parameter}.pdf"
                )
            ):
                plt.savefig(
                    f"{self.folder_name}/correctionMap_{self.estimator.name}_{self.estimator.parameter}.pdf",
                    bbox_inches="tight",
                )
            if self.display:
                plt.show(corr_fig2.number)
            else:
                plt.close(corr_fig2.number)

    def _correct(self, I):
        if self.no_correction:
            return I
        index = np.argmin(np.abs(I - self.correction))
        return self.true_value[index]

    def correct(self, vec):
        if self.no_correction:
            return vec.copy()
        return correct_vector(vec, self.correction, self.true_value)
