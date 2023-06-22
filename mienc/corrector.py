from .support import single_iter, correct_vector, fake_pool
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import glob
import json
import configparser
import socket
from warnings import warn
from typing import Union


class Corrector:
    def __init__(
        self,
        bins: int,
        duration: int,
        steps: int = None,
        iterations: int = None,
        samples: int = None,
        folder_name: str = None,
        cache_dir: Union[str, bytes, os.PathLike] = None,
        workers: int = 1,
        smoothed: bool = False,
        display: bool = False,
        retrieve: bool = True,
        config: Union[str, configparser.ConfigParser] = None,
        **kwargs
    ):
        self.bins = bins

        if config is not None:
            if isinstance(config, str) and os.path.isfile(config):
                try:
                    self.config = configparser.ConfigParser()
                    self.config.read(config)
                except:
                    self.config = None
            elif isinstance(config, configparser.ConfigParser):
                self.config = config
            else:
                self.config = None
        else:
            self.config = None

        if self.config is not None:
            self.steps = config.getint("correction", "steps", fallback=200)
            self.iterations = config.getint(
                "correction", "iters", fallback=10000)
            self.samples = config.getint("correction", "nsamples", fallback=0)
            self.cache_dir = self.config.getint(
                "correction", "cacheDir", fallback=None)

            this_host = socket.gethostname()
            if self.config.has_section(this_host):
                self.cache_dir = self.config.get(
                    this_host, "cacheDir", fallback=self.cache_dir)

            path = os.path.dirname(os.path.realpath(__file__))
            if self.cache_dir is not None and not os.path.isdir(self.cache_dir) and os.path.isdir(os.path.join(path, self.cache_dir)):
                self.cache_dir = os.path.join(path, self.cache_dir)

        if cache_dir is not None:
            self.cache_dir = cache_dir
            if self.cache_dir is not None and not os.path.isdir(self.cache_dir) and os.path.isdir(os.path.join(path, self.cache_dir)):
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
                f"Acquisition duration ({duration}) is different from the set number of samples for correction ({self.samples})."
            )
        self.folder_name = folder_name

        self.smoothed = smoothed
        self.display = display
        self.workers = workers
        if self.workers > 1:
            self.pool = mp.Pool
        else:
            self.pool = fake_pool

        self.correction = None
        self.true_value = None
        self.old_correct = np.vectorize(self._correct)
        self.correct = lambda x: correct_vector(
            x, self.correction, self.true_value)

        self.out_file = f"correction_{self.samples}_{self.bins}.npy"

        self.__retrieve(retrieve)

    def __retrieve(self, retrieve):
        self.earlyResultsPath = None

        if self.folder_name is not None and os.path.isfile(os.path.join(self.folder_name, self.out_file)):
            self.earlyResultsPath = os.path.join(
                self.folder_name, self.out_file)
            return

        if self.cache_dir is not None and os.path.isfile(os.path.join(self.cache_dir, self.out_file)):
            self.earlyResultsPath = os.path.join(
                self.cache_dir, self.out_file)
            return

        if not retrieve or self.folder_name is None:
            return

        for fold in glob.glob(os.path.abspath(os.path.join(self.folder_name, os.pardir, f"*bin{self.bins}"))):
            if os.path.isfile(os.path.join(fold, "shape.json")):
                with open(os.path.join(fold, "shape.json")) as fp:
                    shape = json.load(fp)
                    if shape[0] == self.samples:
                        if os.path.isfile(os.path.join(fold, self.out_file)):
                            print("Retrieving correction from: ", fold)
                            self.earlyResultsPath = os.path.join(
                                fold, self.out_file)
                            return

    def compute_correction(self):
        """Computes the correction lookup table or loads the cached values."""
        incre = 1 / self.steps
        correction = np.zeros(self.steps)
        if self.earlyResultsPath is None:
            with self.pool(self.workers) as pool:
                for i in tqdm(range(self.steps), "Computing correction"):
                    means = 0, 0
                    corre = [[1, i * incre], [i * incre, 1]]
                    I = pool.map(
                        single_iter,
                        (
                            (means, corre, self.samples, self.bins)
                            for __ in range(self.iterations)
                        ),
                    )
                    correction[i] = np.average(I)

            if self.smoothed:
                tosmo = np.zeros(correction.shape[0]+4)
                tosmo[:2] = correction[0]
                tosmo[2, -2] = correction
                tosmo[-2:] = correction[-1]

                self.correction = np.zeros_like(correction)
                for i in range(len(correction)):
                    self.correction[i] = np.mean(tosmo[i: i + 5])
                if self.display:
                    try:
                        plt.plot(correction[:50])
                        plt.plot(self.correction[:50])
                        plt.show()
                    except:
                        pass
            else:
                self.correction = correction
        else:
            self.correction = np.load(self.earlyResultsPath)
            correction = self.correction

        self.true_value = -0.5 * \
            np.log(1 - (np.arange(self.steps) / self.steps) ** 2)

        if self.cache_dir and not self.cache_dir in self.earlyResultsPath:
            np.save(os.path.join(
                self.cache_dir, self.out_file, self.correction))

        if self.folder_name and not self.folder_name in self.earlyResultsPath:
            np.save(os.path.join(self.folder_name,
                    self.out_file, self.correction))

        if self.folder_name or self.display:
            # this is needed to get an estimate of the size of the bias we are correcting
            weights = np.zeros_like(self.true_value)
            weights[:-1] += 0.5 * (self.true_value[1:] - self.true_value[:-1])
            weights[1:] += weights[:-1]
            deviation = np.sqrt(
                np.average(
                    np.square(correction[:] - self.true_value[:]), weights=weights)
            )

            plt.title(f"RMS correction: {deviation:.4}")
            plt.plot(self.true_value, correction)
            plt.plot(self.true_value, self.correction)
            plt.plot(
                [min(self.true_value), max(self.true_value)],
                [min(self.true_value), max(self.true_value)],
                ":k",
            )
            plt.xlabel("True MI")
            plt.ylabel("Estimated MI")
            if self.folder_name and not os.path.isfile(f"{self.folder_name}/correctionMap_{self.bins}.pdf"):
                plt.savefig(
                    f"{self.folder_name}/correctionMap_{self.bins}.pdf", bbox_inches="tight")
            if self.display:
                plt.show()
            else:
                plt.close()

    def _correct(self, I):
        index = np.argmin(np.abs(I - self.correction))
        return self.true_value[index]
