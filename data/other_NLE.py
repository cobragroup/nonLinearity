#!/usr/bin/env python3
# This package contains some useful code

import configparser
import json
import os
import socket
from multiprocessing.pool import Pool as pool_type
from typing import Literal, Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from .corrector import Corrector
from .innovationOrthogonalization import innOr
from .support import (
    a_normal_map,
    adjust_jitter,
    get_pool,
    statistics,
    surrogate,
    task_producer,
    total_mutual_information,
)

path = os.path.dirname(os.path.realpath(__file__))


class NonLinearEstimator:
    statsNames = [
        "totalMI",
        "gaussMI",
        "sigmaGaussMI",
        "ratio95control",
        "ratio99control",
        "ratio05",
        "ratio95",
        "ratio99",
    ]

    def __init__(
        self,
        config_file: Union[str, bytes, os.PathLike] = None,
        knn: int = None,
        surrogates: int = None,
        cache: Union[str, bytes, os.PathLike] = "cache",
        save_out: Union[
            bool, str, bytes, os.PathLike, int, Literal["in_memory"]
        ] = False,
        suffix: str = "",
        retrieve: bool = True,
        jitter: Union[bool, float] = False,
        ortho: bool = False,
        dataset: str = None,
        dataset_sub: str = "",
        truncate_input: int = None,
        workers: int = None,
        verbose: bool = False,
        EC: bool = False,
        random_state: Union[np.random.Generator, int] = None,
    ):
        self.suffix = suffix
        self.retrieve = retrieve
        self.dataset = dataset
        self.dataset_sub = dataset_sub
        self.truncate_input = truncate_input
        self.cacheDir = cache
        self.save_out = save_out
        self.verbose = verbose
        self.EC = EC
        self.first_session = 0
        if self.EC:
            self.statsNames = ["mean", "std"]
        self.__read_config(config_file)

        self.random_state = np.random.default_rng(random_state)

        if ortho is not None:
            self.ortho = ortho

        if knn is not None:
            self.knn = knn
        if surrogates is not None:
            self.surrogates = surrogates
        if workers is not None:
            self.workers = workers
        if jitter is not None:
            self.jitter = adjust_jitter(jitter)

        assert (
            self.surrogates is not None
        ), "Number of surrogates undefined, can't create the NonLinearEstimator"
        if self.workers is None:
            self.workers = 1

    def __read_config(self, config_file):
        if config_file is None:
            config_file = os.path.join(path, "config.ini")
        try:
            self.config = configparser.ConfigParser()
            self.config.read(config_file)

            self.ortho = self.config.getboolean(
                "global", "orthogonalise", fallback=False
            )
            self.jitter = self.config.get("global", "jitter", fallback="0.")
            self.display = self.config.getboolean("global", "display", fallback=True)
            self.workers = self.config.getint("global", "workers", fallback=4)
            self.output_folder = self.config.get(
                "global", "output_folder", fallback=".."
            )
            self.knn = self.config.getint("global", "bins", fallback=8)
            self.surrogates = self.config.getint("global", "surrogates", fallback=99)

            thisHost = socket.gethostname()
            if self.config.has_section(thisHost):
                self.workers = self.config.getint(
                    thisHost, "workers", fallback=self.workers
                )
                self.output_folder = self.config.get(
                    thisHost, "output_folder", fallback=self.output_folder
                )
            if not os.path.isabs(self.output_folder):
                self.output_folder = os.path.abspath(
                    os.path.join(path, self.output_folder)
                )
        except Exception as e:
            warn("Unable to read config file.\n" + "\n".join(e.args))
            self.config = None
            self.ortho = None
            self.jitter = None
            self.display = None
            self.workers = None
            self.output_folder = None
            self.knn = None

    def __read_config_dataset(
        self, dataset_sub=None, truncate_input=None, dataset=None, **kwargs
    ):
        if dataset is not None:
            self.dataset = dataset
        if dataset_sub is not None:
            self.dataset_sub = dataset_sub
        if truncate_input is not None:
            self.truncate_input = truncate_input
        assert (
            self.config is not None
        ), "When not passing data directly, a valid config.ini file is necessary."
        self.config["DEFAULT"]["dataset_sub"] = self.dataset_sub

        if self.dataset is None:
            self.dataset = self.config.get("global", "dataset", fallback=None)
        assert self.dataset is not None, "Unspecified dataset in .ini file."
        assert self.config.has_section(
            self.dataset
        ), "The details for the specified dataset are missing in .ini file."

        file_path = self.config.get(self.dataset, "file_path", fallback=None)
        assert file_path is not None, "Missing dataset file path in .ini file."

        file_name = self.config.get(self.dataset, "file_name", fallback=None)
        assert file_name is not None, "Missing dataset filename in .ini file."

        self.field_name = self.config.get(self.dataset, "field_name", fallback=None)
        if self.field_name is None:
            warn("Missing dataset fieldname in .ini file. Trying with heuristics.")

        hc_start = self.config.getint(
            self.dataset, "healthy_control_start", fallback=None
        )
        hc_start = hc_start if hc_start else None
        hc_end = self.config.getint(self.dataset, "healthy_control_end", fallback=None)
        hc_end = hc_end if hc_end else None
        self.hc_slice = slice(hc_start, hc_end)
        self.first_session = hc_start if hc_start else 0

        self.file_name = os.path.join(file_path, file_name)
        assert os.path.isfile(
            self.file_name
        ), f"Missing dataset at specified path: {self.file_name}."
        if self.verbose:
            print(f"Using: {os.path.abspath(self.file_name)}")

    def load_data(self, data=None, jitter=None, ortho=None, **kwargs):
        self.global_stats = None
        if ortho is not None:
            self.ortho = ortho
        if jitter is not None:
            self.jitter = adjust_jitter(jitter)
        if data is None:
            self.__read_config_dataset(**kwargs)
            if self.field_name is None:
                tmp_mat = sio.loadmat(self.file_name)
                self.field_name = [
                    k
                    for k in tmp_mat.keys()
                    if k not in ["__header__", "__version__", "__globals__"]
                ][0]
                tmp_mat = tmp_mat[self.field_name]
            else:
                tmp_mat = sio.loadmat(self.file_name)[self.field_name]
            if hasattr(self.truncate_input, "__len__"):
                if len(self.truncate_input) == 2:
                    ti_start = self.truncate_input[0]
                    ti_end = self.truncate_input[1]
                elif len(self.truncate_input) == 1:
                    ti_start = None
                    ti_end = self.truncate_input[0]
                else:
                    raise ValueError("If truncate_input should have len == 1 or 2.")
            else:
                ti_start = None
                ti_end = self.truncate_input
            truncate_slice = slice(ti_start, ti_end)
            tmp_mat = tmp_mat[truncate_slice, :, self.hc_slice]
        else:
            self.file_name = None
            tmp_mat = data
        tmp_mat = np.squeeze(tmp_mat)
        if len(tmp_mat.shape) != 3:
            if len(tmp_mat.shape) == 2:
                tmp_mat = tmp_mat[:, :, np.newaxis]
            else:
                raise RuntimeError(
                    "The number of effective dimensions of input data ({}) can't be constrained to samples x series (x sessions) format.".format(
                        len(tmp_mat.shape)
                    )
                )

        if self.ortho:
            try:
                self.mat = innOr(tmp_mat, **kwargs)
            except np.linalg.LinAlgError as e:
                raise RuntimeError(*e.args)
        else:
            self.mat = tmp_mat

        if self.jitter:
            diffs = np.diff(np.sort(self.mat[:, 0, 0]))
            spa = np.min(diffs[diffs > 0])
            self.mat += self.random_state.normal(0, spa * self.jitter, self.mat.shape)

        self.duration, self.series, self.sessions = self.mat.shape

        if self.verbose:
            print(
                "Loaded data matrix: {} samples by {} series by {} sessions".format(
                    self.duration, self.series, self.sessions
                )
            )

        if self.knn == 0:
            self.knn = int(self.duration ** (1 / 3))
        if self.knn < 1:
            self.knn = int(self.duration * self.knn)

        self.pairNum = int((self.series * (self.series - 1)) / 2)
        if self.stop_saving is True:
            self.stop_saving = self.sessions

    def __output_folder(self, suffix=None, **kwargs):
        if suffix is not None:
            self.suffix = suffix

        if self.file_name is not None:
            nameParts = [os.path.splitext(os.path.split(self.file_name)[1])[0]]
        else:
            if isinstance(self.save_out, bool):
                nameParts = ["".join(map(chr, self.random_state.integers(65, 91, 7)))]
            else:
                nameParts = [str(self.save_out)]
                self.save_out = True
        if self.suffix:
            nameParts.append(str(self.suffix))
        if self.EC:
            nameParts.append("EC")
        nameParts.append(f"{self.knn}nn")
        folder_name = "_".join(nameParts)

        if not os.path.isabs(folder_name) and self.output_folder is not None:
            self.folder_name = os.path.abspath(
                os.path.join(self.output_folder, folder_name)
            )

        if not os.path.isdir(self.folder_name):
            os.makedirs(self.folder_name)
        if self.verbose:
            print(f"Output saved in: {self.folder_name}")

    def estimate(
        self,
        data=None,
        knn=None,
        save_out=None,
        retrieve=None,
        display=None,
        truncate_input=None,
        **kwargs,
    ):
        """KWARGS INCLUDE
        jitter=None, ortho=None for load_data
        dataset_sub=None, truncate_input=None, dataset=None for __read_config_dataset
        verbose: bool = False, all_matrices: bool = False for innOr
        suffix=None for __output_folder
        extended_stats=False, compute_shadow: Union[bool, Literal["extend"]] = False for _do_estimate
        steps: int = None, iterations: int = None, samples: int = None, ensure_monotonic: bool = True, no_correction: bool = False for Corrector
        """
        assert (data is not None) != bool(
            self.dataset or ("dataset" in kwargs and bool(kwargs["dataset"]))
        ), f"You can't pass data and a dataset name at the same time. Data: {len(data.shape)}-D array, dataset: '{self.dataset if self.dataset else kwargs['dataset']}'."
        if save_out is not None:
            self.save_out = save_out

        if knn is not None:
            self.knn = knn

        if self.knn is None:
            self.knn = 0

        if isinstance(self.save_out, int):
            self.stop_saving = self.save_out
        else:
            self.stop_saving = bool(self.save_out)
        self.base_save_out = self.stop_saving

        if retrieve is not None:
            self.retrieve = retrieve
        if display is not None:
            self.display = display
        if truncate_input is not None:
            self.truncate_input = truncate_input

        self.load_data(data=data, **kwargs)

        if self.save_out:
            if self.save_out == "in_memory":
                self.out_data = {"input_shape": self.mat.shape}
                self.folder_name = "in_memory"
            else:
                self.__output_folder(**kwargs)
                with open(os.path.join(self.folder_name, "shape.json"), "w") as fp:
                    json.dump(self.mat.shape, fp)
        else:
            self.folder_name = None

        self.save_out = bool(self.save_out)

        return self._do_estimate(**kwargs)

    def _single_session_numeric(
        self, session: int, pool: Union[pool_type, a_normal_map], compute_shadow: bool
    ):
        if self.folder_name is not None and self.folder_name != "in_memory":
            base_output_name = f"session{self.first_session + session:02}_{self.knn}"
            base_output_path = os.path.join(self.folder_name, base_output_name)

        if (
            self.folder_name is not None
            and self.folder_name != "in_memory"
            and os.path.isfile(base_output_path + ".npy")
        ):
            true_and_surrogate_MI = np.load(base_output_path + ".npy")
        else:
            if self.EC:
                true_and_surrogate_MI = np.zeros(
                    [self.series, self.series, self.surrogates + 1]
                )
            else:
                true_and_surrogate_MI = np.zeros([self.pairNum, self.surrogates + 1])
            for ns, tmi in enumerate(
                pool.imap(
                    total_mutual_information,
                    (
                        (session_mat, self.knn, self.EC)
                        for session_mat in task_producer(
                            self.mat[:, :, session],
                            self.surrogates,
                            random_state=self.random_state,
                        )
                    ),
                )
            ):
                if self.EC:
                    true_and_surrogate_MI[:, :, ns] = tmi
                else:
                    true_and_surrogate_MI[:, ns] = tmi[np.triu_indices(self.series, 1)]
            if self.save_out:
                if self.folder_name == "in_memory":
                    self.out_data[session] = {"MI": np.squeeze(true_and_surrogate_MI)}
                else:
                    np.save(base_output_path + ".npy", true_and_surrogate_MI)

        if compute_shadow:
            if (
                self.folder_name is not None
                and self.folder_name != "in_memory"
                and os.path.isfile(base_output_path + "_sha.npy")
            ):
                true_and_surrogate_MI_shadow = np.load(base_output_path + "_sha.npy")
            else:
                shadow_mat = surrogate(
                    self.mat[:, :, session],
                    multivariate=True,
                    extension=compute_shadow,
                    random_state=self.random_state,
                )
                if self.EC:
                    true_and_surrogate_MI_shadow = np.zeros(
                        [self.series, self.series, self.surrogates + 1]
                    )
                else:
                    true_and_surrogate_MI_shadow = np.zeros(
                        [self.pairNum, self.surrogates + 1]
                    )
                for ns, tmi in enumerate(
                    pool.imap(
                        total_mutual_information,
                        (
                            (session_mat, self.knn, self.EC)
                            for session_mat in task_producer(
                                shadow_mat[:, :],
                                self.surrogates,
                                random_state=self.random_state,
                            )
                        ),
                    )
                ):
                    if self.EC:
                        true_and_surrogate_MI_shadow[:, :, ns] = tmi
                    else:
                        true_and_surrogate_MI_shadow[:, ns] = tmi[
                            np.triu_indices(self.series, 1)
                        ]

                if self.save_out:
                    if self.folder_name == "in_memory":
                        self.out_data[session]["MI_shadow"] = np.squeeze(
                            true_and_surrogate_MI_shadow
                        )
                    else:
                        np.save(
                            base_output_path + "_sha.npy", true_and_surrogate_MI_shadow
                        )
        else:
            true_and_surrogate_MI_shadow = None

        if (
            self.folder_name is not None
            and self.folder_name != "in_memory"
            and os.path.isfile(base_output_path + "_cor.npy")
        ):
            correlation = np.load(base_output_path + "_cor.npy")
        else:
            for normalised in task_producer(
                self.mat[:, :, session], 0, random_state=self.random_state
            ):
                correlation = np.corrcoef(normalised, rowvar=False)[
                    np.triu_indices(self.series, 1)
                ]
            if self.save_out:
                if self.folder_name == "in_memory":
                    self.out_data[session]["correlation"] = correlation
                else:
                    np.save(base_output_path + "_cor.npy", correlation)

        if (
            self.folder_name is not None
            and self.folder_name != "in_memory"
            and os.path.isfile(base_output_path + "_spe.npy")
        ):
            spearman = np.load(base_output_path + "_spe.npy")
        else:
            for normalised in task_producer(
                self.mat[:, :, session], 0, random_state=self.random_state
            ):
                spearman = np.corrcoef(
                    np.argsort(np.argsort(normalised, axis=0), axis=0), rowvar=False
                )[np.triu_indices(self.series, 1)]
            if self.save_out:
                if self.folder_name == "in_memory":
                    self.out_data[session]["spearman"] = spearman
                else:
                    np.save(base_output_path + "_spe.npy", spearman)

        return (
            true_and_surrogate_MI,
            true_and_surrogate_MI_shadow,
            correlation,
            spearman,
        )

    def _do_estimate(
        self,
        extended_stats=False,
        compute_shadow: Union[bool, Literal["extend"]] = False,
        **kwargs,
    ):
        extended_stats = extended_stats and self.surrogates > 20
        tmp_statsNames = self.statsNames if extended_stats else self.statsNames[:3]
        if (
            self.folder_name is not None
            and self.folder_name != "in_memory"
            and os.path.isfile(os.path.join(self.folder_name, "global_stats.json"))
        ):
            with open(os.path.join(self.folder_name, "global_stats.json")) as fp:
                self.global_stats = json.load(fp)
        else:
            self.global_stats = {name: [] for name in tmp_statsNames}
            if compute_shadow:
                self.global_stats.update(
                    {name + "shadow": [] for name in tmp_statsNames}
                )

        self.corrector = Corrector(
            self.knn,
            duration=self.duration,
            folder_name=self.folder_name,
            cache_dir=self.cacheDir,
            workers=self.workers,
            display=self.display,
            retrieve=self.retrieve,
            config=self.config,
            verbose=self.verbose,
            **kwargs,
        )
        self.corrector.compute_correction()

        if compute_shadow:
            if compute_shadow == "extend":
                compute_shadow = max(1, int(5e3 // self.duration))
                if self.verbose:
                    print(
                        "Extending the shadow dataset by {} times".format(
                            compute_shadow
                        )
                    )
            else:
                compute_shadow = 1

            if compute_shadow == 1:
                self.shadow_corrector = self.corrector

            else:
                self.shadow_corrector = Corrector(
                    self.knn,
                    duration=self.duration * compute_shadow,
                    folder_name=self.folder_name,
                    cache_dir=self.cacheDir,
                    workers=self.workers,
                    display=self.display,
                    retrieve=self.retrieve,
                    config=self.config,
                    verbose=self.verbose,
                    **kwargs,
                )
                self.shadow_corrector.compute_correction()

        with get_pool(self.workers) as pool:
            for session in tqdm(
                range(self.sessions),
                desc=f"Session",
                leave=True,
                disable=not self.verbose,
            ):
                if session >= self.stop_saving:
                    self.save_out = False
                globalStatsComputedSessions = min(map(len, self.global_stats.values()))
                globalsToBeComputed = globalStatsComputedSessions < session + 1
                assert (
                    max(map(len, self.global_stats.values()))
                    == globalStatsComputedSessions
                ), "Inconsistent global_stats.json"

                plotAlreadyThere = (
                    self.folder_name is not None
                    and self.folder_name != "in_memory"
                    and os.path.isfile(
                        os.path.join(
                            self.folder_name,
                            f"session{self.first_session + session:02}_{self.knn}.pdf",
                        )
                    )
                )
                plottingNeeded = (
                    self.folder_name is not None
                    and self.folder_name != "in_memory"
                    and self.save_out
                    and not plotAlreadyThere
                    and not self.EC
                ) or self.display

                if globalsToBeComputed or plottingNeeded:
                    (
                        true_and_surrogate_MI,
                        true_and_surrogate_MI_shadow,
                        corr,
                        spearman,
                    ) = self._single_session_numeric(session, pool, compute_shadow)

                    if globalsToBeComputed:
                        for key, val in statistics(
                            true_and_surrogate_MI,
                            extended_stats,
                            self.EC,
                            pool,
                            self.corrector,
                        ).items():
                            if key in self.global_stats:
                                self.global_stats[key].append(val)
                        if compute_shadow:
                            for key, val in statistics(
                                true_and_surrogate_MI_shadow,
                                extended_stats,
                                self.EC,
                                pool,
                                self.corrector,
                            ).items():
                                if key in self.global_stats:
                                    self.global_stats[key + "shadow"].append(val)

                    if plottingNeeded:
                        self._smile_plot(
                            session,
                            corr,
                            true_and_surrogate_MI,
                            extended_stats,
                            compute_shadow,
                        )

                if self.folder_name is not None and self.folder_name != "in_memory":
                    with open(
                        os.path.join(
                            self.folder_name,
                            os.path.split(self.folder_name)[1] + "_globalStats.json",
                        ),
                        "w",
                    ) as fp:
                        json.dump(self.global_stats, fp)
        self.save_out = self.base_save_out
        global_to_return = {
            k: np.array(v) if len(v) > 1 else v[0] for k, v in self.global_stats.items()
        }

        if self.folder_name == "in_memory":
            self.out_data["global_stats"] = global_to_return
            return self.out_data
        else:
            return global_to_return

    def _smile_plot(
        self,
        session: int,
        correlation: np.ndarray,
        true_and_surrogate_MI: np.ndarray,
        extended_stats: bool,
        compute_shadow: bool,
    ):
        if self.surrogates > 1:
            corrected_percentile01pointer = (
                (self.surrogates - 0.5) * (0.01) / (self.surrogates - 1)
            )
            corrected_percentile99pointer = (
                (self.surrogates - 0.5) * (0.99) / (self.surrogates - 1)
            )
        corrected_statsMI = self.corrector.correct(true_and_surrogate_MI)

        plt.scatter(correlation, corrected_statsMI[:, 0])
        new_order = np.argsort(correlation)
        expected = -0.5 * np.log(1 - correlation**2)
        plt.plot(correlation[new_order], expected[new_order], "purple")
        if self.surrogates > 1:
            pair_gauss_mi = np.mean(corrected_statsMI[:, 1:], 1)
            percentile01_PLOT, percentile99_PLOT = np.quantile(
                corrected_statsMI[:, 1:],
                [corrected_percentile01pointer, corrected_percentile99pointer],
                1,
            )
            plt.plot(correlation[new_order], pair_gauss_mi[new_order], "r")
            plt.plot(correlation[new_order], percentile01_PLOT[new_order], "lightblue")
            plt.plot(correlation[new_order], percentile99_PLOT[new_order], "g")
        plt.xlabel("correlation")
        plt.ylabel("mutual information (nats)")
        title = f"Session {session} $-$ $MI_T:${self.global_stats['totalMI'][session]:.3} vs $MI_G:${self.global_stats['gaussMI'][session]:.3}"
        if extended_stats:
            title += f"({self.global_stats['ratio95'][session]:.3}>95%"
            if compute_shadow:
                title += (
                    f"$-$ {self.global_stats['ratio95shadow'][session]:.3}>95% shadow"
                )
            title += ")"
        plt.title(title)
        plt.ylim(bottom=0)
        if self.folder_name is not None and not os.path.isfile(
            f"{self.folder_name}/session{self.first_session + session:02}_{self.knn}.pdf"
        ):
            plt.savefig(
                f"{self.folder_name}/session{self.first_session + session:02}_{self.knn}.pdf",
                bbox_inches="tight",
            )
        if self.display:
            plt.show()
        else:
            plt.close()
