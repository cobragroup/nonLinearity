# This package contain some useful code

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
import configparser
import json
import scipy.io as sio
from warnings import warn
import sys
import os
import socket
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path)
from corrector import Corrector
from support import quantile_vector, total_mutual_information, surrogate, task_producer, statistics


class NonLinearEstimator:
    statsNames = ["globalratio95control", "globalratio99control", "globalratio05", "globalratio95", "globalratio99", "globaltotalMI",
                  "globalgaussMI", "globalratio05shadow", "globalratio95shadow", "globalratio99shadow", "globaltotalMIshadow", "globalgaussMIshadow"]

    def __init__(self, configFile=None, dataset=None, nbins=None, regions="", savenpy=False):
        config = configparser.ConfigParser()
        self.savenpy = savenpy
        configfile = configFile if configFile is not None else os.path.join(
            path, "config.ini")
        assert os.path.isfile(configfile)
        config.read(configfile)

        config['DEFAULT']['regions'] = regions
        self.steps = config.getint("correction", "steps", fallback=200)
        self.iters = config.getint("correction", "iters", fallback=1000)
        self.nsamples = config.getint("correction", "nsamples", fallback=0)

        if nbins is None:
            self.nbins = config.getint("global", "nbins", fallback=8)
        else:
            self.nbins = nbins
        self.display = config.getboolean("global", "display", fallback=True)
        self.workers = config.getint("global", "workers", fallback=4)
        self.ouputFolder = config.get("global", "outputFolder", fallback="..")

        thisHost = socket.gethostname()
        if config.has_section(thisHost):
            self.workers = config.getint(thisHost, "workers", fallback=self.workers)
            self.ouputFolder = config.get(thisHost, "outputFolder", fallback=self.ouputFolder)

        if not os.path.isabs(self.ouputFolder):
            self.ouputFolder = os.path.join(path, self.ouputFolder)

        self.Nsurrogates = config.getint(
            "estimate", "Nsurrogates", fallback=99)
        if dataset is None:
            dataset = config.get("global", "dataset", fallback=None)
        assert dataset is not None, "Unspecified dataset in .ini file."
        assert config.has_section(
            dataset), "The details for the specified dataset are missing in .ini file."
        filePath = config.get(dataset, "filePath", fallback=None)
        assert filePath is not None, "Missing dataset file path in .ini file."
        fileName = config.get(dataset, "fileName", fallback=None)
        self.fieldName = config.get(dataset, "fieldName", fallback=None)
        assert fileName is not None, "Missing dataset filename in .ini file."
        if self.fieldName is None:
            warn("Missing dataset fieldname in .ini file. Trying with euristics.")
        hc_start = config.getint(
            dataset, "healthy_control_start", fallback=None)
        hc_start = hc_start if hc_start else None
        hc_end = config.getint(dataset, "healthy_control_end", fallback=None)
        hc_end = hc_end if hc_end else None
        self.hc_slice = slice(hc_start, hc_end)

        self.fileName = os.path.join(filePath, fileName)
        assert os.path.isfile(
            self.fileName), f"Missing dataset at specified path: {self.fileName}."
        folderName = os.path.splitext(fileName)[0] + f"_bin{self.nbins}"
        self.folderName = os.path.join(self.ouputFolder, folderName)
        if not os.path.isdir(self.folderName):
            os.mkdir(self.folderName)
        print(
            f"Using: {os.path.abspath(self.fileName)} and {os.path.abspath(self.folderName)}"
        )

    def load_data(self):
        tmp_mat = sio.loadmat(self.fileName)
        if self.fieldName is None:
            self.fieldName = [k for k in tmp_mat.keys() if k not in [
                '__header__', '__version__', '__globals__']][0]

        self.mat = tmp_mat[self.fieldName][:, :, self.hc_slice]
        duration, self.regions, self.sessions = self.mat.shape
        with open(os.path.join(self.folderName, "shape.json"), "w") as fp:
            json.dump(self.mat.shape, fp)

        print(
            "Loaded data matrix: {} samples by {} regions by {} sessions".format(
                duration, self.regions, self.sessions
            )
        )
        if self.nsamples == 0:
            self.nsamples = duration
        if duration != self.nsamples:
            warn(
                f"Acquisition duration ({duration}) is different from the set number of samples for correction ({self.nsamples})."
            )
        self.pairNum = int((self.regions * (self.regions - 1)) / 2)

    def run(self):
        self.load_data()

        self.corrector = Corrector(
            self.steps,
            self.folderName,
            self.iters,
            self.nsamples,
            self.nbins,
            self.workers,
            display=self.display,
        )
        self.corrector.compute_correction()

        self.estimate()

    def _single_patient_numeric(self, patientN, pool: mp.Pool):
        if not os.path.isfile(
            f"{self.folderName}/patient{patientN:02}_{self.nbins}.npy"
        ):
            statsMI = np.zeros([self.pairNum, self.Nsurrogates + 1])
            for ns, tmi in enumerate(pool.imap(total_mutual_information, ((patient, self.nbins) for patient in task_producer(self.mat[:, :, patientN], self.Nsurrogates)))):#tqdm(, disable=True, total=self.Nsurrogates + 1, desc=f"Patient {patientN} true", leave=False):
                statsMI[:, ns] = tmi
            if self.savenpy:
                np.save(
                    f"{self.folderName}/patient{patientN:02}_{self.nbins}", statsMI)
        else:
            statsMI = np.load(
                f"{self.folderName}/patient{patientN:02}_{self.nbins}.npy"
            )

        if not os.path.isfile(
            f"{self.folderName}/patient{patientN:02}_{self.nbins}_sha.npy"
        ):
            shadow = surrogate(self.mat[:, :, patientN])
            statsMI_shadow = np.zeros([self.pairNum, self.Nsurrogates + 1])
            for ns, tmi in enumerate(pool.imap(total_mutual_information, ((patient, self.nbins) for patient in task_producer(shadow[:, :], self.Nsurrogates)))):#tqdm(, disable=True, total=self.Nsurrogates + 1, desc=f"Patient {patientN} shadow", leave=False):
                statsMI_shadow[:, ns] = tmi

            corr = np.array(
                [
                    np.corrcoef(
                        self.mat[:, zone1, patientN], self.mat[:,
                                                               zone2, patientN]
                    )[1, 0]
                    for zone1 in range(self.regions)
                    for zone2 in range(zone1 + 1, self.regions)
                ]
            )

            if self.savenpy:
                np.save(
                    f"{self.folderName}/patient{patientN:02}_{self.nbins}_sha",
                    statsMI_shadow,
                )
                np.save(
                    f"{self.folderName}/patient{patientN:02}_{self.nbins}_cor", corr
                )
        else:
            statsMI_shadow = np.load(
                f"{self.folderName}/patient{patientN:02}_{self.nbins}_sha.npy"
            )
            corr = np.load(
                f"{self.folderName}/patient{patientN:02}_{self.nbins}_cor.npy"
            )

        return statsMI, statsMI_shadow, corr

    def estimate(self):
        if os.path.isfile(os.path.join(self.folderName, "globalStats.json")):
            with open(os.path.join(self.folderName, "globalStats.json")) as fp:
                self.globalStats = json.load(fp)
        else:
            self.globalStats = {name: [] for name in self.statsNames}

        with mp.Pool(self.workers) as pool:
            for patientN in tqdm(range(self.sessions), desc=f"Patient", leave=True):

                globalStatsComputedSubjects = min(
                    map(len, self.globalStats.values()))
                globalsToBeComputed = globalStatsComputedSubjects < patientN+1

                plotAlreadyThere = os.path.isfile(
                    f"{self.folderName}/patient{patientN:02}_{self.nbins}.pdf")
                plottingNeeded = (not plotAlreadyThere) or self.display

                if globalsToBeComputed or plottingNeeded:
                    statsMI, statsMI_shadow, corr = self._single_patient_numeric(
                        patientN, pool)

                    if globalsToBeComputed:
                        assert max(map(len, self.globalStats.values(
                        ))) == globalStatsComputedSubjects, "Inconsistent globalStats.json"
                        self._statistics(statsMI, statsMI_shadow)

                    if plottingNeeded:
                        self._smile_plot(patientN, corr, statsMI)

                with open(os.path.join(self.folderName, "globalStats.json"), "w") as fp:
                    json.dump(self.globalStats, fp)

    def _statistics(self, statsMI, statsMI_shadow):
        statTrue = statistics(statsMI, self.corrector.newco, self.corrector.trueval)
        statShadow = statistics(statsMI_shadow, self.corrector.newco, self.corrector.trueval)
        for key in self.statsNames:
            if "shadow" in key:
                self.globalStats[key].append(statShadow[key[:-len("shadow")]])
            else:
                self.globalStats[key].append(statTrue[key])

    def _smile_plot(self, patientN, corr, statsMI):
        corr_statsMI = self.corrector.correctI(statsMI)
        mean_cont_mi_multisurr = np.mean(corr_statsMI[:, 1:], 1)
        # std_cont_mi_multisurr=np.std(corr_statsMI[:,1:],1)
        perc99_PLOT = quantile_vector(corr_statsMI, 0.99)
        perc01_PLOT = quantile_vector(corr_statsMI, 0.01)

        plt.scatter(corr, corr_statsMI[:, 0])
        neworder = np.argsort(corr)
        expected = -0.5 * np.log(1 - corr**2)
        plt.plot(corr[neworder], expected[neworder], "purple")
        plt.plot(corr[neworder], mean_cont_mi_multisurr[neworder], "r")
        plt.plot(corr[neworder], perc01_PLOT[neworder], "lightblue")
        plt.plot(corr[neworder], perc99_PLOT[neworder], "g")
        plt.xlabel("correlation")
        plt.ylabel("mutual information (nats)")
        title = f"Patient {patientN} - {self.globalStats['globaltotalMI'][patientN]:.3}/{self.globalStats['globalgaussMI'][patientN]:.3} (^{self.globalStats['globalratio95'][patientN]:.3}-{self.globalStats['globalratio95'][patientN]:.3}_{self.globalStats['globalratio95shadow'][patientN]:.3})"
        plt.title(title)
        plt.ylim(bottom=0)
        if not os.path.isfile(
            f"{self.folderName}/patient{patientN:02}_{self.nbins}.pdf"
        ):
            plt.savefig(
                f"{self.folderName}/patient{patientN:02}_{self.nbins}.pdf", bbox_inches="tight"
            )
        if self.display:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    estimator = NonLinearEstimator(dataset="benchmark")
    estimator.run()
