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
import warnings
from innovationOrthogonalization import innOr

class NonLinearEstimator:
    statsNames = ["globaltotalMI", "globalgaussMI", "globalsigmaGaussMI", "globalratio95control", "globalratio99control", "globalratio05", "globalratio95", "globalratio99"]

    def __init__(self, configFile=None, nbins=None, Nsurrogates=None, cache=None, savenpy=False, suffix="", retrieve=True, jitter=False, ortho=False, dataset=None, regions="", truncateInput=None, workers=None):
        self.savenpy = savenpy
        self.suffix = suffix
        self.retrieve = retrieve
        self.dataset = dataset
        self.regions = regions
        self.truncateInput = truncateInput

        self.__read_config(configFile)

        if ortho is not None:
            self.ortho = ortho
        if cache is not None:
            self.cacheDir = cache
            if self.cacheDir is not None and not os.path.isdir(self.cacheDir) and os.path.isdir(os.path.join(path, self.cacheDir)):
                self.cacheDir = os.path.join(path, self.cacheDir)
        if nbins is not None:
            self.nbins = nbins
        if Nsurrogates is not None:
            self.Nsurrogates = Nsurrogates
        if workers is not None:
            self.workers = workers
        if jitter is not None:
            self.jitter = self.__adjust_jitter (jitter)

        assert self.nbins is not None, "Number of bins undefined, can't create the NonLinearEstimator"
        assert self.Nsurrogates is not None, "Number of surrogates undefined, can't create the NonLinearEstimator"
        if self.workers is None:
            self.workers=1
        / aggiungi la finta pool

    def __read_config (self, configFile):
        configfile = configFile if configFile is not None else os.path.join(
            path, "config.ini")
        try:
            self.config = configparser.ConfigParser()
            self.config.read(configfile)
        
            self.ortho = self.config.getboolean("global", "orthogonalise", fallback=False)
            self.jitter = self.config.get("global", "jitter", fallback="0.")
            self.display = self.config.getboolean("global", "display", fallback=True)
            self.workers = self.config.getint("global", "workers", fallback=4)
            self.ouputFolder = self.config.get("global", "outputFolder", fallback="..")
            self.cacheDir = self.config.getint("correction", "cacheDir", fallback=None)
            self.nbins = self.config.getint("global", "nbins", fallback=8)
            self.Nsurrogates = self.config.getint("global", "Nsurrogates", fallback=99)

            thisHost = socket.gethostname()
            if self.config.has_section(thisHost):
                self.workers = self.config.getint(
                    thisHost, "workers", fallback=self.workers)
                self.ouputFolder = self.config.get(
                    thisHost, "outputFolder", fallback=self.ouputFolder)
                self.cacheDir = self.config.get(thisHost, "cacheDir", fallback=self.cacheDir)

            if self.cacheDir is not None and not os.path.isdir(self.cacheDir) and os.path.isdir(os.path.join(path, self.cacheDir)):
                self.cacheDir = os.path.join(path, self.cacheDir)
            if not os.path.isabs(self.ouputFolder):
                self.ouputFolder = os.path.abspath(os.path.join(path, self.ouputFolder))
        except Exception as e:
            warnings.warn("Unable to read config file. "+e.args[0])
            self.config = None
            self.ortho = None
            self.jitter = None
            self.display = None
            self.workers = None
            self.ouputFolder = None
            self.cacheDir = None
            self.nbins = None

    def __read_config_dataset (self, regions=None, truncateInput=None, dataset=None, **kwargs):
        if dataset is not None:
            self.dataset = dataset
        if regions is not None:
            self.regions = regions
        if truncateInput is not None:
            self.truncateInput = truncateInput
        assert self.config is not None, "When not passing data directly, a valid config.ini file is necessary."
        self.config['DEFAULT']['regions'] = self.regions

        if dataset is None:
            dataset = self.config.get("global", "dataset", fallback=None)
        assert dataset is not None, "Unspecified dataset in .ini file."
        assert self.config.has_section(
            dataset), "The details for the specified dataset are missing in .ini file."
        
        filePath = self.config.get(dataset, "filePath", fallback=None)
        assert filePath is not None, "Missing dataset file path in .ini file."
        
        fileName = self.config.get(dataset, "fileName", fallback=None)
        assert fileName is not None, "Missing dataset filename in .ini file."
        
        self.fieldName = self.config.get(dataset, "fieldName", fallback=None)
        if self.fieldName is None:
            warn("Missing dataset fieldname in .ini file. Trying with euristics.")
        
        hc_start = self.config.getint(
            dataset, "healthy_control_start", fallback=None)
        hc_start = hc_start if hc_start else None
        hc_end = self.config.getint(dataset, "healthy_control_end", fallback=None)
        hc_end = hc_end if hc_end else None
        self.hc_slice = slice(hc_start, hc_end)

        self.fileName = os.path.join(filePath, fileName)
        assert os.path.isfile(
            self.fileName), f"Missing dataset at specified path: {self.fileName}."
        print(f"Using: {os.path.abspath(self.fileName)}")

    def __adjust_jitter (self, jitter):
        if isinstance(jitter, str):
            try:
                jitter = float(jitter)
            except:
                jitter = bool(jitter)
        if isinstance(jitter, bool):
            if jitter:
                jitter = 1e-3
        else:
            if np.isclose(jitter, 0):
                jitter = False
        return jitter 

    def load_data(self, data=None, jitter=None, ortho=None, **kwargs):
        if ortho is not None:
            self.ortho = ortho
        if jitter is not None:
            self.jitter = self.__adjust_jitter (jitter)
        if data is None:
            self.__read_config_dataset(**kwargs)
            if self.fieldName is None:
                tmp_mat = sio.loadmat(self.fileName)
                self.fieldName = [k for k in tmp_mat.keys() if k not in [
                    '__header__', '__version__', '__globals__']][0]
                tmp_mat = tmp_mat[self.fieldName]
            else:
                tmp_mat = sio.loadmat(self.fileName)[self.fieldName]
            truncate_slice = slice(None, self.truncateInput)
            tmp_mat = tmp_mat[truncate_slice, :, self.hc_slice]
        else:
            self.fileName = None
            tmp_mat = data
            
        if self.ortho:
            try:
                self.mat = innOr(data, **kwargs)
            except np.linalg.LinAlgError as e:
                raise RuntimeError(*e.args)    
        if self.jitter:
            spa = np.sort(np.diff(np.sort(self.mat[:,0,0])))[0]
            self.mat += np.random.normal(0, spa*1e-3, self.mat.shape)
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

    def __output_folder (self, suffix, **kwargs):
        if suffix is not None:
            self.suffix = suffix

        if self.fileName is not None:
            nameParts = [os.path.splitext(os.path.split(self.fileName)[1])[0]]
        else:
            if isinstance(self.savenpy, bool):
                nameParts = ["".join(map(chr, np.random.randint(65, 91, 7)))]
            else:
                nameParts = [str(self.savenpy)]
        if self.suffix:
            nameParts.append(str(self.suffix))
        nameParts.append(f"bin{self.nbins}")
        folderName =  "_".join(nameParts)
        
        if not os.path.isabs(folderName) and self.ouputFolder is not None:
            self.folderName = os.path.abspath(os.path.join(self.ouputFolder, folderName))
        
        if not os.path.isdir(self.folderName):
            os.makedirs(self.folderName)
        print(f"Output saved in: {self.folderName}")
        self.savenpy = True

    def estimate(self, data=None, savenpy=None, retrieve=None, display=None, **kwargs):
        self.load_data(data=data, **kwargs)

        if savenpy is not None:
            self.savenpy = savenpy
        if retrieve is not None:
            self.retrieve = retrieve
        if display is not None:
            self.display = display

        if  self.savenpy:
            self.__output_folder(**kwargs)
        else:
            self.folderName = None

        self.corrector = Corrector(
            self.nbins,
            folderName=self.folderName,
            cacheDir=self.cacheDir,
            workers=self.workers,
            display=self.display,
            retrieve=self.retrieve,
            config=self.config,
            **kwargs
        )
        self.corrector.compute_correction()

        return self.do_estimate(**kwargs)

    def _single_patient_numeric(self, patientN, pool: mp.Pool, compute_shadow):
        base_output_name = f"patient{patientN:02}_{self.nbins}"
        base_output_path = os.path.join(self.folderName, base_output_name)
        if self.folderName is not None and not os.path.isfile(base_output_path + ".npy"):
            statsMI = np.zeros([self.pairNum, self.Nsurrogates + 1])
            # tqdm(, disable=True, total=self.Nsurrogates + 1, desc=f"Patient {patientN} true", leave=False):
            for ns, tmi in enumerate(pool.imap(total_mutual_information, ((patient, self.nbins) for patient in task_producer(self.mat[:, :, patientN], self.Nsurrogates)))):
                statsMI[:, ns] = tmi
            if self.savenpy:
                np.save(base_output_path + ".npy", statsMI)
        else:
            statsMI = np.load(base_output_path + ".npy")

        if compute_shadow:
            if self.folderName is not None and not os.path.isfile(base_output_path + "_sha.npy"):
                shadow = surrogate(self.mat[:, :, patientN])
                statsMI_shadow = np.zeros([self.pairNum, self.Nsurrogates + 1])
                # tqdm(, disable=True, total=self.Nsurrogates + 1, desc=f"Patient {patientN} shadow", leave=False):
                for ns, tmi in enumerate(pool.imap(total_mutual_information, ((patient, self.nbins) for patient in task_producer(shadow[:, :], self.Nsurrogates)))):
                    statsMI_shadow[:, ns] = tmi

                if self.savenpy:
                    np.save(base_output_path + "_sha.npy", statsMI_shadow)
            else:
                statsMI_shadow = np.load(base_output_path + "_sha.npy")
        else:
            statsMI_shadow = None

        if self.folderName is not None and not os.path.isfile(base_output_path + "_cor.npy"):
            for norm in task_producer(self.mat[:, :, patientN], 0):
                corr = np.corrcoef(norm, rowvar=False)[np.triu_indices(self.regions,1)]
            if self.savenpy:
                np.save(base_output_path + "_cor.npy", corr)
        else:
            corr = np.load(base_output_path + "_cor.npy")

        return statsMI, statsMI_shadow, corr

    def do_estimate(self, extended_stats=False, compute_shadow=False, **kwargs):
        tmp_statsNames = self.statsNames if extended_stats else self.statsNames[:3]
        if self.folderName is not None and os.path.isfile(os.path.join(self.folderName, "globalStats.json")):
            with open(os.path.join(self.folderName, "globalStats.json")) as fp:
                self.globalStats = json.load(fp)
        else:
            self.globalStats = {name: [] for name in tmp_statsNames}
            if compute_shadow:
                self.globalStats.update({name+"shadow": [] for name in tmp_statsNames})

        with mp.Pool(self.workers) as pool:
            for patientN in tqdm(range(self.sessions), desc=f"Patient", leave=True):

                globalStatsComputedSubjects = min(
                    map(len, self.globalStats.values()))
                globalsToBeComputed = globalStatsComputedSubjects < patientN+1
                assert max(map(len, self.globalStats.values())) == globalStatsComputedSubjects, "Inconsistent globalStats.json"

                plotAlreadyThere = os.path.isfile(
                    f"{self.folderName}/patient{patientN:02}_{self.nbins}.pdf")
                plottingNeeded = (self.folderName is not None and not plotAlreadyThere) or self.display

                if globalsToBeComputed or plottingNeeded:
                    statsMI, statsMI_shadow, corr = self._single_patient_numeric(
                        patientN, pool, compute_shadow)

                    if globalsToBeComputed:
                        for key, val in statistics(statsMI, self.corrector.newco, self.corrector.trueval, self.workers, extended_stats):
                            self.globalStats[key].append(val)
                        if compute_shadow:
                            for key, val in statistics(statsMI_shadow, self.corrector.newco, self.corrector.trueval, self.workers, extended_stats):
                                self.globalStats[key+"shadow"].append(val)

                    if plottingNeeded:
                        self._smile_plot(patientN, corr, statsMI)

                if self.folderName is not None:
                    with open(os.path.join(self.folderName, os.path.split(self.folderName)[1]+"_globalStats.json"), "w") as fp:
                        json.dump(self.globalStats, fp)
        return {k: np.array(v) for k,v in self.globalStats.items()}

    def _smile_plot(self, patientN, corr, statsMI):
        correctedperc01pointer = (self.Nsurrogates * (0.01) - 0.5) / (
            self.Nsurrogates - 1
        )
        correctedperc99pointer = (self.Nsurrogates * (0.99) - 0.5) / (
            self.Nsurrogates - 1
        )
        corr_statsMI = self.corrector.correctI(statsMI)
        mean_cont_mi_multisurr = np.mean(corr_statsMI[:, 1:], 1)
        perc01_PLOT, perc99_PLOT = np.quantile(
            corr_statsMI[:, 1:], [correctedperc01pointer, correctedperc99pointer], 1)

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
        if self.folderName is not None and not os.path.isfile(
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
