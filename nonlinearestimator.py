# %%
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
path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path)
from corrector import Corrector
from support import pair_mutual_information, surrogate, task_producer


# %%
class NonLinearEstimator:
    def __init__(self, configFile=None, dataset=None, nbins=None, regions = ""):
        config = configparser.ConfigParser()
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

        self.Nsurrogates = config.getint(
            "estimate", "Nsurrogates", fallback=99)
        if dataset is None:
            dataset = config.get("global", "dataset", fallback=None)
        assert dataset is not None, "Unspecified dataset in .ini file."
        assert config.has_section(
            dataset), "The details for the specified dataset are missing in .ini file."
        filePath = config.get(dataset, "filePath", fallback="./")
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
        self.folderName = os.path.join(filePath, folderName)
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

    def estimate(self):
        pairNum = int((self.regions * (self.regions - 1)) / 2)
        statsNames = ["globalratio95control", "globalratio99control", "globalratio05", "globalratio95", "globalratio99", "globaltotalMI",
                      "globalgaussMI", "globalratio05shadow", "globalratio95shadow", "globalratio99shadow", "globaltotalMIshadow", "globalgaussMIshadow"]
        globalStats = {name: np.zeros(self.sessions) for name in statsNames}

        pool = mp.Pool(self.workers)
        for patientN in tqdm(range(self.sessions), desc=f"Patient:", leave=True):
            if not os.path.isfile(
                f"{self.folderName}/patient{patientN:02}_{self.nbins}.npy"
            ):
                statsMI = np.zeros([pairNum, self.Nsurrogates + 1])
                for ns, patient in tqdm(
                    enumerate(
                        task_producer(
                            self.mat[:, :, patientN], self.Nsurrogates)
                    ),
                    total=self.Nsurrogates + 1,
                    desc=f"Patient {patientN} true", leave=False
                ):
                    statsMI[:, ns] = pool.starmap(
                        pair_mutual_information,
                        (
                            (patient[:, zone1], patient[:, zone2], self.nbins)
                            for zone1 in range(self.regions)
                            for zone2 in range(zone1 + 1, self.regions)
                        ),
                    )
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
                statsMI_shadow = np.zeros([pairNum, self.Nsurrogates + 1])
                for ns, patient in tqdm(
                    enumerate(task_producer(shadow[:, :], self.Nsurrogates)),
                    total=self.Nsurrogates + 1,
                    desc=f"Patient {patientN} shadow", leave=False
                ):
                    statsMI_shadow[:, ns] = pool.starmap(
                        pair_mutual_information,
                        (
                            (patient[:, zone1], patient[:, zone2], self.nbins)
                            for zone1 in range(self.regions)
                            for zone2 in range(zone1 + 1, self.regions)
                        ),
                    )

                # statsMI_univar = np.zeros([pairNum, self.Nsurrogates])
                # for ns, patient in tqdm(enumerate(task_producer(self.mat[:, :, patientN], self.Nsurrogates, False)), total=self.Nsurrogates+1, desc=f"Patient {patientN} univar", leave=False):
                #     statsMI_univar[:, ns] = pool.starmap(pair_mutual_information, ((
                #         patient[:, zone1], patient[:, zone2]) for zone1 in range(self.regions) for zone2 in range(zone1+1, self.regions)))

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

                np.save(
                    f"{self.folderName}/patient{patientN:02}_{self.nbins}_sha",
                    statsMI_shadow,
                )
                # np.save(f"{self.folderName}/patient{patientN:02}_{self.nbins}_uni", statsMI_univar)
                np.save(
                    f"{self.folderName}/patient{patientN:02}_{self.nbins}_cor", corr
                )
            else:
                statsMI_shadow = np.load(
                    f"{self.folderName}/patient{patientN:02}_{self.nbins}_sha.npy"
                )
                # statsMI_univar = np.load(
                #     f"{self.folderName}/patient{patient:02}_{self.nbins}_uni.npy")
                corr = np.load(
                    f"{self.folderName}/patient{patientN:02}_{self.nbins}_cor.npy"
                )

            correctedperc95pointer = (self.Nsurrogates * (0.95) - 0.5) / (
                self.Nsurrogates - 1
            )
            correctedperc99pointer = (self.Nsurrogates * (0.99) - 0.5) / (
                self.Nsurrogates - 1
            )
            correctedperc05pointer = (self.Nsurrogates * (0.05) - 0.5) / (
                self.Nsurrogates - 1
            )
            correctedperc01pointer = (self.Nsurrogates * (0.01) - 0.5) / (
                self.Nsurrogates - 1
            )

            perc95 = np.quantile(statsMI[:, 1:], correctedperc95pointer, 1)
            nonlin95 = statsMI[:, 0] > perc95
            ratio95 = np.mean(nonlin95)
            perc99 = np.quantile(statsMI[:, 1:], correctedperc99pointer, 1)
            nonlin99 = statsMI[:, 0] > perc99
            ratio99 = np.mean(nonlin99)
            perc05 = np.quantile(statsMI[:, 1:], correctedperc05pointer, 1)
            nonlin05 = statsMI[:, 0] < perc05
            ratio05 = np.mean(nonlin05)

            # pvalue_MI = 1 - np.sum(statsMI_univar < statsMI[:, 0], 1)/(self.Nsurrogates+1)
            pvalue_NMI = 1 - np.sum(statsMI[:, 1:].T < statsMI[:, 0], 0) / (
                self.Nsurrogates + 1
            )
            ratio95control = np.mean(pvalue_NMI < 0.0500001)
            ratio99control = np.mean(pvalue_NMI < 0.0100001)
            globalStats["globalratio95control"][patientN] = ratio95control
            globalStats["globalratio99control"][patientN] = ratio99control

            perc95_shadow = np.quantile(
                statsMI_shadow[:, 1:], correctedperc95pointer, 1
            )
            nonlin95_shadow = statsMI_shadow[:, 0] > perc95_shadow
            ratio95_shadow = np.mean(nonlin95_shadow)
            perc99_shadow = np.quantile(
                statsMI_shadow[:, 1:], correctedperc99pointer, 1
            )
            nonlin99_shadow = statsMI_shadow[:, 0] > perc99_shadow
            ratio99_shadow = np.mean(nonlin99_shadow)
            perc05_shadow = np.quantile(
                statsMI_shadow[:, 1:], correctedperc05pointer, 1
            )
            nonlin05_shadow = statsMI_shadow[:, 0] < perc05_shadow
            ratio05_shadow = np.mean(nonlin05_shadow)

            corr_statsMI = self.corrector.correctI(statsMI)

            mean_cont_mi_multisurr = np.mean(corr_statsMI[:, 1:], 1)
            # std_cont_mi_multisurr=np.std(corr_statsMI[:,1:],1)
            perc99_PLOT = np.quantile(
                corr_statsMI[:, 1:], correctedperc99pointer, 1)
            perc01_PLOT = np.quantile(
                corr_statsMI[:, 1:], correctedperc01pointer, 1)

            allpairs_cont_mi_data = np.mean(corr_statsMI[:, 1])
            allpairs_mean_cont_mi_multisurr = np.mean(corr_statsMI[:, 1:])

            # corr_statsMI_univar = self.corrector.correctI(statsMI_univar)
            # mean_cont_mi_unisurr=np.mean(corr_statsMI_univar,1)
            # std_cont_mi_unisurr=np.std(corr_statsMI_univar,1)
            # allpairs_mean_cont_mi_unisurr = np.mean(corr_statsMI_univar)

            globalStats["globalratio05"][patientN] = ratio05
            globalStats["globalratio95"][patientN] = ratio95
            globalStats["globalratio99"][patientN] = ratio99
            globalStats["globaltotalMI"][patientN] = allpairs_cont_mi_data
            globalStats["globalgaussMI"][patientN] = allpairs_mean_cont_mi_multisurr

            corr_statsMI_shadow = self.corrector.correctI(statsMI_shadow)

            # mean_cont_mi_multisurrshadow=np.mean(corr_statsMI_shadow[:,1:],1)
            # std_cont_mi_multisurrshadow=np.std(corr_statsMI_shadow[:,1:],1)

            allpairs_cont_mi_datashadow = np.mean(corr_statsMI_shadow[:, 1])
            allpairs_mean_cont_mi_multisurrshadow = np.mean(
                corr_statsMI_shadow[:, 1:])

            globalStats["globalratio05shadow"][patientN] = ratio05_shadow
            globalStats["globalratio95shadow"][patientN] = ratio95_shadow
            globalStats["globalratio99shadow"][patientN] = ratio99_shadow
            globalStats["globaltotalMIshadow"][patientN] = allpairs_cont_mi_datashadow
            globalStats["globalgaussMIshadow"][patientN] = allpairs_mean_cont_mi_multisurrshadow

            if (
                not os.path.isfile(
                    f"{self.folderName}/patient{patientN:02}_{self.nbins}.pdf"
                )
                or self.display
            ):
                plt.scatter(corr, corr_statsMI[:, 1])
                neworder = np.argsort(corr)
                expected = -0.5 * np.log(1 - corr**2)
                plt.plot(corr[neworder], expected[neworder], "purple")
                plt.plot(corr[neworder], mean_cont_mi_multisurr[neworder], "r")
                plt.plot(corr[neworder], perc01_PLOT[neworder], "lightblue")
                plt.plot(corr[neworder], perc99_PLOT[neworder], "g")
                plt.xlabel("correlation")
                plt.ylabel("mutual information (nats)")
                plt.title(
                    f"Patient {patientN} - {allpairs_cont_mi_data:.3}/{allpairs_mean_cont_mi_multisurr:.3} (^{ratio95:.3}-{ratio95control:.3}_{ratio95_shadow:.3})"
                )
                plt.ylim(bottom=0)
                if not os.path.isfile(
                    f"{self.folderName}/patient{patientN:02}_{self.nbins}.pdf"
                ):
                    plt.savefig(
                        f"{self.folderName}/patient{patientN:02}_{self.nbins}.pdf"
                    )
                if self.display:
                    plt.show()
                else:
                    plt.close()
        pool.close()
        globalStats = {k: v.tolist() for k, v in globalStats.items()}

        with open(f"{self.folderName}/globalStats.json", "w") as fp:
            json.dump(globalStats, fp)
