from support import single_iter, correct_vector
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import glob
import json
import shutil
import configparser

class Corrector:
    def __init__(
        self,
        nbins,
        steps = None,
        iters = None,
        nsamples = None,
        folderName = None,
        cacheDir = None,
        workers=1,
        smoothed=False,
        display=False,
        retrieve=True,
        config = None,
        **kwargs
    ):
        self.nbins = nbins
             
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
            self.iters = config.getint("correction", "iters", fallback=10000)
            self.nsamples = config.getint("correction", "nsamples", fallback=0)
        
        if steps is not None:
            self.steps = steps
        if iters is not None:
            self.iters = iters
        if nsamples is not None:
            self.nsamples = nsamples

        self.folderName = folderName
        self.cacheDir = cacheDir

        self.smoothed = smoothed
        self.display = display
        self.workers = workers

        self.newco = None
        self.trueval = None
        self.old_correctI = np.vectorize(self._correctI)
        self.correctI = lambda x: correct_vector(x, self.newco, self.trueval)
        if retrieve:
            self.__retrieve()
    
    def __retrieve (self):
        if os.path.isfile(os.path.join(self.folderName,f"newco_{self.nbins}.npy")) or os.path.isfile(os.path.join(self.cacheDir,f"newco_{self.nbins}_{self.nsamples}.npy")):
            return
        
        for fold in glob.glob(os.path.abspath(os.path.join(self.folderName, os.pardir, f"*bin{self.nbins}"))):
            if os.path.isfile(os.path.join(fold,"shape.json")):
                with open(os.path.join(fold,"shape.json")) as fp:
                    shape = json.load(fp)
                    if shape[0] == self.nsamples:
                        if os.path.isfile(os.path.join(fold,f"newco_{self.nbins}.npy")):
                            print("Retrieving correction from: ", fold)
                            shutil.copy2(os.path.join(fold,f"newco_{self.nbins}.npy"), self.folderName)
                            if os.path.isfile(os.path.join(fold,f"trueval_{self.nbins}.npy")):
                                shutil.copy2(os.path.join(fold,f"trueval_{self.nbins}.npy"), self.folderName)
                            break

    def compute_correction(self):
        """Computes the correction lookup table or loads the cached values."""
        / cache e salvataggio   
        incre = 1 / self.steps
        correction = np.zeros(self.steps)
        self.trueval = np.zeros(self.steps)
        if not os.path.isfile(f"{self.folderName}/newco_{self.nbins}.npy"):
            pool = mp.Pool(self.workers)
            for i in tqdm(range(self.steps), "Computing correction"):
                means = 0, 0
                corre = [[1, i * incre], [i * incre, 1]]
                I = pool.map(
                    single_iter,
                    (
                        (means, corre, self.nsamples, self.nbins)
                        for __ in range(self.iters)
                    ),
                )
                correction[i] = np.average(I)
                self.trueval[i] = -0.5 * np.log(1 - (i * incre) ** 2)

            if self.smoothed:
                tosmo = np.array(
                    [
                        correction[0],
                    ]
                    * 2
                    + correction.tolist()
                    + [
                        correction[-1],
                    ]
                    * 2
                )
                self.newco = np.zeros_like(correction)
                for i in range(len(correction)):
                    self.newco[i] = np.mean(tosmo[i : i + 5])
                if self.display:
                    try:
                        plt.plot(correction[:50])
                        plt.plot(self.newco[:50])
                        plt.show()
                    except:
                        pass
            else:
                self.newco = correction

            np.save(f"{self.folderName}/newco_{self.nbins}.npy", self.newco)
            np.save(f"{self.folderName}/trueval_{self.nbins}.npy", self.trueval)
            pool.close()
        else:
            self.newco = np.load(f"{self.folderName}/newco_{self.nbins}.npy")
            correction = self.newco
            if os.path.isfile(f"{self.folderName}/trueval_{self.nbins}.npy"):
                self.trueval = np.load(f"{self.folderName}/trueval_{self.nbins}.npy")
            else:
                self.trueval = -0.5 * np.log(
                    1 - (np.arange(self.steps) / self.steps) ** 2
                )
                np.save(f"{self.folderName}/trueval_{self.nbins}.npy", self.trueval)

            weights = np.zeros_like(self.trueval)
            weights[:-1] += 0.5 * (self.trueval[1:] - self.trueval[:-1])
            weights[1:] += weights[:-1]
            deviation = np.sqrt(
                np.average(np.square(self.newco[:] - self.trueval[:]), weights=weights)
            )

        # this is needed to get an estimate of the size of the bias we are correcting
        weights = np.zeros_like(self.trueval)
        weights[:-1] += 0.5 * (self.trueval[1:] - self.trueval[:-1])
        weights[1:] += weights[:-1]
        deviation = np.sqrt(
            np.average(np.square(correction[:] - self.trueval[:]), weights=weights)
        )

        plt.title(f"RMS correction: {deviation:.4}")
        plt.plot(self.trueval, correction)
        plt.plot(self.trueval, self.newco)
        plt.plot(
            [min(self.trueval), max(self.trueval)],
            [min(self.trueval), max(self.trueval)],
            ":k",
        )
        plt.xlabel("True MI")
        plt.ylabel("Estimated MI")
        if not os.path.isfile(f"{self.folderName}/correctionMap_{self.nbins}.pdf"):
            plt.savefig(f"{self.folderName}/correctionMap_{self.nbins}.pdf", bbox_inches="tight")
        if self.display:
            plt.show()
        else:
            plt.close()

    def _correctI(self, I):
        ind = np.argmin(np.abs(I - self.newco))
        return self.trueval[ind]
