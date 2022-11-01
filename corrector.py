from support import single_iter
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np


class Corrector:
    def __init__(
        self,
        steps,
        folderName,
        iters,
        nsamples,
        nbins,
        workers=4,
        smoothed=False,
        display=False,
    ):
        self.steps = steps
        self.folderName = folderName
        self.iters = iters
        self.nsamples = nsamples
        self.nbins = nbins
        self.smoothed = smoothed
        self.display = display
        self.workers = workers
        self.newco = None
        self.trueval = None
        self.correctI = np.vectorize(self._correctI)

    def compute_correction(self):
        """Computes the correction lookup table or loads the cached values."""
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
            plt.savefig(f"{self.folderName}/correctionMap_{self.nbins}.pdf")
        if self.display:
            plt.show()
        else:
            plt.close()

    def _correctI(self, I):
        ind = np.argmin(np.abs(I - self.newco))
        return self.trueval[ind]
