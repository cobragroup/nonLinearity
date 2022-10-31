from support import *
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp

def compute_correction(steps, folderName, iters, nsamples, nbins, smoothed=False, display=False):
    global bins
    incre = 1/steps
    correction = np.zeros(steps)
    trueval = np.zeros(steps)
    if not os.path.isfile(f"{folderName}/newco_{nbins}.npy"):
        pool = mp.Pool(24)
        for i in tqdm(range(steps),"Computing correction"):
            means = 0, 0
            corre = [[1, i*incre], [i*incre, 1]]
            I = pool.map(single_iter, ((means, corre, nsamples, nbins)
                         for __ in range(iters)))
            correction[i] = np.average(I)
            trueval[i] = -0.5*np.log(1-(i*incre)**2)

        if smoothed:
            tosmo = np.array([correction[0], ]*2 +
                             correction.tolist()+[correction[-1], ]*2)
            newco = np.zeros_like(correction)
            for i in range(len(correction)):
                newco[i] = np.mean(tosmo[i:i+5])
            if display:
                try:
                    plt.plot(correction[:50])
                    plt.plot(newco[:50])
                    plt.show()
                except:
                    pass
        else:
            newco = correction

        np.save(f"{folderName}/newco_{nbins}.npy", newco)
        np.save(f"{folderName}/trueval_{nbins}.npy", trueval)
        pool.close()
    else:
        newco = np.load(f"{folderName}/newco_{nbins}.npy")
        correction = newco
        trueval = np.load(f"{folderName}/trueval_{nbins}.npy")
        weights = np.zeros_like(trueval)
        weights[:-1] += 0.5*(trueval[1:]-trueval[:-1])
        weights[1:] += weights[:-1]
        deviation = np.sqrt(np.average(
            np.square(newco[:]-trueval[:]), weights=weights))
    if display:
        # this is needed to get an estimate of the size of the bias we are correcting
        weights = np.zeros_like(trueval)
        weights[:-1] += 0.5*(trueval[1:]-trueval[:-1])
        weights[1:] += weights[:-1]
        deviation = np.sqrt(np.average(
            np.square(correction[:]-trueval[:]), weights=weights))

        plt.title(deviation)
        plt.plot(trueval, correction)
        plt.plot(trueval, newco)
        plt.plot([min(trueval), max(trueval)], [
                 min(trueval), max(trueval)], ":k")
        plt.xlabel('True MI')
        plt.ylabel('Estimated MI')
        plt.show()
    return trueval, newco