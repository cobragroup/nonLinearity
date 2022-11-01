#!/usr/bin/env python3
# %%
import os
import sys
from support import pair_mutual_information, surrogate, surrogate
from corrector import Corrector
from warnings import warn
import scipy.io as sio
import json
import configparser
import multiprocessing as mp
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


# %%
def read_Config (configFile = None):
    config = configparser.ConfigParser()
    config.read(configFile if configFile is not None else 'options.ini')

    steps = config.getint('correction','steps',fallback=200)
    iters = config.getint('correction','iters',fallback=1000)
    nsamples = config.getint('correction','nsamples',fallback=0)

    nbins = config.getint('global','nbins',fallback=8)
    display = config.getboolean('global','display',fallback=True)
    workers = config.getboolean('global','workers',fallback=4)

    Nsurrogates = config.getint('estimate','Nsurrogates', fallback=99)

    filePath = config.get('dataset','filePath', fallback="./")
    fileName = config.get('dataset','fileName', fallback=None)
    fieldName = config.get('dataset','fieldName', fallback=None)
    assert fileName is not None, "Missing dataset filename in .ini file."
    assert fieldName is not None, "Missing dataset fieldname in .ini file."
    folderName = os.path.splitext(fileName)[0]+f"_bin{nbins}"
    hc_start = config.getint('dataset','healthy_control_start', fallback=None)
    hc_start = hc_start if hc_start else None
    hc_end = config.getint('dataset','healthy_control_end', fallback=None)
    hc_end = hc_end if hc_end else None
    hc_slice = slice(hc_start, hc_end)

    if not os.path.isdir(os.path.join(filePath,folderName)):
        os.mkdir(os.path.join(filePath,folderName))
    print(f"Using: {os.path.abspath(os.path.join(filePath, fileName))} and {os.path.abspath(os.path.join(filePath,folderName))}")
        
    return steps, iters, nsamples, nbins, display, Nsurrogates, os.path.join(filePath, fileName), fieldName, os.path.join(filePath,folderName), hc_slice, workers

# %%
def load_data (fileName, fieldName, hc_slice):
    mat = sio.loadmat(fileName)[fieldName][:, :, hc_slice]
    duration, regions, sessions = mat.shape

    return duration, regions, sessions, mat

# %%
def run (configFile):
    steps, iters, nsamples, nbins, display, Nsurrogates, fileName, fieldName, folderName, hc_slice, workers = read_Config (configFile)

    duration, regions, sessions, mat = load_data (fileName, fieldName, hc_slice)
    if nsamples == 0:
        nsamples = duration
    if duration != nsamples:
        warn(f"Acquisition duration ({duration}) is different from the set number of samples for correction ({nsamples}).")
    corrector = Corrector(steps, folderName, iters, nsamples, nbins, workers, display=display)
    corrector.compute_correction()

    nsamples = duration
    pairNum = int((regions * (regions-1))/2)

    globalratio95control = np.zeros(sessions)
    globalratio99control = np.zeros(sessions)
    globalratio95 = np.zeros(sessions)
    globalratio99 = np.zeros(sessions)
    globaltotalMI = np.zeros(sessions)
    globalgaussMI = np.zeros(sessions)
    globalratio95shadow = np.zeros(sessions)
    globalratio99shadow = np.zeros(sessions)
    globaltotalMIshadow = np.zeros(sessions)
    globalgaussMIshadow = np.zeros(sessions)

    pool = mp.Pool(workers)
    for patientN in range(sessions):
        if not os.path.isfile(f"{folderName}/patient{patientN:02}_{nbins}.npy"):
            shadow = surrogate(mat[:, :, patientN])
            pair = 0
            statsMI = np.zeros([pairNum, Nsurrogates+1])
            for ns, patient in tqdm(enumerate(task_producer(mat[:, :, patientN], Nsurrogates)), total=Nsurrogates+1, desc=f"Patient {patientN} true"):
                statsMI[:, ns] = pool.starmap(pair_mutual_information, ((
                    patient[:, zone1], patient[:, zone2], nbins) for zone1 in range(regions) for zone2 in range(zone1+1, regions)))

            statsMI_shadow = np.zeros([pairNum, Nsurrogates+1])
            for ns, patient in tqdm(enumerate(task_producer(shadow[:, :], Nsurrogates)), total=Nsurrogates+1, desc=f"Patient {patientN} shadow"):
                statsMI_shadow[:, ns] = pool.starmap(pair_mutual_information, ((
                    patient[:, zone1], patient[:, zone2], nbins) for zone1 in range(regions) for zone2 in range(zone1+1, regions)))

            # statsMI_univar = np.zeros([pairNum, Nsurrogates])
            # for ns, patient in tqdm(enumerate(task_producer(mat[:, :, patientN], Nsurrogates, False)), total=Nsurrogates+1, desc=f"Patient {patientN} univar"):
            #     statsMI_univar[:, ns] = pool.starmap(pair_mutual_information, ((
            #         patient[:, zone1], patient[:, zone2]) for zone1 in range(regions) for zone2 in range(zone1+1, regions)))

            corr = np.array([np.corrcoef(mat[:, zone1, patientN], mat[:, zone2, patientN])[
                            1, 0] for zone1 in range(regions) for zone2 in range(zone1+1, regions)])

            np.save(f"{folderName}/patient{patientN:02}_{nbins}", statsMI)
            np.save(f"{folderName}/patient{patientN:02}_{nbins}_sha", statsMI_shadow)
            # np.save(f"{folderName}/patient{patientN:02}_{nbins}_uni", statsMI_univar)
            np.save(f"{folderName}/patient{patientN:02}_{nbins}_cor", corr)
        else:
            statsMI = np.load(f"{folderName}/patient{patientN:02}_{nbins}.npy")
            statsMI_shadow = np.load(
                f"{folderName}/patient{patientN:02}_{nbins}_sha.npy")
            # statsMI_univar = np.load(
            #     f"{folderName}/patient{patient:02}_{nbins}_uni.npy")
            corr = np.load(f"{folderName}/patient{patientN:02}_{nbins}_cor.npy")

        correctedperc95pointer = (Nsurrogates*(0.95)-0.5)/(Nsurrogates-1)
        correctedperc99pointer = (Nsurrogates*(0.99)-0.5)/(Nsurrogates-1)
        correctedperc01pointer = (Nsurrogates*(0.01)-0.5)/(Nsurrogates-1)

        perc95 = np.quantile(statsMI[:, 1:], correctedperc95pointer, 1)
        nonlin95 = statsMI[:, 0] > perc95
        ratio95 = np.mean(nonlin95)
        perc99 = np.quantile(statsMI[:, 1:], correctedperc99pointer, 1)
        nonlin99 = statsMI[:, 0] > perc99
        ratio99 = np.mean(nonlin99)

        # pvalue_MI = 1 - np.sum(statsMI_univar < statsMI[:, 0], 1)/(Nsurrogates+1)
        pvalue_NMI = 1 - np.sum(statsMI[:,1:].T < statsMI[:, 0], 0)/(Nsurrogates+1)
        ratio95control = np.mean(pvalue_NMI<0.0500001)
        ratio99control = np.mean(pvalue_NMI<0.0100001)
        globalratio95control[patientN] = ratio95control
        globalratio99control[patientN] = ratio99control

        perc95_shadow = np.quantile(statsMI_shadow[:, 1:], correctedperc95pointer, 1)
        nonlin95_shadow = statsMI_shadow[:, 0] > perc95_shadow
        ratio95_shadow = np.mean(nonlin95_shadow)
        perc99_shadow = np.quantile(statsMI_shadow[:, 1:], correctedperc99pointer, 1)
        nonlin99_shadow = statsMI_shadow[:, 0] > perc99_shadow
        ratio99_shadow = np.mean(nonlin99_shadow)

        corr_statsMI = corrector.correctI(statsMI)

        mean_cont_mi_multisurr=np.mean(corr_statsMI[:,1:],1)
        std_cont_mi_multisurr=np.std(corr_statsMI[:,1:],1)
        perc99_PLOT = np.quantile(corr_statsMI[:, 1:], correctedperc99pointer, 1)
        perc01_PLOT = np.quantile(corr_statsMI[:, 1:], correctedperc01pointer, 1)

        allpairs_cont_mi_data = np.mean(corr_statsMI[:,1])
        allpairs_mean_cont_mi_multisurr = np.mean(corr_statsMI[:,1:])

        # corr_statsMI_univar = corrector.correctI(statsMI_univar)
        # mean_cont_mi_unisurr=np.mean(corr_statsMI_univar,1)
        # std_cont_mi_unisurr=np.std(corr_statsMI_univar,1)
        # allpairs_mean_cont_mi_unisurr = np.mean(corr_statsMI_univar)

        globalratio95[patientN] = ratio95
        globalratio99[patientN] = ratio99
        globaltotalMI[patientN] = allpairs_cont_mi_data
        globalgaussMI[patientN] = allpairs_mean_cont_mi_multisurr

        corr_statsMI_shadow = corrector.correctI(statsMI_shadow)

        mean_cont_mi_multisurrshadow=np.mean(corr_statsMI_shadow[:,1:],1)
        std_cont_mi_multisurrshadow=np.std(corr_statsMI_shadow[:,1:],1)

        allpairs_cont_mi_datashadow = np.mean(corr_statsMI_shadow[:,1])
        allpairs_mean_cont_mi_multisurrshadow = np.mean(corr_statsMI_shadow[:,1:])

        globalratio95shadow[patientN] = ratio95_shadow
        globalratio99shadow[patientN] = ratio99_shadow
        globaltotalMIshadow[patientN] = allpairs_cont_mi_datashadow
        globalgaussMIshadow[patientN] = allpairs_mean_cont_mi_multisurrshadow


        plt.scatter(corr, corr_statsMI[:,1])
        neworder = np.argsort(corr)
        expected = -0.5*np.log(1-corr**2)
        plt.plot(corr[neworder], expected[neworder], 'purple')
        plt.plot(corr[neworder], mean_cont_mi_multisurr[neworder], 'r')
        plt.plot(corr[neworder], perc01_PLOT[neworder], 'lightblue')
        plt.plot(corr[neworder], perc99_PLOT[neworder], 'g')
        plt.xlabel('correlation')
        plt.ylabel('mutual information (bits)')
        plt.title(
            f"Patient {patientN} - {allpairs_cont_mi_data:.3}/{allpairs_mean_cont_mi_multisurr:.3} (^{ratio95:.3}-{ratio95control:.3}_{ratio95_shadow:.3})")
        plt.ylim(bottom=0)
        plt.savefig(f"{folderName}/patient{patientN:02}_{nbins}.pdf")
        if display:
            plt.show()
        else:
            plt.close()
    pool.close()
    globalStats={"globalratio95control":globalratio95control.tolist(), "globalratio99control":globalratio99control.tolist(), "globalratio95":globalratio95.tolist(), "globalratio99":globalratio99.tolist(), "globaltotalMI":globaltotalMI.tolist(), "globalgaussMI":globalgaussMI.tolist(), "globalratio95shadow":globalratio95shadow.tolist(), "globalratio99shadow":globalratio99shadow.tolist(), "globaltotalMIshadow":globaltotalMIshadow.tolist(), "globalgaussMIshadow":globalgaussMIshadow.tolist()}
    with open(f"{folderName}/globalStats",'w') as fp:
        json.dump(globalStats, fp)


if __name__ == "__main__":
    if len(sys.argv)>1:
        config = sys.argv[1]
    else: config = None

    run(config)