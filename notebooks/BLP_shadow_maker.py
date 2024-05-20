import json
import matplotlib.pyplot as plt
import io, h5py, zipfile
import numpy as np
from tqdm import tqdm
import os
import mienc.support as ms
from mienc import Corrector, NonLinearEstimator
import pandas as pd
from scipy.signal import hilbert
from scipy.io import loadmat, savemat
import multiprocessing as mp

AVG = 2  # 1 4


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


shadow = np.empty((1120 // AVG, 195, 50, 9))
for subject in tqdm(range(50), total=50):
    tmp_subj = np.empty((1120 // AVG, 195, 9))
    with zipfile.PyZipFile(
        "../NonLinearityData/EEG_el_so_BLP_20230714.zip"
    ) as zip_file:
        archive = h5py.File(io.BytesIO(zip_file.read(f"Sub_{subject+1}/EEG_bands.mat")))

    for band in range(1, 10):
        band_data = archive[archive["EEG_bands"][0, band - 1]]
        tmp_pieces = []
        tot_len = 0

        for i in range(band_data.shape[0]):
            this_piece = archive[band_data[i, 0]]
            tmp = ms.surrogate(this_piece)
            z = hilbert(tmp, axis=0)
            power = np.absolute(z)
            full_samp = power.shape[0] // (125 * AVG)
            extra = full_samp * 125 * AVG < power.shape[0]
            dsp = np.zeros((full_samp + extra, power.shape[1]))
            dsp[:full_samp, :] = np.average(
                np.reshape(
                    power[: full_samp * 125 * AVG], (-1, 125 * AVG, power.shape[1])
                ),
                1,
            )
            if extra:
                dsp[-1, :] = np.average(power[full_samp * 125 * AVG :, :], 0)
            tmp_pieces.append(dsp.copy())
            tot_len += dsp.shape[0]
            if tot_len > 1120 / AVG:
                break
        tmp_subj[:, :, band - 1] = np.concatenate(tmp_pieces)[: 1120 // AVG, :]
    shadow[:, :, subject, :] = tmp_subj
    np.savez_compressed(
        f"/home/raffaelli/NonLinearity/NonLinearityData/EEG_el_so_BLP_NEW/NEW_electrodeBLP_fixedTime_avg{AVG:1}_shadow_bands",
        BLP=shadow[:, :, :, band - 1],
    )
for band in range(1, 10):
    savemat(
        f"/home/raffaelli/NonLinearity/NonLinearityData/EEG_el_so_BLP_NEW/NEW_electrodeBLP_fixedTime_avg{AVG:1}_shadow_band{band}.mat",
        {"BLP": shadow[:, :, :, band - 1]},
    )
