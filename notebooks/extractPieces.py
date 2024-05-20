import io, h5py, zipfile
import numpy as np
from tqdm import tqdm
import pickle
from scipy import signal

bands = {
    i: b
    for i, b in enumerate(
        [
            [1.0, 4.0],
            [4.0, 8.0],
            [8.0, 12.0],
            [12.0, 15.0],
            [15.0, 18.0],
            [18.0, 30.0],
            [30.0, 44.0],
            [12.0, 30.0],
            [1.0, 40.0],
        ],
        start=1,
    )
}

basePath = "../NonLinearityData/EEG_el_so_BLP_NEW"
f_sample = 1000
fs_under = 120


def fs_band(band):
    return int(bands[band][1] * 2 * 1.125 + 0.5)


for band in range(9):
    with open(f"auxiliary_data/new_pieces_band{band+1}.pickle", "wb") as fp:
        pass

zip_file = zipfile.PyZipFile("../NonLinearityData/EEG_el_so_BLP_20230714.zip")
new_pieces = [[[] for subject in range(50)] for band in range(9)]
for subject in tqdm(range(50), desc="Subject"):
    archive = h5py.File(io.BytesIO(zip_file.read(f"Sub_{subject+1}/EEG_bands.mat")))

    for band in tqdm(bands, desc="Band", total=9):
        band_data = archive[archive["EEG_bands"][0, band - 1]]
        fs_dest = fs_band(band)
        needed_samples = max(1000, 45 * fs_dest)
        pieces = []
        tot_len = 0

        for i in range(band_data.shape[0]):
            this_piece = archive[band_data[i, 0]]
            chunks = int(this_piece.shape[0] / f_sample)

            if chunks == 0:
                continue

            tot_samples_dest = chunks * fs_dest
            border = max(1, tot_samples_dest // 20)
            tot_len += tot_samples_dest - 2 * border

            tot_samples_under = chunks * fs_under
            blaf = signal.resample(
                this_piece[: f_sample * chunks, :], tot_samples_under
            )
            new_pieces[band - 1][subject].append(blaf)
            if tot_len > needed_samples:
                break

for band in range(9):
    with open(f"auxiliary_data/new_pieces_band{band+1}.pickle", "wb") as fp:
        pickle.dump(new_pieces[band], fp)
