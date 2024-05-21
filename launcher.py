#!/usr/bin/env python3
from mienc.nonlinearestimator import NonLinearEstimator as NLE
import numpy as np
from scipy.io import loadmat

regions = [
    10,
    30,
    50,
    70,
    100,
    150,
    200,
    230,
    270,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    800,
    850,
    900,
    950,
]

# nle = NLE(
#     config_file="config.ini",
#     bins=8,
#     surrogates=99,
#     verbose=True,
#     cache="cache/",
#     save_out=True,
#     dataset="trimmed_EEG_bands_time",
#     workers=23,
# )
# for t in [1823, 912, 456]:
#     tmp = nle.estimate(
#         display=False,
#         dataset_sub="5",
#         extended_stats=True,
#         compute_shadow="extend",
#         suffix=f"B{t}_Ext",
#         truncate_input=t,
#     )
#     print(f"B{t}_Ext")
#     for k in tmp:
#         print(k, np.mean(tmp[k]))
#     tmp = nle.estimate(
#         display=False,
#         dataset_sub="5",
#         extended_stats=True,
#         compute_shadow=True,
#         suffix=f"B{t}_Nor",
#         truncate_input=t,
#     )
#     print(f"B{t}_Nor")
#     for k in tmp:
#         print(k, np.mean(tmp[k]))


# for band in range(1, 10):
#     samples = loadmat(f"/home/raffaelli/NonLinearity/NonLinearityData/EEG_el_so_BLP_NEW/trimmed_EEG_fixedTime_band{band}.mat")["EEG"].shape[0]
#     bins = max(8, int(np.power(samples, 1/3)))
#     nle = NLE(config_file="config.ini", bins=bins, surrogates=99, verbose=True,
#               cache="cache/", save_out=True, dataset="trimmed_EEG_bands_time", suffix="Extended", workers=23)
#     nle.estimate(display=False, dataset_sub=str(band), extended_stats=True, compute_shadow="extend")
#     del nle

# bands={i:b for i, b in enumerate([[ 1.,  4.], [ 4.,  8.], [ 8., 12.], [12., 15.], [15., 18.], [18., 30.], [30., 44.], [12., 30.], [ 1., 40.]], start=1)}
# def fs_band(band):
#     return int(bands[band][1]*2*1.125+0.5)

# for band in range(1,8):
#     samples=45*fs_band(band)
#     bins = max(8, int(np.power(samples, 1/3)))
#     nle = NLE(config_file="config.ini", bins=bins, surrogates=99, verbose=True,
#                 cache="cache/", save_out=False, dataset="trimmed_EEG_bands_time", suffix="Cutting", workers=23)
#     nle.estimate(display=False, dataset_sub="7", extended_stats=True, compute_shadow=True, truncate_input=samples)
#     del nle

# nle = NLE(config_file="config.ini", bins=8, surrogates=99, cache="cache", save_out=20, dataset="shaved_eso245_cra_strin",workers=23, suffix="Extended", verbose=True)

# for region in regions:
#     nle.estimate(display=False, dataset_sub=str(region), extended_stats=True, compute_shadow="extend")

# del nle

# nle = NLE(config_file="config.ini", bins=8, surrogates=99, cache="cache", save_out=True, dataset="FIX_electrode_BLP",workers=23)

# for band in range(1,10):
#     nle.estimate(display=False, dataset_sub=str(band), extended_stats=True, compute_shadow=True)

# del nle

# nle = NLE(config_file="config.ini", bins=8, surrogates=99, cache="cache", save_out=1, dataset="FIX_source_BLP",workers=23)

# for band in range(1,10):
#     nle.estimate(display=False, dataset_sub=str(band), extended_stats=True, compute_shadow=True)

# del nle

# for band in range(1, 10):
#     samples = loadmat(f"/home/raffaelli/NonLinearityData/iEEG/FIX_iEEG_fixedTime_band{band}.mat")["EEG"].shape[0]
#     bins = max(8, int(np.power(samples, 1/3)))
#     nle = NLE(config_file="config.ini", bins=bins, surrogates=99,
#               cache="cache", save_out=True, dataset="FIX_iEEG_time", workers=23)
#     nle.estimate(display=False, dataset_sub=str(band),
#                  extended_stats=True, compute_shadow=True)
#     del nle

# for band in range(1, 6):
#     samples = loadmat(
#         f"/home/raffaelli/NonLinearity/NonLinearityData/iEEG/iEEG_band{band}.mat"
#     )["iEEG"].shape[0]
#     bins = round(np.power(samples, 1 / 3))
#     print(samples, bins)
#     nle = NLE(
#         config_file="config.ini",
#         bins=bins,
#         surrogates=99,
#         cache="cache",
#         save_out=True,
#         dataset="iEEG_long",
#         workers=23,
#     )
#     nle.estimate(
#         display=False, dataset_sub=str(band), extended_stats=True, compute_shadow=True
#     )
#     del nle


for i, samples in enumerate([1116, 2232, 3348, 8432, 12276], start=1):
    bins = round(np.power(samples, 1 / 3))
    print(samples, bins)
    nle = NLE(
        config_file="config.ini",
        bins=bins,
        surrogates=99,
        cache="cache",
        save_out=True,
        dataset="iEEG_long",
        workers=23,
        suffix=f"_asBand{i}",
    )
    nle.estimate(
        display=False,
        dataset_sub="5",
        extended_stats=True,
        compute_shadow=True,
        truncate_input=samples,
    )
    del nle

# nle = NLE(config_file="config.ini", bins=10, surrogates=99, cache="cache", save_out=True, dataset="FIX_iEEG_samples", workers=23)

# for band in range(1, 10):
#     nle.estimate(display=False, dataset_sub=str(band),
#                  extended_stats=True, compute_shadow=True)

# del nle

# for band in range(1, 10):
#     samples = loadmat(f"/home/raffaelli/NonLinearityData/EEG_el_so_BLP_NEW/NEW_EEG_fixedTime_band{band}.mat")["EEG"].shape[0]
#     bins = max(8, int(np.power(samples, 1/3)))
#     nle = NLE(config_file="config.ini", bins=bins, surrogates=99,
#               cache="cache", save_out=True, dataset="NEW_EEG_bands_time", workers=23)
#     nle.estimate(display=False, dataset_sub=str(band),
#                  extended_stats=True, compute_shadow=True)
#     del nle

# nle = NLE(config_file="config.ini", bins=10, surrogates=99, cache="cache", save_out=True, dataset="NEW_EEG_bands_samples", workers=23)

# for band in range(1, 10):
#     nle.estimate(display=False, dataset_sub=str(band),
#                  extended_stats=True, compute_shadow=True)

# del nle

# for kind, save_out in [("electrode", True), ("source", 1)]:
#     print(kind)
#     prefix = "NEW" if kind=="electrode" else "PCA"
#     for av in [1,2,4]:
#         for who in ["", "_shadow"]:
#             if not who and kind=="electrode": #true electrode c'è già
#                 continue

#             print(str(av)+who, "\t1", end="\t")
#             samples = loadmat(f"/home/raffaelli/NonLinearity/NonLinearityData/EEG_el_so_BLP_NEW/{prefix}_{kind}BLP_fixedTime_avg{av}{who}_band1.mat")["EEG" if kind=="source" else "BLP"].shape[0]
#             bins = int(np.power(samples, 1/3))

#             nle = NLE(config_file="config.ini", bins=bins, surrogates=99, cache="cache", save_out=save_out, dataset=f"{prefix}_{kind}BLP_time_avg{av}{who}", workers=23, verbose=True)
#             for band in range(1, 10):
#                 if av==1 and kind=="electrode": #avg1 c'è già
#                     continue
#                 nle.estimate(display=False, dataset_sub=str(band),
#                             extended_stats=True, compute_shadow=(not who))
#                 nle.verbose = False
#             del nle
#             if av == 4:
#                 print()
#                 continue

#             print(" 2")
#             nle = NLE(config_file="config.ini", bins=6, surrogates=99, cache="cache", save_out=save_out, dataset=f"{prefix}_{kind}BLP_samples_avg{av}{who}", workers=23, verbose=True)

#             for band in range(1, 10):
#                 nle.estimate(display=False, dataset_sub=str(band),
#                             extended_stats=True, compute_shadow=(not who))
#                 nle.verbose = False

#             del nle


# %%
# import matplotlib.pyplot as plt
# import numpy as np

# corr = np.load("../NonLinearityResultsNew/eso245_cra_strin_10_bin8/correction_400_8.npy")
# true = -0.5 * np.log(1 - (np.arange(200) / 200) ** 2)
# #%%
# plt.plot(corr[:50], true[:50], ".-")
# plt.show()
# #%%
# print(*np.polyfit(corr[:48], true[:48],2,full=True))
# # %%

# from mienc import Corrector
# import json

# for p in range(7):
#     samples = int(16000/2**p)
#     bins = int(samples**(1/3))
#     Corrector(bins, samples, cache_dir="cache",workers=23, config="config.ini").compute_correction()

# times = [125*2**i for i in range(7)]
# GOOD_SESSIONS = [831882777, 816200189, 771160300, 786091066, 779839471, 778998620, 781842082, 778240327, 793224716, 839068429, 794812542, 768515987, 767871931, 840012044, 766640955, 847657808]
# for session in GOOD_SESSIONS:
#    for t in times:
#        nle = NLE(config_file="config.ini", bins=0, surrogates=99, jitter=True,
#                cache="cache", save_out=True, dataset=f"spiking_{session}", workers=23)
#        nle.estimate(display=False, dataset_sub=f"{t:04}",
#                    extended_stats=True, compute_shadow=True)
#        del nle
