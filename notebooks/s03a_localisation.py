import os, sys, configparser, h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FixedFormatter
import seaborn as sns
import numpy as np
import pandas as pd

from scipy.stats import ks_1samp, uniform, binom
from scipy.spatial import Voronoi

from nilearn import datasets, plotting, image
import nibabel as nib

settings_parser = configparser.ConfigParser()
settings_parser.read("localsettings.ini")
MAIN_DATA_FOLDER = settings_parser.get("global", "data_path")
MIENC_PATH = settings_parser.get("global", "mienc_path")
cache_dir = os.path.join(MAIN_DATA_FOLDER, "cache")
config_ini = os.path.join(MAIN_DATA_FOLDER, "config.ini")

sys.path.append(os.path.abspath(MIENC_PATH))
from mienc import Corrector


sns.set_theme("poster", "ticks")
DATA_DIR_fMRI = os.path.join(MAIN_DATA_FOLDER, "fMRI_region_size")
DATA_DIR_EEG = os.path.join(MAIN_DATA_FOLDER, "EEG")

aal_atlas_centers = pd.read_excel(os.path.join(DATA_DIR_fMRI, "AAL_regions.xls"))
aal_atlas_centers_labels = (
    aal_atlas_centers["<b>Labels</b>"].apply(lambda x: x.split()[1]).tolist()
)
atlas_data = datasets.fetch_atlas_aal()
atlas_filename = atlas_data.maps
region_indices = {l: i for l, i in zip(atlas_data["labels"], atlas_data["indices"])}

reg_loc = []
image_data = image.get_data(atlas_filename)
affine = image.load_img(atlas_filename).affine
for l in region_indices:
    if l in aal_atlas_centers_labels:
        reg_loc.append(np.where(image_data == int(region_indices[l])))

infoMat = h5py.File(os.path.join(DATA_DIR_EEG, "info.mat"))
vec = np.where(infoMat["elec_in_mask"])[1]
electrode_positions = (
    pd.read_csv(os.path.join(DATA_DIR_EEG, "electrode_positions.csv"))
    .loc[vec, ["Labels", "x", "y", "z"]]
    .reset_index()
)


div_palette = sns.color_palette("vlag", as_cmap=True)  # plt.cm.seismic
div_palette.set_over(div_palette.get_bad())
div_palette.set_under(color=[0.5, 0.5, 0.5, 0.5])
div_palette.set_bad(color="grey")


def plot_brain(
    in_array,
    title=None,
    cut_coords=None,
    axes=None,
    cmap=div_palette,
    vmin=None,
    vmax=None,
    norm=None,
):
    MAXINT = np.iinfo(np.int32).max

    if norm is None:
        if vmin is None:
            vmin = np.nanmin(in_array)
        if vmax is None:
            vmax = np.nanmax(in_array)
        normalised = (in_array - vmin) / (vmax - vmin)
    else:
        normalised = norm(in_array)
    normalised[np.isinf(normalised)] = 1
    values = np.full_like(image_data, np.nan, dtype=np.int32)
    for i in range(90):
        values[reg_loc[i][0], reg_loc[i][1], reg_loc[i][2]] = normalised[i] * MAXINT

    img = nib.nifti1.Nifti1Image(values, affine)
    plotting.plot_roi(
        img,
        title=title,
        cut_coords=cut_coords,
        cmap=cmap,
        axes=axes,
        vmin=0,
        vmax=MAXINT,
    )


def plot_cap(
    in_array,
    title=None,
    axes=None,
    cmap=div_palette,
    vmin=None,
    vmax=None,
    norm=None,
    plane_distance=10,
):
    source_z = electrode_positions.z.min()
    diameter = electrode_positions.z.max() - source_z
    electrode_positions["XP"] = (
        electrode_positions.x
        / (electrode_positions.z - source_z + plane_distance)
        * (diameter + plane_distance)
    )
    electrode_positions["YP"] = (
        electrode_positions.y
        / (electrode_positions.z - source_z + plane_distance)
        * (diameter + plane_distance)
    )
    if norm is None:
        if vmin is None:
            vmin = np.nanmin(in_array)
        if vmax is None:
            vmax = np.nanmax(in_array)
        normalised = (in_array - vmin) / (vmax - vmin)
    else:
        normalised = norm(in_array)
    vor = Voronoi(
        np.concatenate(
            [
                electrode_positions[["XP", "YP"]],
                np.transpose(
                    [
                        22 * np.sin(np.linspace(0, 2 * np.pi, 100)),
                        22 * np.cos(np.linspace(0, 2 * np.pi, 100)) - 0.5,
                    ]
                ),
            ]
        )
    )
    if axes is not None:
        plt.sca(axes)
    plt.gca().set_aspect("equal")

    for i, v in enumerate(normalised):
        region = vor.regions[vor.point_region[i]]
        vertices = np.array([vor.vertices[n] for n in region])
        plt.fill(vertices[:, 0], vertices[:, 1], color=cmap(v))
    if title is not None:
        plt.title(title)


def HolmThresholdFromP(p_values: np.ndarray):
    """Returns the Holm-Bonferroni threshold given an array of p-values.
    NOTA BENE: reject the null hypotheses when **strictly smaller** than this thresold. This corresponds to the *p-value* of the first non-rejected null hypothesis.

    Parameters
    ----------
    p_values : np.ndarray
        The *p-values* to consider, can be N-dimensional, will be flattened.

    Returns
    -------
    float
        The *p-value* of the first non-rejected hypothesis.
    """
    sorted_p = np.sort(p_values.flatten())
    good = sorted_p < 0.05 / (sorted_p.size - np.arange(sorted_p.size))
    which = np.argmin(good)

    if which == 0 and sorted_p[-1] < 0.05:
        return 0.05

    return sorted_p[which]


def compute_localised_non_linearity(
    results_file,
    subset_identifiers,
    output_prefix,
    pair_num,
    subj_num,
    nbins,
    samples,
):
    if isinstance(nbins, int):
        nbins = [
            nbins,
        ] * len(subset_identifiers)
    else:
        assert len(nbins) == len(subset_identifiers)

    if isinstance(samples, int):
        samples = [
            samples,
        ] * len(subset_identifiers)
    else:
        assert len(samples) == len(subset_identifiers)

    for subset_id, bins, samp in zip(subset_identifiers, nbins, samples):
        corrct = Corrector(
            bins,
            samp,
            folder_name=os.path.dirname(results_file).format(subset_id, bins),
            config=config_ini,
            cache_dir=cache_dir,
        )
        corrct.compute_correction()

        if not (
            os.path.isfile(
                os.path.join(
                    MAIN_DATA_FOLDER,
                    "Results",
                    "localised",
                    f"{output_prefix}_ks_stat_{subset_id}.npy",
                )
            )
            and os.path.isfile(
                os.path.join(
                    MAIN_DATA_FOLDER,
                    "Results",
                    "localised",
                    f"{output_prefix}_ks_p_{subset_id}.npy",
                )
            )
        ):
            if not os.path.isfile(
                os.path.join(
                    MAIN_DATA_FOLDER,
                    "Results",
                    "localised",
                    f"{output_prefix}_region_quantiles_{subset_id}.npy",
                )
            ):
                region_quantiles = np.zeros((subj_num, pair_num))
                for s in tqdm(range(subj_num), "Subject"):
                    pat = np.load(results_file.format(subset_id, bins, s))
                    true = pat[:, 0]
                    surr = np.sort(pat[:, 1:], 1)
                    for i in range(pair_num):
                        region_quantiles[s, i] = np.searchsorted(
                            surr[i, :], true[i], "left"
                        )
                    np.save(
                        os.path.join(
                            MAIN_DATA_FOLDER,
                            "Results",
                            "localised",
                            f"{output_prefix}_region_quantiles_{subset_id}.npy",
                        ),
                        region_quantiles,
                    )
            ks_stat = np.zeros(pair_num)
            ks_p = np.zeros(pair_num)
            for i in tqdm(range(pair_num), "Pair"):
                stat, pval = ks_1samp(
                    region_quantiles[:, i], uniform.cdf, (0, 100), alternative="less"
                )
                ks_stat[i] = stat
                ks_p[i] = pval
            np.save(
                os.path.join(
                    MAIN_DATA_FOLDER,
                    "Results",
                    "localised",
                    f"{output_prefix}_ks_stat_{subset_id}.npy",
                ),
                ks_stat,
            )
            np.save(
                os.path.join(
                    MAIN_DATA_FOLDER,
                    "Results",
                    "localised",
                    f"{output_prefix}_ks_p_{subset_id}.npy",
                ),
                ks_p,
            )


def show_localised_non_linearity(
    results_file,
    subset_description,
    subset_identifiers,
    subset_names,
    output_prefix,
    pair_num,
    elec_num,
    cut_position=(-15, -75, 27),
):
    global MAX_CM, MAX_CM2
    for subset_id, subset_na in zip(subset_identifiers, subset_names):
        ks_stat = np.load(
            os.path.join(
                MAIN_DATA_FOLDER,
                "Results",
                "localised",
                f"{output_prefix}_ks_stat_{subset_id}.npy",
            )
        )
        ks_p = np.load(
            os.path.join(
                MAIN_DATA_FOLDER,
                "Results",
                "localised",
                f"{output_prefix}_ks_p_{subset_id}.npy",
            )
        )

        ks_stat_sha = np.load(
            os.path.join(
                MAIN_DATA_FOLDER,
                "Results",
                "localised",
                f"{output_prefix+'_sha'}_ks_stat_{subset_id}.npy",
            )
        )
        ks_p_sha = np.load(
            os.path.join(
                MAIN_DATA_FOLDER,
                "Results",
                "localised",
                f"{output_prefix+'_sha'}_ks_p_{subset_id}.npy",
            )
        )
        thresh = HolmThresholdFromP(np.concatenate([ks_p, ks_p_sha]))
        print(thresh)
        corrected = np.full(pair_num, np.nan)
        corrected[ks_p < thresh] = ks_stat[ks_p < thresh]
        sig_pair = np.zeros([elec_num, elec_num])
        sig_pair[np.triu_indices(elec_num, 1)] = corrected
        sig_pair += sig_pair.T
        np.fill_diagonal(sig_pair, np.nan)
        siginreg = np.sum(sig_pair > 0, 1)
        print("Non linear connections:", np.sum(siginreg) / 2)

        # thresh_sha = HolmThresholdFromP(ks_p_sha)

        corrected_sha = np.full(pair_num, np.nan)
        corrected_sha[ks_p_sha < thresh] = ks_stat_sha[ks_p_sha < thresh]
        sig_pair_sha = np.zeros([elec_num, elec_num])
        sig_pair_sha[np.triu_indices(elec_num, 1)] = corrected_sha
        sig_pair_sha += sig_pair_sha.T
        np.fill_diagonal(sig_pair_sha, np.nan)
        siginreg_sha = np.sum(sig_pair_sha > 0, 1)
        print("Non linear connections:", np.sum(siginreg_sha) / 2)

        fix, ax = plt.subplots(
            1, 3, gridspec_kw={"width_ratios": [4, 4, 0.2]}, figsize=(12, 7)
        )
        vmax = max(np.nanmax(sig_pair), np.nanmax(sig_pair_sha))
        vmin = min(np.nanmin(sig_pair), np.nanmin(sig_pair_sha))
        plt.sca(ax[0])
        plt.imshow(sig_pair, interpolation="none", vmax=vmax, vmin=vmin)
        step = 10 if elec_num < 100 else 20
        plt.yticks(np.arange(0, elec_num, step), np.arange(elec_num)[::step])
        step = 20 if elec_num < 100 else 40
        plt.xticks(
            np.arange(0, elec_num, step),
            np.arange(elec_num)[::step],
        )
        plt.xlabel(f"Region")
        plt.title("True")

        plt.sca(ax[1])
        plt.imshow(sig_pair_sha, interpolation="none", vmax=vmax, vmin=vmin)
        plt.yticks([])
        step = 20 if elec_num < 100 else 40
        plt.xticks(
            np.arange(0, elec_num, step),
            np.arange(elec_num)[::step],
        )
        plt.xlabel(f"Region")
        plt.title("Shadow")
        plt.colorbar(ax=ax[1], cax=ax[2], shrink=0.2).ax.set_ylabel(
            "KS statistcs", rotation=90
        )

        plt.suptitle(
            subset_description
            + subset_na
            + f" - {np.sum(ks_p<thresh)} ({100*np.sum(ks_p<thresh)/pair_num:.3} %) significant pairs"
        )
        plt.show()

        if (siginreg > 0).any():

            print("Siginreg max:", np.max(siginreg))
            if "aal" in results_file:
                fig, ax = plt.subplots(
                    3, 1, gridspec_kw={"height_ratios": [4, 4, 0.5]}, figsize=(8, 8)
                )
                sc = ax[2].scatter(
                    [np.nan],
                    [np.nan],
                    c=0,
                    cmap=div_palette,
                    norm=TwoSlopeNorm(vmin=0, vcenter=elec_num / 2, vmax=elec_num - 1),
                )
                cbar = plt.colorbar(
                    sc,
                    cax=ax[2],
                    shrink=0.35,
                    ticks=[0, elec_num / 2, elec_num - 1],
                    location="bottom",
                    format=FixedFormatter(
                        [
                            "no\nnon-linear\nconnections",
                            "significantly\nabove random\ngraph",
                            "complete\nnon-linear\nconnections",
                        ]
                    ),
                )

                dist = binom(
                    elec_num - 1, np.sum(siginreg) / (elec_num * (elec_num - 1))
                )
                sig_rg = dist.ppf(1 - 0.05 / (2 * elec_num))
                norm = TwoSlopeNorm(
                    vmin=-0.0001,
                    vcenter=sig_rg if np.sum(siginreg) > 0 else elec_num / 2,
                    vmax=elec_num - 1,
                )
                plot_brain(
                    siginreg,
                    "AAL 90 regions - True",
                    cut_position,
                    ax[0],
                    div_palette,
                    norm=norm,
                )  # (-15,-75,27)

                dist = binom(
                    elec_num - 1, np.sum(siginreg_sha) / (elec_num * (elec_num - 1))
                )
                sig_rg = dist.ppf(1 - 0.05 / (2 * elec_num))
                norm = TwoSlopeNorm(
                    vmin=-0.0001,
                    vcenter=sig_rg if np.sum(siginreg_sha) > 0 else elec_num / 2,
                    vmax=elec_num - 1,
                )
                plot_brain(
                    siginreg_sha,
                    "AAL 90 regions - Shadow",
                    cut_position,
                    ax[1],
                    div_palette,
                    norm=norm,
                )

            else:
                fig, ax = plt.subplots(
                    1, 3, gridspec_kw={"width_ratios": [4, 4, 0.5]}, figsize=(8, 8)
                )
                sc = ax[2].scatter(
                    [np.nan],
                    [np.nan],
                    c=0,
                    cmap=div_palette,
                    norm=TwoSlopeNorm(vmin=0, vcenter=elec_num / 2, vmax=elec_num - 1),
                )
                cbar = plt.colorbar(
                    sc,
                    cax=ax[2],
                    shrink=0.35,
                    ticks=[0, elec_num / 2, elec_num - 1],
                    location="right",
                    format=FixedFormatter(
                        [
                            "no\nnon-linear\nconnections",
                            "significantly\nabove random\ngraph",
                            "complete\nnon-linear\nconnections",
                        ]
                    ),
                )

                dist = binom(
                    elec_num - 1, np.sum(siginreg) / (elec_num * (elec_num - 1))
                )
                sig_rg = dist.ppf(1 - 0.05 / (2 * elec_num))
                norm = TwoSlopeNorm(
                    vmin=-0.0001,
                    vcenter=sig_rg if np.sum(siginreg) > 0 else elec_num / 2,
                    vmax=elec_num - 1,
                )
                plot_cap(
                    siginreg,
                    "True",
                    ax[0],
                    div_palette,
                    norm=norm,
                )

                dist = binom(
                    elec_num - 1, np.sum(siginreg_sha) / (elec_num * (elec_num - 1))
                )
                sig_rg = dist.ppf(1 - 0.05 / (2 * elec_num))
                norm = TwoSlopeNorm(
                    vmin=-0.0001,
                    vcenter=sig_rg if np.sum(siginreg_sha) > 0 else elec_num / 2,
                    vmax=elec_num - 1,
                )
                plot_cap(
                    siginreg_sha,
                    "Shadow",
                    ax[1],
                    div_palette,
                    norm=norm,
                )
                ax[0].axis("off")
                ax[1].axis("off")
                plt.suptitle(subset_description + subset_na)
            plt.savefig(f"{output_prefix}_brain.pdf", bbox_inches="tight")
            plt.show()
