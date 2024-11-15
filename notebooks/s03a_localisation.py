import os, sys, configparser, json, re
from tqdm import tqdm
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.ticker import FixedFormatter
from matplotlib.transforms import ScaledTranslation
import seaborn as sns
import numpy as np
import pandas as pd

from scipy.stats import norm as norm_dist, pearsonr
from scipy.spatial import Voronoi

from nilearn import datasets, plotting, image
import nibabel as nib
import mne

sns.set_theme("paper", "ticks")

settings_parser = configparser.ConfigParser()
settings_parser.read("localsettings.ini")
MAIN_DATA_FOLDER = settings_parser.get("global", "data_path")
MIENC_PATH = settings_parser.get("global", "mienc_path")
cache_dir = os.path.join(MAIN_DATA_FOLDER, "cache")
config_ini = os.path.join(MAIN_DATA_FOLDER, "config.ini")

sys.path.append(os.path.abspath(MIENC_PATH))
from mienc import Corrector

bad_electrodes = ["T7", "T8", "Cz", "F7", "CP6", "PO10", "Fp2"]
null_model_samples = 10000
DATA_DIR_fMRI = os.path.join(MAIN_DATA_FOLDER, "fMRI_region_size")
DATA_DIR_EEG = os.path.join(MAIN_DATA_FOLDER, "EEG")
RANDOM_SEED = settings_parser.getint("global", "random_seed")
random_state = np.random.default_rng(RANDOM_SEED)
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

with open(os.path.join(DATA_DIR_EEG, "good_electrodes.json")) as fp:
    electrode_names = json.load(fp)
decades = {
    "Fp": 0,
    "AF": 10,
    "F": 20,
    "FC": 30,
    "FT": 30,
    "C": 40,
    "T": 40,
    "CP": 50,
    "TP": 50,
    "P": 60,
    "PO": 70,
    "O": 80,
}
units = {"9": 0, "7": 1, "5": 2, "3": 3, "1": 4, "z": 5, "2": 6, "4": 7, "6": 8, "8": 9}


def elec_val(elec_name: str):
    parts = re.match(r"([A-Z]+p?)(\d+|z)", elec_name)
    return decades[parts.groups()[0]] + units[parts.groups()[1]]


sort_vals = [elec_val(en) for en in electrode_names]
map_order = np.argsort(sort_vals, kind="stable")
montage = mne.channels.make_standard_montage("standard_1020")
electrode_positions = pd.DataFrame(montage.get_positions()["ch_pos"]).T.rename(
    columns={0: "x", 1: "y", 2: "z"}
)

source_z = electrode_positions.z.min()
diameter = electrode_positions.z.max() - source_z

div_palette = sns.color_palette("vlag", as_cmap=True)
div_palette.set_over(div_palette.get_bad())
div_palette.set_under(color=[0.5, 0.5, 0.5, 0.5])
div_palette.set_bad(color="grey")


class SignificantNormalize(Normalize):
    def __init__(self, siglo, sighi, vmin=None, vmax=None, clip=False):
        self.siglo = siglo
        self.sighi = sighi
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        # Note also that we must extrapolate beyond vmin/vmax
        x, y = [
            self.vmin,
            self.siglo,
            self.siglo,
            self.sighi,
            self.sighi + 0.00001,
            self.vmax,
        ], [
            0,
            1 / 3,
            5 / 12,
            7 / 12,
            2 / 3,
            1.0,
        ]
        return np.ma.masked_array(np.interp(value, x, y, left=-np.inf, right=np.inf))

    def inverse(self, value):
        y, x = [
            self.vmin,
            self.siglo,
            self.siglo + 0.00001,
            self.sighi,
            self.sighi,
            self.vmax,
        ], [
            0,
            1 / 3,
            5 / 12,
            7 / 12,
            2 / 3,
            1.0,
        ]
        return np.interp(value, x, y, left=-np.inf, right=np.inf)


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


# fmt: off
aal_order = np.argsort(
    [
        0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        7, 7, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 6,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 2, 2,
        5, 5, 5, 5, 5, 5,
    ], kind="stable"
)
# fmt: on
aal_names = [
    "Central",
    "Frontal",
    "Limbic",
    "Occipital",
    "Parietal",
    "Temporal",
    "Sub.GrayNuc.",
    "Insula",
]
print(aal_order)


def plot_cap(
    in_array,
    title=None,
    axes=None,
    cmap=div_palette,
    vmin=None,
    vmax=None,
    norm=None,
    plane_distance=0.1,
):
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

    displayed_electrodes = pd.DataFrame(
        np.concatenate([normalised, np.full(len(bad_electrodes), np.nan)]),
        index=electrode_names + bad_electrodes,
        columns=["activation"],
    ).join(electrode_positions, how="left")

    radius = (
        np.max(np.sqrt(displayed_electrodes.XP**2 + displayed_electrodes.YP**2)) * 1.15
    )
    vor = Voronoi(
        np.concatenate(
            [
                displayed_electrodes[["XP", "YP"]],
                np.transpose(
                    [
                        radius * np.sin(np.linspace(0, 2 * np.pi, 100)),
                        radius * np.cos(np.linspace(0, 2 * np.pi, 100)),
                    ]
                ),
            ]
        )
    )
    if axes is not None:
        plt.sca(axes)
    plt.gca().set_aspect("equal")

    for i, v in enumerate(displayed_electrodes.activation):
        region = vor.regions[vor.point_region[i]]
        vertices = np.array([vor.vertices[n] for n in region])
        plt.fill(vertices[:, 0], vertices[:, 1], color=cmap(v))
    if title is not None:
        plt.title(title, fontsize="x-large")


def HolmThresholdFromP(p_values: np.ndarray, threshold: float = 0.05):
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
    good = sorted_p < threshold / (sorted_p.size - np.arange(sorted_p.size))
    which = np.argmin(good)

    if which == 0 and sorted_p[-1] < threshold:
        return threshold

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
            folder_name="in_memory",  # os.path.dirname(results_file).format(subset_id, bins),
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
                    f"{output_prefix}_z_stat_{subset_id}.npy",
                )
            )
        ):
            all_values = np.zeros((subj_num, pair_num, 100))
            for s in tqdm(range(subj_num), "Subject"):
                all_values[s, :, :] = corrct.correct(
                    np.load(results_file.format(subset_id, bins, s))
                )
            pair_z = np.empty([pair_num])
            for p in range(pair_num):
                mean = np.mean(all_values[:, p, 1:])
                std = np.std(all_values[:, p, 1:].mean(0))
                pair_z[p] = (np.mean(all_values[:, p, 0]) - mean) / std  # **2
            np.save(
                os.path.join(
                    MAIN_DATA_FOLDER,
                    "Results",
                    "localised",
                    f"{output_prefix}_z_stat_{subset_id}.npy",
                ),
                pair_z,
            )


boundaries = np.array([0, 6, 14, 22, 28, 36, 45, 51, 54])
aal_boundaries = np.array([0, 6, 30, 42, 56, 68, 78, 88, 90])


def show_localised_non_linearity(
    results_file,
    subset_description,
    subset_identifiers,
    subset_names,
    output_prefix,
    pair_num,
    elec_num,
    FIGURES_FOLDER,
    cut_position=(-15, -75, 27),
):
    normalistions = {subset_id: {} for subset_id in subset_identifiers}
    values = {subset_id: {} for subset_id in subset_identifiers}
    for subset_id, subset_na in zip(subset_identifiers, subset_names):
        z_stat = np.load(
            os.path.join(
                MAIN_DATA_FOLDER,
                "Results",
                "localised",
                f"{output_prefix}_z_stat_{subset_id}.npy",
            )
        )
        z_p = norm_dist.sf(abs(z_stat))
        z_stat_sha = np.load(
            os.path.join(
                MAIN_DATA_FOLDER,
                "Results",
                "localised",
                f"{output_prefix+'_sha'}_z_stat_{subset_id}.npy",
            )
        )
        z_p_sha = norm_dist.sf(abs(z_stat_sha))
        thresh = HolmThresholdFromP(np.concatenate([z_p, z_p_sha]), 0.01)
        print(thresh)
        corrected = np.full(pair_num, np.nan)
        corrected[z_p < thresh] = z_stat[z_p < thresh]
        sig_pair = np.zeros([elec_num, elec_num])
        sig_pair[np.triu_indices(elec_num, 1)] = corrected
        sig_pair += sig_pair.T
        np.fill_diagonal(sig_pair, np.nan)
        siginreg = np.nansum(sig_pair, 1)  # np.sum(sig_pair > 0, 1)
        print("Non linear connections:", np.sum(np.isfinite(sig_pair)) / 2)

        # thresh_sha = HolmThresholdFromP(ks_p_sha)

        corrected_sha = np.full(pair_num, np.nan)
        corrected_sha[z_p_sha < thresh] = z_stat_sha[z_p_sha < thresh]
        sig_pair_sha = np.zeros([elec_num, elec_num])
        sig_pair_sha[np.triu_indices(elec_num, 1)] = corrected_sha
        sig_pair_sha += sig_pair_sha.T
        np.fill_diagonal(sig_pair_sha, np.nan)
        siginreg_sha = np.nansum(sig_pair_sha, 1)  # np.sum(sig_pair_sha > 0, 1)
        print("Non linear connections:", np.sum(np.isfinite(sig_pair_sha)) / 2)

        fix, ax = plt.subplots(
            1, 3, gridspec_kw={"width_ratios": [4, 4, 0.2]}, figsize=(12, 7)
        )
        vmax = max(np.nanmax(sig_pair), np.nanmax(sig_pair_sha))
        vmin = min(np.nanmin(sig_pair), np.nanmin(sig_pair_sha))
        plt.sca(ax[0])
        if "aal" in results_file:
            plt.imshow(
                sig_pair[:, aal_order][aal_order, :],
                interpolation="none",
                vmax=vmax,
                vmin=vmin,
            )
            # step = 10 if elec_num < 100 else 20
            # plt.yticks(np.arange(0, elec_num, step), np.arange(elec_num)[::step])
            # step = 20 if elec_num < 100 else 40
            # plt.xticks(
            #     np.arange(0, elec_num, step),
            #     np.arange(elec_num)[::step],
            # )
            plt.xlim(plt.xlim())
            plt.ylim(plt.ylim())
            plt.vlines(aal_boundaries[1:-1] - 0.5, *plt.ylim(), colors="g")
            plt.hlines(aal_boundaries[1:-1] - 0.5, *plt.xlim(), colors="g")
            plt.xticks(
                aal_boundaries[:-1] + np.diff(aal_boundaries) / 2 - 0.5,
                aal_names,
                rotation=30,
                ha="right",
                rotation_mode="anchor",
            )
            plt.yticks(
                aal_boundaries[:-1] + np.diff(aal_boundaries) / 2 - 0.5,
                aal_names,
            )
        else:
            plt.imshow(
                sig_pair[:, map_order][map_order, :],
                interpolation="none",
                vmax=vmax,
                vmin=vmin,
            )
            plt.xlim(plt.xlim())
            plt.ylim(plt.ylim())
            plt.vlines(boundaries[1:-1] - 0.5, *plt.ylim(), colors="g")
            plt.hlines(boundaries[1:-1] - 0.5, *plt.xlim(), colors="g")
            plt.xticks(
                boundaries[:-1] + np.diff(boundaries) / 2 - 0.5,
                ["AF", "F", "FC", "C", "CP", "P", "PO", "O"],
            )
            plt.yticks(
                boundaries[:-1] + np.diff(boundaries) / 2 - 0.5,
                ["AF", "F", "FC", "C", "CP", "P", "PO", "O"],
            )
        plt.xlabel(f"Region")
        plt.title("Empiric")

        plt.sca(ax[1])
        if "aal" in results_file:
            plt.imshow(
                sig_pair_sha[:, aal_order][aal_order, :],
                interpolation="none",
                vmax=vmax,
                vmin=vmin,
            )
            plt.xlim(plt.xlim())
            plt.ylim(plt.ylim())
            plt.vlines(aal_boundaries[1:-1] - 0.5, *plt.ylim(), colors="g")
            plt.hlines(aal_boundaries[1:-1] - 0.5, *plt.xlim(), colors="g")
            plt.xticks(
                aal_boundaries[:-1] + np.diff(aal_boundaries) / 2 - 0.5,
                aal_names,
                rotation=30,
                ha="right",
                rotation_mode="anchor",
            )
        else:
            plt.imshow(
                sig_pair_sha[:, map_order][map_order, :],
                interpolation="none",
                vmax=vmax,
                vmin=vmin,
            )
            plt.xlim(plt.xlim())
            plt.ylim(plt.ylim())
            plt.vlines(boundaries[1:-1] - 0.5, *plt.ylim(), colors="g")
            plt.hlines(boundaries[1:-1] - 0.5, *plt.xlim(), colors="g")
            plt.xticks(
                boundaries[:-1] + np.diff(boundaries) / 2 - 0.5,
                ["AF", "F", "FC", "C", "CP", "P", "PO", "O"],
            )
        plt.yticks([])
        # step = 20 if elec_num < 100 else 40
        # plt.xticks(
        #     np.arange(0, elec_num, step),
        #     np.arange(elec_num)[::step],
        # )
        plt.xlabel(f"Region")
        plt.title("Shadow")
        plt.colorbar(ax=ax[1], cax=ax[2], shrink=0.2).ax.set_ylabel(
            "Z statistcs", rotation=90
        )

        plt.suptitle(
            subset_description
            + subset_na
            + f" - {np.sum(z_p<thresh)} ({100*np.sum(z_p<thresh)/pair_num:.3} %) significant pairs"
        )
        plt.savefig(
            os.path.join(
                FIGURES_FOLDER, f"localisation_Z_{output_prefix}_{subset_id}_matrix.pdf"
            ),
            bbox_inches="tight",
        )
        plt.show()

        if (siginreg > 0).any():
            layout = (3, 1) if "aal" in results_file else (1, 3)
            spec = "height_ratios" if "aal" in results_file else "width_ratios"
            print("Siginreg max:", np.max(siginreg))
            fig, ax = plt.subplots(
                *layout, gridspec_kw={spec: [4, 4, 0.5]}, figsize=(8, 8)
            )
            sc = ax[2].scatter(
                [np.nan],
                [np.nan],
                c=0,
                cmap=div_palette,
                norm=SignificantNormalize(
                    vmin=0,
                    siglo=(elec_num - 1) / 3,
                    sighi=(elec_num - 1) * 2 / 3,
                    vmax=elec_num - 1,
                ),
            )
            cbar = plt.colorbar(
                sc,
                cax=ax[2],
                shrink=0.35,
                ticks=[0, (elec_num - 1) / 3, (elec_num - 1) * 2 / 3, elec_num - 1],
                location="bottom" if "aal" in results_file else "right",
                format=FixedFormatter(
                    [
                        "minimum\ndegree",
                        "significantly\nbelow random\ngraph",
                        "significantly\nabove random\ngraph",
                        "maximum\ndegree",
                    ]
                ),
            )
            samples = []
            print(
                np.tril_indices(elec_num, -1)[0].shape,
                np.triu(sig_pair_sha, 1).shape,
            )
            for i in tqdm(range(null_model_samples), desc="Null model"):
                tmp = np.full_like(sig_pair, np.nan)
                tmp[np.triu_indices(elec_num, 1)] = random_state.permutation(
                    sig_pair[np.triu_indices(elec_num, 1)]
                )
                tmp[np.tril_indices(elec_num, -1)] = tmp.T[
                    np.tril_indices(elec_num, -1)
                ]
                samples.append(np.nansum(tmp, 1))
            low_rg, sig_rg = np.quantile(np.concatenate(samples), [0.025, 0.975])
            print(f"{sig_rg=} {low_rg=}")
            norm = SignificantNormalize(
                low_rg if np.sum(siginreg) > 0 else 0.2,
                sig_rg if np.sum(siginreg) > 0 else 0.3,
                np.nanmin(siginreg[siginreg > 0]) if np.sum(siginreg) > 0 else 0.1,
                (
                    (
                        np.nanmax(siginreg)
                        if np.nanmax(siginreg) > sig_rg
                        else sig_rg * 1.1
                    )
                    if np.nansum(siginreg) > 0
                    else 0.4
                ),
            )
            normalistions[subset_id]["Empiric"] = norm
            values[subset_id]["Empiric"] = siginreg
            samples = []
            for i in tqdm(range(null_model_samples), desc="Null model"):
                tmp = np.full_like(sig_pair_sha, np.nan)
                tmp[np.triu_indices(elec_num, 1)] = random_state.permutation(
                    sig_pair_sha[np.triu_indices(elec_num, 1)]
                )
                tmp[np.tril_indices(elec_num, -1)] = tmp.T[
                    np.tril_indices(elec_num, -1)
                ]
                samples.append(np.nansum(tmp, 1))
            low_rg_sha, sig_rg_sha = np.quantile(
                np.concatenate(samples), [0.025, 0.975]
            )
            norm_sha = SignificantNormalize(
                low_rg_sha if np.sum(siginreg_sha) > 0 else 0.2,
                sig_rg_sha if np.sum(siginreg_sha) > 0 else 0.3,
                (
                    np.nanmin(siginreg_sha[siginreg_sha > 0])
                    if np.sum(siginreg_sha) > 0
                    else 0.1
                ),
                (
                    (
                        np.nanmax(siginreg_sha)
                        if np.nanmax(siginreg_sha) > sig_rg_sha
                        else sig_rg * 1.1
                    )
                    if np.nansum(siginreg_sha) > 0
                    else 0.4
                ),
            )
            normalistions[subset_id]["Shadow"] = norm_sha
            values[subset_id]["Shadow"] = siginreg_sha
            if "aal" in results_file:
                plot_brain(
                    siginreg,
                    "AAL 90 regions - Empiric",
                    cut_position,
                    ax[0],
                    div_palette,
                    norm=norm,
                )  # (-15,-75,27)

                plot_brain(
                    siginreg_sha,
                    "AAL 90 regions - Shadow",
                    cut_position,
                    ax[1],
                    div_palette,
                    norm=norm_sha,
                )

            else:
                plot_cap(
                    siginreg,
                    "Empiric",
                    ax[0],
                    div_palette,
                    norm=norm,
                )
                plot_cap(
                    siginreg_sha,
                    "Shadow",
                    ax[1],
                    div_palette,
                    norm=norm_sha,
                )
                ax[0].axis("off")
                ax[1].axis("off")
                plt.suptitle(subset_description + subset_na)
            plt.savefig(
                os.path.join(
                    FIGURES_FOLDER,
                    f"localisation_Z_{output_prefix}_{subset_id}_map.pdf",
                ),
                bbox_inches="tight",
            )
            plt.show()

    fig = plt.figure(figsize=(16 / 2.54, 6 / 2.54))
    if "aal" in results_file:
        shape = len(values), 3
    else:
        shape = 2, len(values) + 1

    gs = fig.add_gridspec(
        shape[0],
        shape[1],
        width_ratios=[
            5,
        ]
        * (shape[1] - 1)
        + [0.1 * shape[1]],
    )
    ax = np.zeros((shape[0], shape[1] - 1), dtype=object)
    for i in range(shape[0]):
        for j in range(shape[1] - 1):
            ax[i, j] = fig.add_subplot(gs[i, j])
    ax_c = fig.add_subplot(gs[:, -1])

    plt.subplots_adjust(wspace=0.05, hspace=-0.0)
    sc = ax_c.scatter(
        [np.nan],
        [np.nan],
        c=0,
        cmap=div_palette,
        norm=SignificantNormalize(
            vmin=0,
            siglo=1 / 3,
            sighi=1 * 2 / 3,
            vmax=1,
        ),
    )
    cbar = plt.colorbar(
        sc,
        cax=ax_c,
        shrink=0.35,
        ticks=[0, 1 / 3, 1 * 2 / 3, 1],
        location="right",
        format=FixedFormatter(
            [
                "minimum\ndegree",
                "significantly\nbelow random\ngraph",
                "significantly\nabove random\ngraph",
                "maximum\ndegree",
            ]
        ),
    )

    for ha, off, tick in zip(
        ["baseline", "top", "bottom", "top"],
        [0, 1 / 10, -1 / 10, 1 / 10],
        ax_c.yaxis.get_majorticklabels(),  # ["left", "center", "center", "right"]
    ):
        tick.set_horizontalalignment("left")
        tick.set_verticalalignment(ha)
        tick.set_rotation(0)
        tick.set_rotation_mode("anchor")
        tick.set_transform(
            tick.get_transform() + ScaledTranslation(0, off, fig.dpi_scale_trans)
        )
    if "aal" in results_file:
        labels = "ABC"
        for sn, (subset_id, subset_na) in enumerate(zip(values, subset_names)):
            plot_brain(
                values[subset_id]["Empiric"],
                subset_na,
                cut_position,
                ax[sn, 0],
                div_palette,
                norm=normalistions[subset_id]["Empiric"],
            )
            plot_brain(
                values[subset_id]["Shadow"],
                None,
                cut_position,
                ax[sn, 1],
                div_palette,
                norm=normalistions[subset_id]["Shadow"],
            )
            ax[sn, 0].text(
                0.01,
                0.99,
                f"{labels[sn]}",
                horizontalalignment="left",
                verticalalignment="top",
                fontweight="bold",
                transform=ax[sn, 0].transAxes,
                fontsize="xx-large",
            )
            print(
                subset_na,
                pearsonr(values[subset_id]["Empiric"], values[subset_id]["Shadow"]),
            )
        ax[0, 0].set_title("Empiric", pad=-50, fontsize="xx-large")
        ax[0, 1].set_title("Shadow", pad=-50, fontsize="xx-large")
    else:
        for sn, (subset_id, subset_na) in enumerate(zip(values, subset_names)):
            plot_cap(
                values[subset_id]["Empiric"],
                subset_na,
                ax[0, sn],
                div_palette,
                norm=normalistions[subset_id]["Empiric"],
            )
            plot_cap(
                values[subset_id]["Shadow"],
                None,
                ax[1, sn],
                div_palette,
                norm=normalistions[subset_id]["Shadow"],
            )
            ax[0, sn].axis("off")
            ax[1, sn].axis("off")
            print(
                subset_na,
                pearsonr(values[subset_id]["Empiric"], values[subset_id]["Shadow"]),
            )

        ax[0, 0].text(
            0,
            0.5,
            "Empiric",
            horizontalalignment="right",
            verticalalignment="center",
            rotation="vertical",
            transform=ax[0, 0].transAxes,
            fontsize="large",
        )
        ax[1, 0].text(
            0,
            0.5,
            "Shadow",
            horizontalalignment="right",
            verticalalignment="center",
            rotation="vertical",
            transform=ax[1, 0].transAxes,
            fontsize="large",
        )
        # ax[0, 0].text(
        #     0.01,
        #     1.015,
        #     f"D",
        #     horizontalalignment="left",
        #     verticalalignment="bottom",
        #     fontweight="bold",
        #     transform=ax[0, 0].transAxes,
        #     fontsize="xx-large",
        # )

    plt.savefig(
        os.path.join(FIGURES_FOLDER, f"localisation_Z_{output_prefix}_summary.pdf"),
        bbox_inches="tight",
    )
    plt.show()
