import os, sys, configparser, json
from tqdm import tqdm
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.ticker import FixedFormatter
import seaborn as sns
import numpy as np
import pandas as pd

from scipy.stats import ks_1samp, uniform, binom
from scipy.spatial import Voronoi

from nilearn import datasets, plotting, image
import nibabel as nib
import mne

settings_parser = configparser.ConfigParser()
settings_parser.read("localsettings.ini")
MAIN_DATA_FOLDER = settings_parser.get("global", "data_path")
MIENC_PATH = settings_parser.get("global", "mienc_path")
cache_dir = os.path.join(MAIN_DATA_FOLDER, "cache")
config_ini = os.path.join(MAIN_DATA_FOLDER, "config.ini")

sys.path.append(os.path.abspath(MIENC_PATH))
from mienc import Corrector

bad_electrodes = ["T7", "T8", "Cz", "F7", "CP6", "PO10", "Fp2"]
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

with open(os.path.join(DATA_DIR_EEG, "good_electrodes.json")) as fp:
    electrode_names = json.load(fp)
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
        x, y = [self.vmin, self.siglo, self.siglo, self.sighi, self.sighi, self.vmax], [
            0,
            1 / 3,
            0.43,
            0.57,
            2 / 3,
            1.0,
        ]
        return np.ma.masked_array(np.interp(value, x, y, left=-np.inf, right=np.inf))

    def inverse(self, value):
        y, x = [self.vmin, self.siglo, self.siglo, self.sighi, self.sighi, self.vmax], [
            0,
            1 / 3,
            0.43,
            0.57,
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
        plt.title(title, fontsize="xx-large")


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
    FIGURES_FOLDER,
    cut_position=(-15, -75, 27),
):
    normalistions = {subset_id: {} for subset_id in subset_identifiers}
    values = {subset_id: {} for subset_id in subset_identifiers}
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
        thresh = HolmThresholdFromP(np.concatenate([ks_p, ks_p_sha]), 0.01)
        print(thresh)
        corrected = np.full(pair_num, np.nan)
        corrected[ks_p < thresh] = ks_stat[ks_p < thresh]
        sig_pair = np.zeros([elec_num, elec_num])
        sig_pair[np.triu_indices(elec_num, 1)] = corrected
        sig_pair += sig_pair.T
        np.fill_diagonal(sig_pair, np.nan)
        siginreg = np.nansum(sig_pair, 1)  # np.sum(sig_pair > 0, 1)
        print("Non linear connections:", np.sum(siginreg) / 2)

        # thresh_sha = HolmThresholdFromP(ks_p_sha)

        corrected_sha = np.full(pair_num, np.nan)
        corrected_sha[ks_p_sha < thresh] = ks_stat_sha[ks_p_sha < thresh]
        sig_pair_sha = np.zeros([elec_num, elec_num])
        sig_pair_sha[np.triu_indices(elec_num, 1)] = corrected_sha
        sig_pair_sha += sig_pair_sha.T
        np.fill_diagonal(sig_pair_sha, np.nan)
        siginreg_sha = np.nansum(sig_pair_sha, 1)  # np.sum(sig_pair_sha > 0, 1)
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
        plt.title("Empiric")

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
        plt.savefig(
            os.path.join(
                FIGURES_FOLDER, f"localisation_{output_prefix}_{subset_id}_matrix.pdf"
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
                        "no\nnon-linear\nconnections",
                        "significantly\nbelow random\ngraph",
                        "significantly\nabove random\ngraph",
                        "complete\nnon-linear\nconnections",
                    ]
                ),
            )
            samples = []
            print(
                np.tril_indices(elec_num, -1)[0].shape,
                np.triu(sig_pair_sha, 1).shape,
            )
            for i in tqdm(range(1000), desc="Null model"):
                tmp = np.full_like(sig_pair, np.nan)
                tmp[np.triu_indices(elec_num, 1)] = np.random.permutation(
                    sig_pair[np.triu_indices(elec_num, 1)]
                )
                tmp[np.tril_indices(elec_num, -1)] = tmp.T[
                    np.tril_indices(elec_num, -1)
                ]
                samples.append(np.nansum(tmp, 1))
            low_rg, sig_rg = np.quantile(np.concatenate(samples), [0.025, 0.975])
            print(f"{sig_rg=} {low_rg=}")
            norm = SignificantNormalize(
                low_rg if np.sum(siginreg) > 0 else elec_num / 3,
                sig_rg if np.sum(siginreg) > 0 else elec_num * 2 / 3,
                -0.0001,
                (
                    (
                        np.nanmax(siginreg)
                        if np.nanmax(siginreg) > sig_rg
                        else sig_rg * 1.1
                    )
                    if np.nansum(siginreg) > 0
                    else elec_num - 1
                ),
            )
            normalistions[subset_id]["Empiric"] = norm
            values[subset_id]["Empiric"] = siginreg
            samples = []
            for i in tqdm(range(1000), desc="Null model"):
                tmp = np.full_like(sig_pair_sha, np.nan)
                tmp[np.triu_indices(elec_num, 1)] = np.random.permutation(
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
                low_rg_sha if np.sum(siginreg_sha) > 0 else elec_num / 3,
                sig_rg_sha if np.sum(siginreg_sha) > 0 else elec_num * 2 / 3,
                -0.0001,
                (
                    (
                        np.nanmax(siginreg_sha)
                        if np.nanmax(siginreg_sha) > sig_rg
                        else sig_rg * 1.1
                    )
                    if np.nansum(siginreg_sha) > 0
                    else elec_num - 1
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
                    FIGURES_FOLDER, f"localisation_{output_prefix}_{subset_id}_map.pdf"
                ),
                bbox_inches="tight",
            )
            plt.show()

    fig = plt.figure(figsize=(15, 10))
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
            siglo=(elec_num - 1) / 3,
            sighi=(elec_num - 1) * 2 / 3,
            vmax=elec_num - 1,
        ),
    )
    cbar = plt.colorbar(
        sc,
        cax=ax_c,
        shrink=0.35,
        ticks=[0, (elec_num - 1) / 3, (elec_num - 1) * 2 / 3, elec_num - 1],
        location="bottom" if "aal" in results_file else "right",
        format=FixedFormatter(
            [
                "no\nnon-linear\nconnections",
                "significantly\nbelow random\ngraph",
                "significantly\nabove random\ngraph",
                "complete\nnon-linear\nconnections",
            ]
        ),
    )
    for ha, tick in zip(
        ["left", "center", "center", "right"], ax_c.yaxis.get_majorticklabels()
    ):
        tick.set_horizontalalignment(ha)
        tick.set_verticalalignment("top")
        tick.set_rotation(90)
        tick.set_rotation_mode("anchor")
    if "aal" in results_file:
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
        ax[0, 0].set_title("Empiric", pad=-50)
        ax[0, 1].set_title("Shadow", pad=-50)
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

        ax[0, 0].text(
            0,
            0.5,
            "Empiric",
            horizontalalignment="right",
            verticalalignment="center",
            rotation="vertical",
            transform=ax[0, 0].transAxes,
            fontsize="xx-large",
        )
        ax[1, 0].text(
            0,
            0.5,
            "Shadow",
            horizontalalignment="right",
            verticalalignment="center",
            rotation="vertical",
            transform=ax[1, 0].transAxes,
            fontsize="xx-large",
        )

    plt.savefig(
        os.path.join(FIGURES_FOLDER, f"localisation_{output_prefix}_summary.pdf"),
        bbox_inches="tight",
    )
    plt.show()
