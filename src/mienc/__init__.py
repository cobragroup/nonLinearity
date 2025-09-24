#!/usr/bin/env python3
import argparse

from .corrector import Corrector
from .nonlinearestimator import NonLinearEstimator
from ._version import __version__


def main():
    parser = argparse.ArgumentParser(
        prog="MIENC",
        description="Mutual Information Estimation for Non-linear Contribution.",
        epilog="COBRA, Giulio Tani Raffaelli, 2023",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0")
    parser.add_argument(
        "-i",
        type=str,
        metavar="config.ini",
        dest="config_file",
        help="Path to an alternate config.ini file.",
    )
    parser.add_argument(
        "-c",
        type=str,
        metavar="cache",
        dest="cache_dir",
        help="Path to a cache folder for maps.",
    )
    parser.add_argument(
        "-d",
        type=str,
        metavar="dataset",
        dest="dataset",
        help="Name of the dataset to be processed, overrides the corresponding option in the config file.",
    )
    parser.add_argument(
        "-b",
        type=int,
        metavar="num of bins",
        dest="bins",
        help="Number of bins for estimation, overrides the corresponding option in the config file.",
    )
    parser.add_argument(
        "-s",
        type=int,
        metavar="num of surrogates",
        dest="surrogates",
        help="Number of surrogates for estimation, overrides the corresponding option in the config file.",
    )
    parser.add_argument(
        "-r",
        type=str,
        metavar="num of regions",
        dest="dataset_sub",
        default="",
        help="Number of regions in the atlas, useful for some fileName formats defined in the ini file as filename%%(num of regions)s.mat.",
    )
    parser.add_argument(
        "-w",
        type=int,
        metavar="num of processes",
        dest="workers",
        help="Number of workers for parallel estimation, overrides the corresponding option in the config file.",
    )
    parser.add_argument(
        "-x",
        type=str,
        metavar="suffix",
        dest="suffix",
        default="",
        help="Suffix to output folder name.",
    )
    parser.add_argument(
        "-t",
        type=int,
        metavar="input len",
        dest="truncate",
        help="Number of samples to consider.",
    )
    parser.add_argument(
        "-S",
        dest="save_out",
        action="store_true",
        help="If to save the results per region and surrogate. Be careful, may take a lot of disk.",
    )
    parser.add_argument(
        "-N",
        dest="retrieve",
        action="store_false",
        help="If to avoid retrieving precomputed correction data. Be careful, computing correction may take much longer than computation itself.",
    )
    parser.add_argument(
        "-J",
        dest="jitter",
        action="store_true",
        help="If to add a tiny amount of jitter to data.",
    )
    parser.add_argument(
        "-O", dest="ortho", action="store_true", help="If to orthogonalise the input."
    )
    parser.add_argument(
        "-W",
        dest="shadow",
        action="store_true",
        help="If to compute values for the shadow dataset.",
    )
    parser.add_argument(
        "-F",
        dest="full_stats",
        action="store_true",
        help="If to compute extended stats for the MI.",
    )

    args = parser.parse_args()

    estimator = NonLinearEstimator(
        config_file=args.config_file,
        bins=args.bins,
        surrogates=args.surrogates,
        cache=args.cache_dir,
        save_out=args.save_out,
        suffix=args.suffix,
        retrieve=args.retrieve,
        jitter=args.jitter,
        ortho=args.ortho,
        dataset=args.dataset,
        dataset_sub=args.dataset_sub,
        truncate_input=args.truncate,
        workers=args.workers,
        verbose=True,
    )

    estimator.estimate(extended_stats=args.full_stats, compute_shadow=args.shadow)
