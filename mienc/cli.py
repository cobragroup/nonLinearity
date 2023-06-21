#!/usr/bin/env python3

import argparse
from nonlinearestimator import NonLinearEstimator

parser = argparse.ArgumentParser(
    prog="MIENC", description="Mutual Information Estimation for Non-linear Contribution.", epilog="COBRA, Giulio Tani Raffaelli, 2023")
parser.add_argument('--version', action='version', version='%(prog)s 1.0')
parser.add_argument('-c', type=str, metavar='config', dest="configFile",
                    help="Path to an alternate config file.")
parser.add_argument('-m', type=str, metavar='cache', dest="cacheDir",
                    help="Path to a cache folder for maps.")
parser.add_argument('-d', type=str, metavar='dataset', dest="dataset",
                    help="Name of the dataset to be processed, overrides the corresponding option in the config file.")
parser.add_argument('-b', type=int, metavar='num of bins', dest="bins",
                    help="Number of bins for estimation, overrides the corresponding option in the config file.")
parser.add_argument('-s', type=int, metavar='num of surrogates', dest="Nsurrogates",
                    help="Number of surrogates for estimation, overrides the corresponding option in the config file.")
parser.add_argument('-r', type=str, metavar='num of regions', dest="regions", default="",
                    help="Number of regions in the atlas, useful for some fileName formats defined in the ini file as filename%%(num of regions)s.mat.")
parser.add_argument('-w', type=int, metavar='num of processes', dest="workers",
                    help="Number of workers for parallel estimation, overrides the corresponding option in the config file.")
parser.add_argument('-x', type=str, metavar='suffix', dest="suffix", default="",
                    help="Suffix to output folder name.")
parser.add_argument('-t', type=int, metavar='input len', dest="truncate",
                    help="Number of samples to consider.")
parser.add_argument('-S', dest="savenpy", action="store_true",
                    help="If to save the results per region and surrogate. Be careful, may take a lot of disk.")
parser.add_argument('-N', dest="retrieve", action="store_false",
                    help="If to avoid retrieving precomputed correction data. Be careful, computing correction may take much longer than computation itself.")
parser.add_argument('-J', dest="jitter", action="store_true",
                    help="If to add a tiny amount of jitter to data.")
parser.add_argument('-O', dest="ortho", action="store_true",
                    help="If to orthogonalise the input.")


if __name__ == "__main__":
    args = parser.parse_args()

    estimator = NonLinearEstimator(
        configFile=args.configFile,
        nbins=args.bins,
        Nsurrogates=args.Nsurrogates,
        cache=args.cacheDir,
        savenpy=args.savenpy,
        suffix=args.suffix,
        retrieve=args.retrieve,
        jitter=args.jitter,
        ortho=args.ortho,
        dataset=args.dataset,
        regions=args.regions,
        truncateInput=args.truncate,
        workers=args.workers
        )
    estimator.run()