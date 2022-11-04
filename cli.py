#!/usr/bin/env python3

import argparse
from nonlinearestimator import NonLinearEstimator

parser = argparse.ArgumentParser(
    prog="MIENLIC", description="Mutual Information Estimation for Non-Linear Information Contribution.", epilog="COBRA, Giulio Tani Raffaelli, 2022")
parser.add_argument('--version', action='version', version='%(prog)s 0.1')
parser.add_argument('-c', type=str, metavar='config', dest="configFile",
                        help="Path to an alternate config file.")
parser.add_argument('-d', type=str, metavar='dataset', dest="dataset",
                        help="Name of the dataset to be processed, overrides the corresponding option in the config file.")
parser.add_argument('-b', type=str, metavar='num of bins', dest="bins",
                        help="Number of bins for estimation, overrides the corresponding option in the config file.")



if __name__ == "__main__":
    args = parser.parse_args()

    estimator = NonLinearEstimator(args.configFile, args.dataset, args.bins)
    estimator.run()
