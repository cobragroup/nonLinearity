from .support import binning_pair_mutual_information, binning_total_mutual_information
from typing import Union, Tuple
from numpy.typing import ArrayLike
import numpy as np


class GenericEstimator:
    name = ""

    def __init__(self):
        self._EC = False
        self._parameter = 0

    def single_iter(self, data: Tuple[ArrayLike, ArrayLike, int]):
        """
        Single iteration of the Gaussian MI calculation for a given mean, covariance, number of samples and number of bins.
        Useful to estimate bias.

        Parameters
        ----------
        data : tuple of (array_like, array_like, int, int)
            Tuple containing the mean, covariance, number of samples and number of bins.

        Returns
        -------
        mi : float
            The calculated mutual information.

        """
        points = np.random.multivariate_normal(*data).T.copy()

        return self.pair_mutual_information(points[0], points[1])

    def total_mutual_information(self, data: np.ndarray) -> np.ndarray:
        pass

    def pair_mutual_information(self, x: ArrayLike, y: ArrayLike):
        pass

    def infer_parameter(self, duration: int):
        pass

    def get_suffix(self) -> str:
        return ""

    @property
    def EC(self):
        return self._EC

    @EC.setter
    def EC(self, value):
        self._EC = value

    @property
    def parameter(self):
        return self._parameter

    @parameter.setter
    def parameter(self, value):
        self._parameter = value


class BinEstimator(GenericEstimator):
    name = "bin"

    def pair_mutual_information(self, x: ArrayLike, y: ArrayLike):
        return binning_pair_mutual_information(x, y, self.parameter)

    def total_mutual_information(self, data: np.ndarray):
        return binning_total_mutual_information(data, self.parameter)

    def get_suffix(self) -> str:
        return f"_{self.parameter}bins"

    def infer_parameter(self, duration: int):
        if self._parameter == 0:
            self._parameter = int(np.power(duration, 1 / 3))
        elif self._parameter < 1:
            self._parameter *= duration

    @property
    def EC(self):
        return False

    @EC.setter
    def EC(self, value):
        pass


def get_estimator(
    estimator_name: str, effective_connectivity: bool
) -> GenericEstimator:
    if estimator_name.lower() in ["bin", "binning", "binning_estimator"]:
        return BinEstimator()
    elif estimator_name.lower() == ["knn", "knn_estimator"]:
        return GenericEstimator()
    elif estimator_name.lower() == [
        "chatterje",
        "chatterje_estimator",
        "chatterje_distance",
    ]:
        return GenericEstimator()
    else:
        raise ValueError("Invalid estimator name")
