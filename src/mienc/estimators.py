from .support import (
    binning_pair_mutual_information,
    binning_total_mutual_information,
    pair_Chatterjee,
    total_Chatterjee,
)
from typing import Union, Tuple
from numpy.typing import ArrayLike
import numpy as np


class GenericEstimator:
    name = ""

    def __init__(self):
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
        return bool(self._EC)

    @EC.setter
    def EC(self, value: int):
        assert value >= 0
        self._EC = value

    @property
    def delay(self):
        return self._EC

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
        return f"{self.parameter}bins"

    def infer_parameter(self, duration: int):
        if self._parameter == 0:
            self._parameter = max(2, int(np.power(duration, 1 / 3)))
        elif self._parameter < 1:
            self._parameter = max(2, int(duration * self._parameter))

    @property
    def EC(self):
        return False

    @EC.setter
    def EC(self, value):
        pass

    @property
    def delay(self):
        return 0


class KNNEstimator(GenericEstimator):
    name = "knn"

    def __init__(self, effective_connectivity: int = 0):
        try:
            from tigramite.independence_tests.cmiknn import CMIknn

            self.CMIknn = CMIknn
            self.EC = effective_connectivity
            GenericEstimator.__init__(self)
        except ImportError as e:
            print(e)
            raise ImportError(
                "KNNEstimator requires tigramite to be installed\n"
                "install mienc with `pip install .[alt]` or\n"
                "`pip install tigramite`."
            )

    def pair_mutual_information(self, x: ArrayLike, y: ArrayLike):
        assert len(x) == len(y), "x and y must have the same length"
        estim = self.CMIknn(knn=self._parameter, workers=1)
        indices = np.array([0, 1])
        input = np.stack([x, y], axis=0)
        return estim.get_dependence_measure(input, indices)

    def total_mutual_information(self, data: np.ndarray):
        estim = self.CMIknn(knn=self._parameter, workers=1, transform="none")
        out = np.zeros([data.shape[1], data.shape[1]])
        indices = np.array([0, 1, 2]) if self.EC else np.array([0, 1])
        for i in range(data.shape[1]):
            for j in range(i + 1, data.shape[1]):
                if self._EC:
                    input = np.stack(
                        [
                            data[: -self.delay, i],
                            data[self.delay :, j],
                            data[: -self.delay, j],
                        ],
                        axis=0,
                    )
                    out[i, j] = estim.get_dependence_measure(input, indices)
                    input = np.stack(
                        [
                            data[: -self.delay, j],
                            data[self.delay :, i],
                            data[: -self.delay, i],
                        ],
                        axis=0,
                    )
                    out[j, i] = estim.get_dependence_measure(input, indices)
                else:
                    input = np.stack([data[:, i], data[:, j]], axis=0)
                    out[i, j] = estim.get_dependence_measure(input, indices)
                    out[j, i] = out[i, j]

        return out if self.EC else out[np.triu_indices(data.shape[1], 1)]

    def get_suffix(self) -> str:
        return f"{f'EC{self.delay}' if self.EC else ''}_{self.parameter}nn"

    def infer_parameter(self, duration: int):
        if self._parameter == 0:
            self._parameter = 1


class ChatterjeEstimator(GenericEstimator):
    name = "chatt"

    def __init__(self, effective_connectivity: int = 0):
        self.EC = effective_connectivity
        GenericEstimator.__init__(self)

    def pair_mutual_information(self, x: np.ndarray, y: np.ndarray):
        return max(pair_Chatterjee(x, y, False), pair_Chatterjee(y, x, False))

    def total_mutual_information(self, data: np.ndarray):
        res = total_Chatterjee(data, False)
        return (
            res
            if self.EC
            else np.max(np.stack([res, res.T], axis=-1), axis=-1)[
                np.triu_indices(data.shape[1], 1)
            ]
        )

    def get_suffix(self) -> str:
        return f"{f'EC' if self.EC else ''}_chatt"

    def infer_parameter(self, duration: int):
        pass

    @property
    def EC(self):
        return bool(self._EC)

    @EC.setter
    def EC(self, value: int):
        assert value in [0, 1], "EC must be 0 or 1 for Chatterje correlation estimator"
        self._EC = value


class dChatterjeEstimator(GenericEstimator):
    name = "dchatt"

    def __init__(self, effective_connectivity: int = 0):
        self.EC = effective_connectivity
        GenericEstimator.__init__(self)

    def pair_mutual_information(self, x: np.ndarray, y: np.ndarray):
        return max(pair_Chatterjee(x, y, True), pair_Chatterjee(y, x, True))

    def total_mutual_information(self, data: np.ndarray):
        input = np.stack(
            [data[: -self.delay, :], data[self.delay :, :]],
            axis=-1,
        )
        return total_Chatterjee(input, True)

    def get_suffix(self) -> str:
        return f"EC{self.delay}_dchatt"

    def infer_parameter(self, duration: int):
        pass

    @property
    def EC(self):
        return bool(self._EC)

    @EC.setter
    def EC(self, value: int):
        assert value >= 1, "EC must be >=1 for Chatterjee distance estimator"
        self._EC = value


def get_estimator(estimator_name: str, effective_connectivity: int) -> GenericEstimator:
    if estimator_name.lower() in ["bin", "binning", "binning_estimator"]:
        return BinEstimator()
    elif estimator_name.lower() in ["knn", "knn_estimator"]:
        return KNNEstimator(effective_connectivity)
    elif estimator_name.lower() in [
        "chatt",
        "chatterjee",
        "chatterjee_correlation",
    ]:
        return ChatterjeEstimator(effective_connectivity)
    elif estimator_name.lower() in [
        "dchatt",
        "dchatterjee",
        "chatterjee_distance",
    ]:
        return dChatterjeEstimator(effective_connectivity)
    else:
        raise ValueError(f"Invalid estimator name: {estimator_name}")
