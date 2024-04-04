import abc
import numpy as np
from random import randint

class WeightsInitializer(abc.ABC):
    @abc.abstractmethod
    def create_initial_weights(self, node_name: str, n_comp: int) -> list[float]:
        """
         Abstract method to create initial weights for a given number of components.

         Parameters:
             n_comp (int): Number of components.

         Returns:
             list[float]: List of initial weights.
         """
        pass


class AvgWeightsInitializer(WeightsInitializer):
    def create_initial_weights(self, node_name: str, n_comp: int) -> list[float]:
        """
        Creates initial weights by dividing 1 equally among all components.
        """
        return [1 / n_comp] * n_comp

class RandomWeightsInitializer(WeightsInitializer):
    def create_initial_weights(self, node_name: str, n_comp: int) -> list[float]:
        """
        Creates initial weights by generating random values and normalizing them.
        """
        vals = np.random.random(size=n_comp)
        return vals / sum(vals)

class DirichletWeightsInitializer(WeightsInitializer):
    def __init__(self, multiplier: float = 1):
        self.__multiplier = multiplier

    def create_initial_weights(self, node_name: str, n_comp: int) -> list[float]:
        """
        Creates initial weights using Dirichlet distribution.
        """
        return np.random.dirichlet(np.ones(10) * self.__multiplier, size=1)[0]

class MeansInitializer(abc.ABC):
    @abc.abstractmethod
    def create_initial_means(self, node_name: str, n_comp: int) -> list[list[float]]:
        """
        Abstract method to create initial means for components.

        Args:
            n_comp (int): Number of components.

        Returns:
            list[list[float]]: Initial means for each component.
        """
        pass

class RandomMeansInitializer(MeansInitializer):
    def __init__(self, min_val: float, max_val: float):
        if min_val > max_val:
            raise ValueError("min_val must be less or equals than max_val")

        self.__min_val = min_val
        self.__max_val = max_val

    def create_initial_means(self, node_name: str, n_comp: int) -> list[list[float]]:
        return [[randint(self.__min_val, self.__max_val)] for _ in range(n_comp)]

class ConstantMeansInitializer(MeansInitializer):
    __means: dict

    def __init__(self, means: dict):
        self.__means = means

    def create_initial_means(self, node_name: str, n_comp: int) -> list[list[float]]:
        return self.__means[node_name]


class CovariancesInitializer(abc.ABC):
    @abc.abstractmethod
    def create_initial_covariances(self, node_name: str, n_comp: int) -> list[list[list[float]]]:
        """
        Abstract method to create initial covariances for components.

        Args:
            n_comp (int): Number of components.

        Returns:
            list[list[float]]: Initial means for each component.
        """
        pass

class RandomCovariancesInitializer(CovariancesInitializer):
    def __init__(self, min_val: float, max_val: float):
        if min_val > max_val:
            raise ValueError("min_val must be less or equals than max_val")

        self.__min_val = min_val
        self.__max_val = max_val

    def create_initial_covariances(self, node_name: str, n_comp: int) -> list[list[list[float]]]:
        return [[[randint(self.__min_val, self.__max_val)]] for _ in range(n_comp)]

class ConstantCovariancesInitializer(CovariancesInitializer):
    __covs: dict

    def __init__(self, covs: dict):
        self.__covs = covs

    def create_initial_covariances(self, node_name: str, n_comp: int) -> list[list[list[float]]]:
        return self.__covs[node_name]

class GMMParametersInitializer:
    def __init__(
            self,
            weights_initializer: WeightsInitializer,
            means_initializer: MeansInitializer,
            covariances_initializer: CovariancesInitializer
    ):
        self.weights_initializer = weights_initializer
        self.means_initializer = means_initializer
        self.covariances_initializer = covariances_initializer
