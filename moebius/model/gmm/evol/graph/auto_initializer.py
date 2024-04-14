import numpy as np
from gmr import GMM

from .initializer import MeansInitializer
from .graph import extract_index_from_node_name

class AutoMeansInitializer(MeansInitializer):
    data_means: list[list[float]]

    def __init__(self, data_means: list[list[float]]):
        self.data_means = data_means

    @staticmethod
    def from_data(data: list[list[float]], n_comps: int) -> 'AutoMeansInitializer':
        gmm = GMM(n_components=n_comps)
        gmm.from_samples(np.array(data))

        return AutoMeansInitializer(gmm.means.tolist())

    def create_initial_means(self, node_name: str, n_comp: int) -> list[list[float]]:
        idx = extract_index_from_node_name(node_name)

        means = []
        for i in range(n_comp):
            means.append([self.data_means[i][idx]])

        return means
