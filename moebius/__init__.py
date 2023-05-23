import numpy as np

from .moebius import *

__doc__ = moebius.__doc__
if hasattr(moebius, "__all__"):
    __all__ = moebius.__all__

def parameters_from_flat(components: int, dims: int, data: list[float]):
    weights = [data[i] for i in range(components)]

    means_start = components
    means_end = means_start + (components * dims)

    covs_start = means_end
    covs_end = covs_start + (components * dims ** 2)

    if len(data) < covs_end:
        raise IndexError()

    means = np.array(data[means_start: means_end]).reshape(components, dims)

    covs = np.array(data[covs_start: covs_end]).reshape(components, dims, dims)

    return weights, means, covs

def olr_from_flat(components: int, dims: int, data: list[float]) -> list[float]:
    weights, means, covs = parameters_from_flat(components, dims, data)

    return olr(weights, means, covs)
