import numpy as np
from scipy import stats

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

def pdf_gmm(x, weights, means, covs):
    p = 0
    for i in range(len(weights)):
        p += weights[i] * stats.multivariate_normal.pdf(x, mean=means[i], cov=covs[i], allow_singular=True)
    return p

def olr_non_native(w, means, covs):
    n_comp = len(w)
    olr_values = []
    for i in range(n_comp):
        for j in range(i+1, n_comp, 1):
            delta = (np.array(means[j]) - np.array(means[i])) * 1/1000
            points = [np.array(means[i]) - 10*delta]
            current_point = np.array(means[i]) - 10*delta
            for k in range(1030):
                new_point = current_point + delta
                current_point = new_point
                points.append(new_point)
            w1 = w[i]
            w2 = w[j]
            w1_new = w1 / (w1 + w2)
            w2_new = 1 - w1_new
            w_new = [w1_new, w2_new]
            m_new = [means[i], means[j]]
            cov_new = [covs[i], covs[j]]
            peaks = []
            saddles = []
            for k in range(1, 1030, 1):
                pdf_k = pdf_gmm(points[k], w_new, m_new, cov_new)
                pdf_prev_k = pdf_gmm(points[k-1], w_new, m_new, cov_new)
                pdf_next_k = pdf_gmm(points[k+1], w_new, m_new, cov_new)
                if ((pdf_k - pdf_prev_k) > 0) & ((pdf_k - pdf_next_k) > 0):
                    peaks.append(pdf_k)
                if (((pdf_k - pdf_prev_k) < 0) & ((pdf_k - pdf_next_k) < 0)) | (((pdf_k - pdf_prev_k) == 0) & ((pdf_k - pdf_next_k) == 0)):
                    saddles.append(pdf_k)

            if len(peaks) == 1:
                olr_current = 1
            else:
                olr_current = saddles[0] / np.min(peaks)
            olr_values.append(olr_current)
    return np.mean(olr_values)


def olr_universal(w, mean, cov):
    try:
        for cov_i in cov:
            np.linalg.cholesky(cov_i)

        return olr(w, mean, cov)[0]
    except np.linalg.LinAlgError:
        return olr_non_native(w, mean, cov)
