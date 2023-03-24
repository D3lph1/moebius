import numpy as np
from scipy.stats import multivariate_normal


def pdf_gmm(x, weights, means, covs):
    p = 0
    for i in range(len(weights)):
        p += weights[i] * multivariate_normal.pdf(x, mean=means[i], cov=covs[i], allow_singular=True)
    return p

def olr(w, means, covs):
    n_comp = len(w)
    olr_values = []

    for i in range(n_comp):
        for j in range(i + 1, n_comp, 1):
            delta = (np.array(means[j]) - np.array(means[i])) * 1 / 1000
            points = [np.array(means[i]) - 10 * delta]
            current_point = np.array(means[i]) - 10 * delta

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
                pdf_prev_k = pdf_gmm(points[k - 1], w_new, m_new, cov_new)
                pdf_next_k = pdf_gmm(points[k + 1], w_new, m_new, cov_new)
                if ((pdf_k - pdf_prev_k) > 0) & ((pdf_k - pdf_next_k) > 0):
                    peaks.append(pdf_k)
                if ((pdf_k - pdf_prev_k) < 0) & ((pdf_k - pdf_next_k) < 0):
                    saddles.append(pdf_k)

            olr_current = 0
            if len(peaks) == 1:
                olr_current = 1
            else:
                if len(saddles) == 0:
                    return 1
                else:
                    olr_current = saddles[0] / np.min(peaks)
            olr_values.append(olr_current)

    return olr_values
