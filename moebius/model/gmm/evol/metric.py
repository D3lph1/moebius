import pandas as pd
from gmr import GMM
from sklearn.preprocessing import KBinsDiscretizer
from bamt.networks.discrete_bn import DiscreteBN
from bamt.networks.continuous_bn import ContinuousBN

from moebius import olr
from .graph.graph import GeneratorModel
from ....util import make_diagonal

INFINITELY_LARGE_OLR = 100

def metric_olr_avg_discrete(graph: GeneratorModel):
    sample_data = pd.DataFrame()
    sample_data.index = [i for i in range(5000)]
    structure = []
    info = {'types': {}, 'signs': {}}
    n_components = None

    for node in graph.nodes:
        info['types'][node.content['name']] = 'disc'
        info['signs'][node.content['name']] = 'neg'
        w = node.content['w']
        mean = node.content['mean']
        var = node.content['var']
        n_components = len(w)
        gmm = GMM(n_components=n_components, priors=w, means=mean, covariances=var)
        sample_data[node.content['name']] = gmm.sample(5000)
        for parent in node.nodes_from:
            structure.append((parent.content['name'], node.content['name']))

    discretizer = KBinsDiscretizer(n_bins=128, encode='ordinal', strategy='quantile')
    discretizer.fit(sample_data)
    disc_data = discretizer.transform(sample_data)
    disc_data = pd.DataFrame(columns=sample_data.columns, data=disc_data, dtype=int)

    bn = DiscreteBN()
    bn.add_nodes(info)
    bn.set_structure(edges=structure)
    bn.fit_parameters(disc_data)
    data = bn.sample(5000)
    data = discretizer.inverse_transform(data)

    gmm = GMM(n_components=n_components)
    gmm.from_samples(data)

    w = gmm.priors.tolist()
    means = gmm.means.tolist()
    cov = make_diagonal(gmm.covariances.tolist())

    try:
        oo = olr(w, means, cov)

        return oo
    except Exception as e:
        print("ERROR", e)

        return [INFINITELY_LARGE_OLR]

def metric_olr_avg_continues(graph: GeneratorModel):
    sample_data = pd.DataFrame()
    sample_data.index = [i for i in range(5000)]
    structure = []
    info = {'types': {}, 'signs': {}}
    n_components = None

    for node in graph.nodes:
        info['types'][node.content['name']] = 'cont'
        info['signs'][node.content['name']] = 'neg'
        w = node.content['w']
        mean = node.content['mean']
        var = node.content['var']
        n_components = len(w)
        gmm = GMM(n_components=n_components, priors=w, means=mean, covariances=var)
        sample_data[node.content['name']] = gmm.sample(5000)
        for parent in node.nodes_from:
            structure.append((parent.content['name'], node.content['name']))

    discretizer = KBinsDiscretizer(n_bins=128, encode='ordinal', strategy='quantile')
    discretizer.fit(sample_data)
    disc_data = discretizer.transform(sample_data)
    disc_data = pd.DataFrame(columns=sample_data.columns, data=disc_data, dtype=int)
    bn = ContinuousBN(use_mixture=True)

    bn.add_nodes(info)
    data = bn.sample(5000)

    gmm = GMM(n_components=n_components)
    gmm.from_samples(data.to_numpy().astype('int'))

    w = gmm.priors.tolist()
    means = gmm.means.tolist()
    cov = make_diagonal(gmm.covariances.tolist())

    try:
        oo = olr(w, means, cov)

        return oo
    except Exception as e:
        print("ERROR", e)

        return [INFINITELY_LARGE_OLR]


def optimisation_metric(graph: GeneratorModel, target_olr, fn):
    try:
        o = fn(graph)
        if len(o) == 0:
            return 100

        d = (sum(o) / len(o) - target_olr) ** 2

        return d
    except BaseException as e:
        print('Error message')
        print(e)
