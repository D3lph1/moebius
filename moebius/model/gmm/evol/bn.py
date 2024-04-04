import pandas as pd
from .graph.graph import GeneratorModel

from bamt.networks.continuous_bn import ContinuousBN
from gmr import GMM


def build_bn(graph: GeneratorModel):
    sample_data = pd.DataFrame()
    sample_data.index = [i for i in range(5000)]
    structure = []
    info = {'types': {}, 'signs': {}}
    for node in graph.nodes:
        info['types'][node.content['name']] = 'cont'
        info['signs'][node.content['name']] = 'neg'
        w = node.content['w']
        mean = node.content['mean']
        var = node.content['var']
        gmm = GMM(n_components=len(w), priors=w, means=mean, covariances=var)
        sample_data[node.content['name']] = gmm.sample(5000)
        for parent in node.nodes_from:
            structure.append((parent.content['name'], node.content['name']))

    bn = ContinuousBN(use_mixture=True)
    bn.add_nodes(info)
    bn.set_structure(edges=structure)
    bn.fit_parameters(sample_data)

    return bn
