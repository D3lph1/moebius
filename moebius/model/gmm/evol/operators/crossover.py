from random import choice, randint

from ..graph.graph import GeneratorModel


def custom_crossover_exchange_mean(graph1: GeneratorModel, graph2: GeneratorModel, **kwargs):
    node1 = choice(graph1.nodes)
    node2 = choice(graph2.nodes)
    mean1 = node1.content['mean']
    mean2 = node2.content['mean']
    node1.content['mean'] = mean2
    node2.content['mean'] = mean1

    return graph1, graph2


def custom_crossover_exchange_mean_i(graph1: GeneratorModel, graph2: GeneratorModel, **kwargs):
    node1 = choice(graph1.nodes)
    node2 = choice(graph2.nodes)
    mean1 = node1.content['mean']
    mean2 = node2.content['mean']
    n = len(node1.content['w'])
    random_index = randint(0, n)
    new_mean1 = []
    new_mean2 = []
    for i, m in enumerate(mean1):
        if i == random_index:
            new_mean1.append(mean2[i])
            new_mean2.append(mean1[i])
        else:
            new_mean1.append(m)
            new_mean2.append(mean2[i])
    node1.content['mean'] = new_mean1
    node2.content['mean'] = new_mean2
    return graph1, graph2


def custom_crossover_exchange_var_i(graph1: GeneratorModel, graph2: GeneratorModel, **kwargs):
    node1 = choice(graph1.nodes)
    node2 = choice(graph2.nodes)
    var1 = node1.content['var']
    var2 = node2.content['var']
    n = len(node1.content['w'])
    random_index = randint(0, n)
    new_var1 = []
    new_var2 = []
    for i, m in enumerate(var1):
        if i == random_index:
            new_var1.append(var2[i])
            new_var2.append(var1[i])
        else:
            new_var1.append(m)
            new_var2.append(var2[i])
    node1.content['var'] = new_var1
    node2.content['var'] = new_var2
    return graph1, graph2


def custom_crossover_exchange_var(graph1: GeneratorModel, graph2: GeneratorModel, **kwargs):
    node1 = choice(graph1.nodes)
    node2 = choice(graph2.nodes)
    var1 = node1.content['var']
    var2 = node2.content['var']
    node1.content['var'] = var2
    node2.content['var'] = var1
    return graph1, graph2
