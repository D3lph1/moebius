import numpy as np
from random import choice, randint, uniform

from ..graph.graph import GeneratorModel


def custom_mutation_change_mean(graph: GeneratorModel, **kwargs):
    try:
        node = choice(graph.nodes)
        node.content['mean'] = [[randint(-1000, 1000)] for _ in range(len(node.content['w']))]
    except Exception as ex:
        graph.log.warn(f'Incorrect mutation: {ex}')
    return graph


def custom_mutation_change_mean_step(graph: GeneratorModel, **kwargs):
    try:
        node = choice(graph.nodes)

        means = []
        for i in range(len(node.content['w'])):
            mean = node.content['mean'][i][0]
            means.append([mean + uniform(-5, 5)])

        node.content['mean'] = means

    except Exception as ex:
        print(ex)

    return graph


def custom_mutation_change_mean_i(graph: GeneratorModel, **kwargs):
    try:
        node = choice(graph.nodes)
        n = len(node.content['w'])
        random_index = randint(0, n)
        mean = node.content['mean']
        new_mean = []
        for i, m in enumerate(mean):
            if i == random_index:
                new_mean.append([randint(-1000, 1000)])
            else:
                new_mean.append(m)
        node.content['mean'] = new_mean
    except Exception as ex:
        graph.log.warn(f'Incorrect mutation: {ex}')
    return graph


def custom_mutation_change_var(graph: GeneratorModel, **kwargs):
    try:
        node = choice(graph.nodes)
        node.content['var'] = [[[randint(1, 50)]] for _ in range(len(node.content['w']))]
    except Exception as ex:
        graph.log.warn(f'Incorrect var: {ex}')
    return graph


def custom_mutation_change_var_i(graph: GeneratorModel, **kwargs):
    try:
        node = choice(graph.nodes)
        n = len(node.content['w'])
        random_index = randint(0, n)
        var = node.content['var']
        new_var = []
        for i, m in enumerate(var):
            if i == random_index:
                new_var.append([[randint(1, 50)]])
            else:
                new_var.append(m)
        node.content['var'] = new_var
    except Exception as ex:
        graph.log.warn(f'Incorrect mutation: {ex}')
    return graph


def custom_mutation_change_w(graph: GeneratorModel, **kwargs):
    node = choice(graph.nodes)
    w = node.content['w']

    array = np.random.random(len(w))
    array /= np.sum(array)

    for i, node in enumerate(graph.nodes):
        node.content['w'] = array

    return graph
