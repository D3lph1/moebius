import abc

import numpy as np
from random import choice, randrange, uniform

from typing import Tuple

from ..graph.graph import GeneratorModel, GeneratorNode
from .operator import Operator


class NodeSelector(abc.ABC):
    @abc.abstractmethod
    def select(self, graph: GeneratorModel) -> GeneratorNode:
        pass


class RandomNodeSelector(NodeSelector):
    def select(self, graph: GeneratorModel) -> GeneratorNode:
        return choice(graph.nodes)


class ByNameNodeSelector(NodeSelector):
    __name: str

    def __init__(self, name: str):
        self.__name = name

    def select(self, graph: GeneratorModel) -> GeneratorNode:
        for node in graph.nodes:
            if node.name == self.__name:
                return node

        raise ValueError(f'Node with name "{self.__name}" not found')


class NodeMutator(abc.ABC):
    @abc.abstractmethod
    def mutate(self, node: GeneratorNode):
        pass


class RandomRangeNodeMutator(NodeMutator):
    _field: str
    _range_axes: list

    def __init__(self, field: str, range_axes: list):
        shape = np.array(range_axes).shape
        if len(shape) < 2 or shape[-1] != 2:
            raise ValueError('Invalid range_axes shape. It must be (..., 2), but given: {}.'.format(shape))

        self._field = field
        self._range_axes = range_axes

    def _deep(self, arr, range_axes):
        if len(range_axes) == 2 and RandomRangeNodeMutator.__is_range(range_axes):
            return [self._terminal_condition(arr[0], range_axes)]

        return [self._deep(each, range_axes[i]) for i, each in enumerate(arr)]

    @abc.abstractmethod
    def _terminal_condition(self, val: float, range_axes: Tuple[float, float]):
        pass

    @staticmethod
    def __is_range(arr: Tuple[float, float]):
        if not isinstance(arr[0], int) and not isinstance(arr[0], float):
            return False

        return isinstance(arr[1], int) or isinstance(arr[1], float)


class RandomDeltaNodeMutator(RandomRangeNodeMutator):
    def _terminal_condition(self, val: float, range_axes: Tuple[float, float]):
        if range_axes[0] == range_axes[1]:
            return val
        else:
            return val + uniform(range_axes[0], range_axes[1])

    def mutate(self, node: GeneratorNode):
        node.content[self._field] = [
            self._deep(node.content[self._field][i], delta_axes)
            for i, delta_axes in enumerate(self._range_axes)
        ]


class RandomValueNodeMutator(RandomRangeNodeMutator):
    def _terminal_condition(self, val: float, range_axes: Tuple[float, float]):
        if range_axes[0] == range_axes[1]:
            return val
        else:
            return uniform(range_axes[0], range_axes[1])

    def mutate(self, node: GeneratorNode):
        node.content[self._field] = [
            self._deep(node.content[self._field][i], range_axes)
            for i, range_axes in enumerate(self._range_axes)
        ]


class RandomIndexRandomDeltaNodeMutator(RandomRangeNodeMutator):
    def mutate(self, node: GeneratorNode):
        rnd_idx = randrange(0, len(self._range_axes))
        node.content[self._field][rnd_idx] = self._deep(node.content[self._field][rnd_idx], self._range_axes[rnd_idx])

    def _terminal_condition(self, val: float, range_axes: Tuple[float, float]):
        if range_axes[0] == range_axes[1]:
            return val
        else:
            return val + uniform(range_axes[0], range_axes[1])


class ClapNodeDecoratedMutator(RandomRangeNodeMutator):
    __decorated: NodeMutator

    def __init__(self, field: str, decorated: NodeMutator, range_axes: list):
        super().__init__(field, range_axes)
        self.__decorated = decorated

    def mutate(self, node: GeneratorNode):
        self.__decorated.mutate(node)

        node.content[self._field] = [
            self._deep(node.content[self._field][i], range_axes)
            for i, range_axes in enumerate(self._range_axes)
        ]

    def _terminal_condition(self, val: float, range_axes: Tuple[float, float]):
        if val < range_axes[0] or val > range_axes[1]:
            return uniform(range_axes[0], range_axes[1])
        else:
            return val


class RandomValuesSumTo1NodeMutator(NodeMutator):
    __field: str

    def __init__(self, field: str):
        self.__field = field

    def mutate(self, node: GeneratorNode):
        values = np.random.random(len(node.content[self.__field]))
        values /= np.sum(values)

        node.content[self.__field] = values.tolist()


class DirichletValuesSumTo1NodeMutator(NodeMutator):
    __field: str
    __multiplier: float

    def __init__(self, field: str, multiplier: float = 1):
        self.__field = field
        self.__multiplier = multiplier

    def mutate(self, node: GeneratorNode):
        node.content[self.__field] = np.random.dirichlet(
            np.ones(len(node.content[self.__field])) * self.__multiplier,
            size=1
        )[0].tolist()


class Mutation:
    __node_selector: NodeSelector
    __mutator: NodeMutator

    def __init__(self, node_selector: NodeSelector, mutator: NodeMutator):
        self.__node_selector = node_selector
        self.__mutator = mutator

    def apply(self, graph: GeneratorModel, **kwargs) -> GeneratorModel:
        node = self.__node_selector.select(graph)
        self.__mutator.mutate(node)

        return graph

def mutation(mutator: NodeMutator) -> Mutation:
    return Mutation(RandomNodeSelector(), mutator)
