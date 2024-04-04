import abc
from random import choice, randrange, sample

from typing import Tuple

from ..graph.graph import GeneratorModel, GeneratorNode


class NodesSelector(abc.ABC):
    @abc.abstractmethod
    def select(self, graph1: GeneratorModel, graph2: GeneratorModel) -> Tuple[GeneratorNode, GeneratorNode]:
        pass


class RandomNodesSelector(NodesSelector):
    def select(self, graph1: GeneratorModel, graph2: GeneratorModel) -> Tuple[GeneratorNode, GeneratorNode]:
        return choice(graph1.nodes), choice(graph2.nodes)


# class RandomNonDuplicatedNodesSelector(NodesSelector):
#     def select(self, graph1: GeneratorModel, graph2: GeneratorModel) -> Tuple[GeneratorNode, GeneratorNode]:
#         index1, index2 = sample(range(0, len(graph.nodes)), 2)
#
#         return graph.nodes[index1], graph.nodes[index2]


class NodesCrossover(abc.ABC):
    @abc.abstractmethod
    def crossover(self, node1: GeneratorNode, node2: GeneratorNode):
        pass


class ExchangeFieldNodesCrossover(NodesCrossover):
    __field: str

    def __init__(self, field: str):
        self.__field = field

    def crossover(self, node1: GeneratorNode, node2: GeneratorNode):
        val1 = node1.content[self.__field]
        val2 = node2.content[self.__field]
        node1.content[self.__field] = val2
        node2.content[self.__field] = val1


class RandomIndexExchangeFieldNodesCrossover(NodesCrossover):
    __field: str

    def __init__(self, field: str):
        self.__field = field

    def crossover(self, node1: GeneratorNode, node2: GeneratorNode):
        val1 = node1.content[self.__field]
        val2 = node2.content[self.__field]

        idx = randrange(0, len(val1))
        val1_idx = val1[idx]
        val2_idx = val2[idx]
        val1[idx] = val2_idx
        val2[idx] = val1_idx


class Crossover:
    __nodes_selector: NodesSelector
    __crossover: NodesCrossover

    def __init__(self, nodes_selector: NodesSelector, crossover: NodesCrossover):
        self.__nodes_selector = nodes_selector
        self.__crossover = crossover

    def apply(self, graph1: GeneratorModel, graph2: GeneratorModel, **kwargs) -> GeneratorModel:
        node1, node2 = self.__nodes_selector.select(graph1, graph2)
        self.__crossover.crossover(node1, node2)

        return graph1, graph2

def crossover(crossover: NodesCrossover) -> Crossover:
    return Crossover(RandomNodesSelector(), crossover)
