import abc
from random import choice, randrange, sample

from typing import Tuple

from ..graph.graph import GeneratorModel, GeneratorNode


class NodesSelector(abc.ABC):
    """
    Interface for selecting nodes from two generator models for crossover.

    Provides an interface for selecting nodes to be crossed over.
    """

    @abc.abstractmethod
    def select(self, graph1: GeneratorModel, graph2: GeneratorModel) -> Tuple[GeneratorNode, GeneratorNode]:
        """
        Selects two nodes from the given generator models for crossover.

        Args:
            graph1 (GeneratorModel): The first generator model.
            graph2 (GeneratorModel): The second generator model.

        Returns:
            Tuple[GeneratorNode, GeneratorNode]: A tuple containing the selected nodes from each graph.
        """
        pass


class RandomNodesSelector(NodesSelector):
    """
    Randomly selects nodes from two generator models for crossover.

    Inherits from NodesSelector.
    """

    def select(self, graph1: GeneratorModel, graph2: GeneratorModel) -> Tuple[GeneratorNode, GeneratorNode]:
        return choice(graph1.nodes), choice(graph2.nodes)


# class RandomNonDuplicatedNodesSelector(NodesSelector):
#     def select(self, graph1: GeneratorModel, graph2: GeneratorModel) -> Tuple[GeneratorNode, GeneratorNode]:
#         index1, index2 = sample(range(0, len(graph.nodes)), 2)
#
#         return graph.nodes[index1], graph.nodes[index2]


class NodesCrossover(abc.ABC):
    """
    Interface for crossover operations between nodes.

    Provides an interface for performing crossover between two nodes.
    """

    @abc.abstractmethod
    def crossover(self, node1: GeneratorNode, node2: GeneratorNode):
        pass


class ExchangeFieldNodesCrossover(NodesCrossover):
    """
    Crossover operation that exchanges a specified field between two nodes.

    Inherits from NodesCrossover.

    Attributes:
        __field (str): The field to exchange between the nodes.
    """

    __field: str

    def __init__(self, field: str):
        self.__field = field

    def crossover(self, node1: GeneratorNode, node2: GeneratorNode):
        val1 = node1.content[self.__field]
        val2 = node2.content[self.__field]
        node1.content[self.__field] = val2
        node2.content[self.__field] = val1


class RandomIndexExchangeFieldNodesCrossover(NodesCrossover):
    """
    Crossover operation that exchanges values at random indices of a specified field between two nodes.

    Inherits from NodesCrossover.

    Attributes:
        __field (str): The field to exchange between the nodes.
    """

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
    """
    Class to perform crossover between two generator models.

    Attributes:
        __nodes_selector (NodesSelector): Selector for choosing nodes for crossover.
        __crossover (NodesCrossover): Crossover operation to perform.
    """

    __nodes_selector: NodesSelector
    __crossover: NodesCrossover

    def __init__(self, nodes_selector: NodesSelector, crossover: NodesCrossover):
        self.__nodes_selector = nodes_selector
        self.__crossover = crossover

    def apply(self, graph1: GeneratorModel, graph2: GeneratorModel, **kwargs) -> GeneratorModel:
        """
        Applies crossover between two generator models.

        Args:
            graph1 (GeneratorModel): The first generator model.
            graph2 (GeneratorModel): The second generator model.
            **kwargs: Additional arguments for crossover operation.

        Returns:
            GeneratorModel: The modified generator models after crossover.
        """

        node1, node2 = self.__nodes_selector.select(graph1, graph2)
        self.__crossover.crossover(node1, node2)

        return graph1, graph2

def crossover(crossover: NodesCrossover) -> Crossover:
    """
    Factory function to create a Crossover instance with a specified crossover operation.

    Args:
        crossover (NodesCrossover): The crossover operation to perform.

    Returns:
        Crossover: A Crossover instance with the specified crossover operation.
    """

    return Crossover(RandomNodesSelector(), crossover)
