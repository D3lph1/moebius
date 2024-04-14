import abc

import numpy as np
from random import choice, randrange, uniform

from typing import Tuple

from ..graph.graph import GeneratorModel, GeneratorNode


class NodeSelector(abc.ABC):
    """
    Interface for selecting a node from a generator model for mutation operation.

    Provides an interface for selecting a node from a generator model.
    """
    @abc.abstractmethod
    def select(self, graph: GeneratorModel) -> GeneratorNode:
        """
        Selects a node from the generator model for mutation operation.

        Args:
            graph (GeneratorModel): The generator model from which to select the node.

        Returns:
            GeneratorNode: The selected node.
        """
        pass


class RandomNodeSelector(NodeSelector):
    """
    Selects a node randomly from a generator model.

    Inherits from NodeSelector.
    """
    def select(self, graph: GeneratorModel) -> GeneratorNode:
        return choice(graph.nodes)


class ByNameNodeSelector(NodeSelector):
    """
    Selects a node from a generator model by its name.

    Inherits from NodeSelector.

    Attributes:
        __name (str): The name of the node to select.
    """

    __name: str

    def __init__(self, name: str):
        self.__name = name

    def select(self, graph: GeneratorModel) -> GeneratorNode:
        for node in graph.nodes:
            if node.name == self.__name:
                return node

        raise ValueError(f'Node with name "{self.__name}" not found')


class NodeMutator(abc.ABC):
    """
    Interface for mutating a generator node. It defines mutation operator.

    Provides an interface for mutating a generator node.
    """

    @abc.abstractmethod
    def mutate(self, node: GeneratorNode):
        """
        Mutates the given generator node.

        Args:
            node (GeneratorNode): The generator node to mutate.
        """
        pass


class RandomRangeNodeMutator(NodeMutator):
    """
    Abstract base class for mutating a generator node within a random range.

    Inherits from NodeMutator.

    Attributes:
        _field (str): The field to mutate.
        _range_axes (list): The range axes for mutation.
    """

    _field: str
    _range_axes: list

    def __init__(self, field: str, range_axes: list):
        """
        Initializes the RandomRangeNodeMutator.

        Args:
            field (str): The field to mutate.
            range_axes (list): The range axes for mutation.

        Raises:
            ValueError: If the shape of range_axes is invalid.
        """

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
        """
        Terminal condition for mutating a value within the specified range.

        Args:
            val (float): The value to mutate.
            range_axes (Tuple[float, float]): The range axes for mutation.

        Returns:
            float: The mutated value.
        """
        pass

    @staticmethod
    def __is_range(arr: Tuple[float, float]):
        if not isinstance(arr[0], int) and not isinstance(arr[0], float):
            return False

        return isinstance(arr[1], int) or isinstance(arr[1], float)


class RandomDeltaNodeMutator(RandomRangeNodeMutator):
    """
    Mutates a generator node by adding a random delta within a specified range to each value.

    Inherits from RandomRangeNodeMutator.

    Attributes:
        _field (str): The field to mutate.
        _range_axes (list): The range axes for mutation.
    """

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
    """
    Mutates a generator node by assigning random values within a specified range to each field.

    Inherits from RandomRangeNodeMutator.

    Attributes:
        _field (str): The field to mutate.
        _range_axes (list): The range axes for mutation.
    """

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
    """
    Mutates a random index of a generator node by adding a random delta within a specified range.

    Inherits from RandomRangeNodeMutator.

    Attributes:
        _field (str): The field to mutate.
        _range_axes (list): The range axes for mutation.
    """

    def mutate(self, node: GeneratorNode):
        rnd_idx = randrange(0, len(self._range_axes))
        node.content[self._field][rnd_idx] = self._deep(node.content[self._field][rnd_idx], self._range_axes[rnd_idx])

    def _terminal_condition(self, val: float, range_axes: Tuple[float, float]):
        if range_axes[0] == range_axes[1]:
            return val
        else:
            return val + uniform(range_axes[0], range_axes[1])


class ClapNodeDecoratedMutator(RandomRangeNodeMutator):
    """
    Mutates a generator node by applying a decorator mutator and then clamping the values within a specified range.

    Inherits from RandomRangeNodeMutator.

    Attributes:
        _field (str): The field to mutate.
        _range_axes (list): The range axes for mutation.
        __decorated (NodeMutator): The decorated mutator to apply.
    """

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
    """
    Mutates a generator node by assigning random values that sum to 1 to each field.

    Inherits from NodeMutator.

    Attributes:
        __field (str): The field to mutate.
    """

    __field: str

    def __init__(self, field: str):
        self.__field = field

    def mutate(self, node: GeneratorNode):
        values = np.random.random(len(node.content[self.__field]))
        values /= np.sum(values)

        node.content[self.__field] = values.tolist()


class DirichletValuesSumTo1NodeMutator(NodeMutator):
    """
    Mutates a generator node by assigning values from a Dirichlet distribution that sum to 1 to each field.

    Inherits from NodeMutator.

    Attributes:
        __field (str): The field to mutate.
        __multiplier (float): The multiplier for the Dirichlet distribution.
    """

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
    """
    Data Transfer Object for mutation operation to be applied to a generator model.

    Attributes:
        __node_selector (NodeSelector): Selector for choosing a node for mutation.
        __mutator (NodeMutator): Mutator for mutating the chosen node.
    """

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
    """
    Factory function to create a Mutation instance with a specified mutator.

    Args:
        mutator (NodeMutator): The mutator to use for mutation.

    Returns:
        Mutation: A Mutation instance with the specified mutator.
    """

    return Mutation(RandomNodeSelector(), mutator)
