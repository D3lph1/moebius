import abc
from typing import Optional, Union, List
from golem.core.dag.graph_delegate import GraphDelegate
from golem.core.dag.linked_graph_node import LinkedGraphNode
import networkx as nx

from .initializer import GMMParametersInitializer


class GeneratorModel(GraphDelegate):
    """
    Represents a model for a generator.

    Inherits from GraphDelegate, which handles the graph structure.

    Attributes:
        unique_pipeline_id (int): An identifier for the generator model's pipeline.
    """

    def __init__(self, nodes: Optional[Union[LinkedGraphNode, List[LinkedGraphNode]]] = None):
        """
        Initializes the GeneratorModel.

        Args:
            nodes (Optional[Union[LinkedGraphNode, List[LinkedGraphNode]]]): Optional initial nodes for the model.
        """

        super().__init__(nodes)
        self.unique_pipeline_id = 1

    def find_node(self, name: str) -> Union[LinkedGraphNode, None]:
        """
        Finds a node in the generator model by its name.

        Args:
            name (str): The name of the node to find.

        Returns:
            Union[LinkedGraphNode, None]: The found node if it exists, otherwise None.
        """
        for node in self.nodes:
            if node.content[GeneratorNode.CONTENT_NAME] == name:
                return node

        return None


class GeneratorNode(LinkedGraphNode):
    """
    Represents a node in the generator model.

    Inherits from LinkedGraphNode, which handles the connectivity of the nodes.

    Attributes:
        NAME_PREFIX (str): Prefix for the default name of generator nodes.
        CONTENT_NAME (str): Key for the name of the node in the content dictionary.
        CONTENT_WEIGHTS (str): Key for the weights of the node in the content dictionary.
        CONTENT_MEANS (str): Key for the means of the node in the content dictionary.
        CONTENT_COVARIANCES (str): Key for the covariances of the node in the content dictionary.
    """

    NAME_PREFIX = 'Comp_'

    CONTENT_NAME = "name"
    CONTENT_WEIGHTS = "w"
    CONTENT_MEANS = "mean"
    CONTENT_COVARIANCES = "var"

    def __str__(self):
        return self.content[GeneratorNode.CONTENT_NAME]

def generator_node_name(i: int) -> str:
    """
    Generates a node name for a generator based on the given index.

    Args:
        i (int): The index of the generator node.

    Returns:
        str: The generated node name.
    """
    return GeneratorNode.NAME_PREFIX + str(i)

def extract_index_from_node_name(name: str) -> int:
    """
    Extracts the index of a generator node from its name.

    Args:
        name (str): The name of the generator node.

    Returns:
        int: The index extracted from the node name.
    """
    return int(name.split(GeneratorNode.NAME_PREFIX )[1])

def create_generator_model_graph(n_dim: int, n_comp: int, initializer: GMMParametersInitializer) -> GeneratorModel:
    """
    Creates a graph-based model for a generator.

    Args:
        n_dim (int): The dimensionality of the generator model.
        n_comp (int): The number of components in the Gaussian Mixture Model (GMM).
        initializer (GMMParametersInitializer): An initializer for GMM parameters.

    Returns:
        GeneratorModel: The created generator model.
    """
    vertices = []
    # Generate node names for each dimension
    for i in range(n_dim):
        vertices.append(generator_node_name(i))

    return GeneratorModel(
        nodes=[
            GeneratorNode(
                nodes_from=[], # No incoming connections initially
                content={
                    GeneratorNode.CONTENT_NAME: vertex,
                    GeneratorNode.CONTENT_WEIGHTS: initializer.weights_initializer.create_initial_weights(
                        vertex, # Node name
                        n_comp
                    ),
                    GeneratorNode.CONTENT_MEANS: initializer.means_initializer.create_initial_means(
                        vertex, # Node name
                        n_comp
                    ),
                    GeneratorNode.CONTENT_COVARIANCES: initializer.covariances_initializer.create_initial_covariances(
                        vertex, # Node name
                        n_comp
                    )
                })
            for vertex in vertices]
    )


class GraphShuffler(abc.ABC):
    """
    An interface for graph shuffling algorithms.
    """
    @abc.abstractmethod
    def shuffle(self, graph: GeneratorModel) -> GeneratorModel:
        """
        Shuffle the given graph.

        Args:
            graph (GeneratorModel): The graph to be shuffled.

        Returns:
            GeneratorModel: The shuffled graph.
        """
        pass


class ProbabilisticGraphShuffler(GraphShuffler):
    """
    An implementation of graph shuffler that probabilistically reshuffles the graph structure.
    """

    PROBABILITY_EPSILON = 0.01

    def __init__(self, prob_of_edges: float):
        """
        Initializes the ProbabilisticGraphShuffler with the given probability of edges.

        Args:
            prob_of_edges (float): The probability of an edge existing between two nodes.
                                   Must be in the range [0, 1].
        Raises:
            ValueError: If prob_of_edges is less than 0.0.
        """
        if prob_of_edges < 0.0:
            raise ValueError("prob_of_edges must be greater or equals than zero")

        self.__prob_of_edges = prob_of_edges

    def shuffle(self, graph: GeneratorModel) -> GeneratorModel:
        if self.__prob_of_edges < ProbabilisticGraphShuffler.PROBABILITY_EPSILON:
            return graph

        n_dim = len(graph.nodes)

        is_all = True
        dag = []
        while is_all:
            G = nx.gnp_random_graph(n_dim, self.__prob_of_edges, directed=True)
            dag = nx.DiGraph([(u, v) for (u, v) in G.edges() if u < v])
            if len(dag.nodes) == n_dim:
                is_all = False

        structure_parents = {}
        for v in dag:
            structure_parents[generator_node_name(v)] = [generator_node_name(i) for i in dag.pred[v].keys()]

        for node in graph.nodes:
            parents_names = structure_parents[node.content[GeneratorNode.CONTENT_NAME]]
            for name_p in parents_names:
                for node_p in graph.nodes:
                    if node_p.content[GeneratorNode.CONTENT_NAME] == name_p:
                        node.nodes_from.append(node_p)
                        break

        return graph
