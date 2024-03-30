import abc
from typing import Optional, Union, List
from golem.core.dag.graph_delegate import GraphDelegate
from golem.core.dag.linked_graph_node import LinkedGraphNode
import networkx as nx

from .initializer import GMMParametersInitializer


class GeneratorModel(GraphDelegate):
    def __init__(self, nodes: Optional[Union[LinkedGraphNode, List[LinkedGraphNode]]] = None):
        super().__init__(nodes)
        self.unique_pipeline_id = 1

    def find_node(self, name: str) -> Union[LinkedGraphNode, None]:
        for node in self.nodes:
            if node.content[GeneratorNode.CONTENT_NAME] == name:
                return node

        return None


class GeneratorNode(LinkedGraphNode):
    CONTENT_NAME = "name"
    CONTENT_WEIGHTS = "w"
    CONTENT_MEANS = "mean"
    CONTENT_COVARIANCES = "var"

    def __str__(self):
        return self.content[GeneratorNode.CONTENT_NAME]


def create_generator_model_graph(n_dim: int, n_comp: int, initializer: GMMParametersInitializer) -> GeneratorModel:
    vertices = []
    for i in range(n_dim):
        vertices.append("Comp_" + str(i))

    return GeneratorModel(
        nodes=[
            GeneratorNode(
                nodes_from=[],
                content={
                    GeneratorNode.CONTENT_NAME: vertex,
                    GeneratorNode.CONTENT_WEIGHTS: initializer.weights_initializer.create_initial_weights(n_comp),
                    GeneratorNode.CONTENT_MEANS: initializer.means_initializer.create_initial_means(n_comp),
                    GeneratorNode.CONTENT_COVARIANCES: initializer.covariances_initializer.create_initial_covariances(
                        n_comp)
                })
            for vertex in vertices]
    )


class GraphShuffler(abc.ABC):
    @abc.abstractmethod
    def shuffle(self, graph: GeneratorModel) -> GeneratorModel:
        pass


class ProbabilisticGraphShuffler(GraphShuffler):
    PROBABILITY_EPSILON = 0.01

    def __init__(self, prob_of_edges: float):
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
            structure_parents['Comp_' + str(v)] = ['Comp_' + str(i) for i in dag.pred[v].keys()]

        for node in graph.nodes:
            parents_names = structure_parents[node.content[GeneratorNode.CONTENT_NAME]]
            for name_p in parents_names:
                for node_p in graph.nodes:
                    if node_p.content[GeneratorNode.CONTENT_NAME] == name_p:
                        node.nodes_from.append(node_p)
                        break

        return graph
