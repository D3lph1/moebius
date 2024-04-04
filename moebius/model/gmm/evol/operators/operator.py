import abc
from ..graph.graph import GeneratorModel


class Operator(abc.ABC):
    @abc.abstractmethod
    def apply(self, graph: GeneratorModel, **kwargs) -> GeneratorModel:
        pass
