import abc
from typing import TypeVar, Generic
import json

from bamt.networks.base import BaseNetwork
from bamt.networks.continuous_bn import ContinuousBN
from bamt.networks.discrete_bn import DiscreteBN
from bamt.nodes.mixture_gaussian_node import MixtureGaussianNode
from bamt.nodes.discrete_node import DiscreteNode
from bamt.nodes.base import BaseNode

# Type of Bayesian Network
T = TypeVar('T', bound=BaseNetwork)
# Type of Bayesian Network node
N = TypeVar('N', bound=BaseNode)


class BNSerializer(abc.ABC, Generic[T]):
    """
    Custom implementation of Bayesian Network serialization process. Descendants of the
    interface are used as a replacement for broken serialization methods in bamt library.
    The main purpose of the interface is to serialize/deserialize BN to/from dict.
    """

    @abc.abstractmethod
    def serialize(self, bn: T) -> dict:
        """
        Serialize a Bayesian Network object to a dictionary.

        Parameters:
            bn (T): The Bayesian Network object to be serialized.

        Returns:
            dict: A dictionary representing the serialized Bayesian Network.
        """
        pass

    @abc.abstractmethod
    def deserialize(self, serialized: dict) -> T:
        """
        Deserialize a dictionary into a Bayesian Network object.

        Parameters:
            serialized (dict): A dictionary containing serialized Bayesian Network data.

        Returns:
            T: A Bayesian Network object reconstructed from the serialized data.
        """
        pass


class AbstractBNSerializer(BNSerializer, Generic[N]):
    """
    Abstract base class for BayesianSerializers, that contains shared functionality for
    different types of Bayesian networks.
    """

    @abc.abstractmethod
    def _create_bn(self) -> T:
        """
        Create a new instance of the Bayesian Network object.

        Returns:
            T: A new instance of the Bayesian Network.
        """
        pass

    @abc.abstractmethod
    def _create_node(self, name: str) -> N:
        """
        Create a new instance of a Bayesian Network node.

        Parameters:
            name (str): The name of the node.

        Returns:
            N: A new instance of the Bayesian Network node.
        """
        pass

    def serialize(self, bn: T) -> dict:
        return {
            'sf_name': bn.sf_name,
            'type': bn.type,
            '_allowed_dtypes': bn._allowed_dtypes,
            'nodes': [
                {
                    'name': node.name,
                    'type': node.type,
                    'disc_parents': node.disc_parents,
                    'cont_parents': node.cont_parents,
                    'children': node.children
                } for node in bn.nodes
            ],
            'edges': bn.edges,
            'weights': bn.weights,
            'descriptor': bn.descriptor,
            'distributions': bn.distributions,
            'has_logit': bn.has_logit,
            'use_mixture': bn.use_mixture,
            'encoders': bn.encoders
        }

    def deserialize(self, serialized: dict) -> T:
        bn = self._create_bn()
        bn.sf_name = serialized['sf_name']
        bn.type = serialized['type']
        bn._allowed_dtypes = serialized['_allowed_dtypes']

        bn.nodes = []
        for node in serialized['nodes']:
            m_node = self._create_node(node['name'])
            m_node.type = node['type']
            m_node.disc_parents = node['disc_parents']
            m_node.cont_parents = node['cont_parents']
            m_node.children = node['children']
            bn.nodes.append(m_node)

        bn.edges = serialized['edges']
        bn.weights = serialized['weights']
        bn.descriptor = serialized['descriptor']
        bn.distributions = serialized['distributions']
        bn.has_logit = serialized['has_logit']
        bn.use_mixture = serialized['use_mixture']
        bn.encoders = serialized['encoders']

        return bn


class DiscreteBNSerializer(AbstractBNSerializer):
    """
    Serializer for Discrete Bayesian Networks, implementing methods to create
    instances of Discrete Bayesian Network and Mixture Gaussian Node.
    """

    def _create_bn(self) -> T:
        return DiscreteBN()

    def _create_node(self, name: str) -> N:
        return DiscreteNode(name)


class ContinuousBNSerializer(AbstractBNSerializer):
    """
    Serializer for Continuous Bayesian Networks, implementing methods to create
    instances of Continuous Bayesian Network and Network Node.
    """

    def _create_bn(self) -> T:
        return ContinuousBN(use_mixture=True)

    def _create_node(self, name: str) -> N:
        return MixtureGaussianNode(name)


class DictSerializer(abc.ABC):
    """
    Abstract base class for serializing and deserializing dictionaries.
    """

    @abc.abstractmethod
    def serialize(self, d: dict) -> str:
        """
        Serialize a dictionary to a string.

        Args:
            d (dict): The dictionary to serialize.

        Returns:
            str: The serialized string.
        """
        pass

    @abc.abstractmethod
    def deserialize(self, serialized: str) -> dict:
        """
        Deserialize a string to a dictionary.

        Args:
            serialized (str): The serialized string.

        Returns:
            dict: The deserialized dictionary.
        """
        pass


class JsonDictSerializer(DictSerializer):
    """
    Serializer and deserializer for dictionaries using JSON format.
    """

    def serialize(self, d: dict) -> str:
        return json.dumps(d)

    def deserialize(self, serialized: str) -> dict:
        return json.loads(serialized)


class Storage(abc.ABC):
    """
    Abstract base class for storing and retrieving data.
    """

    @abc.abstractmethod
    def write(self, d: dict, pathname: str):
        """
        Write a dictionary to a storage location identified by pathname.

        Args:
            d (dict): The dictionary to write.
            pathname (str): The path or identifier of the storage location.
        """
        pass

    @abc.abstractmethod
    def read(self, pathname: str) -> dict:
        """
        Read data from a storage location identified by pathname and return it as a dictionary.

        Args:
            pathname (str): The path or identifier of the storage location.

        Returns:
            dict: The data read from the storage location.
        """
        pass


class FilesystemStorage(Storage):
    """
    FilesystemStorage class represents a storage mechanism that reads and writes data to the filesystem.
    """

    def write(self, data: str, pathname: str):
        """
        Writes the provided data to the specified file path.

        Args:
            data (str): The data to be written to the file.
            pathname (str): The path to the file where the data will be written.

        Returns:
            None
        """
        with open(pathname, 'w') as f:
            f.write(data)

    def read(self, pathname: str) -> str:
        """
        Reads data from the specified file path and returns it as a string.

        Args:
            pathname (str): The path to the file from which data will be read.

        Returns:
            str: The data read from the file as a string.
        """
        with open(pathname, 'r') as f:
            return f.read()
