from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum

class EvolutionaryConfiguration:
    def __init__(self):
        self.__population_size = 10
        self.__max_population_size = 55
        self.__crossover_probability = 0.8
        self.__mutation_probability = 0.9
        self.__mutation_types = []
        self.__crossover_types = []

    def set_population_size(self, population_size: int):
        self.__population_size = population_size

        return self

    def set_max_population_size(self, max_population_size: int):
        self.__max_population_size = max_population_size

        return self

    def set_crossover_probability(self, crossover_probability: float):
        if crossover_probability < 0.0:
            raise ValueError("crossover_probability must be greater or equal to zero")

        self.__crossover_probability = crossover_probability

        return self

    def set_mutation_probability(self, mutation_probability: float):
        if mutation_probability < 0.0:
            raise ValueError("mutation_probability must be greater or equal to zero")

        self.__mutation_probability = mutation_probability

        return self

    def set_mutation_types(self, mutation_types: list):
        self.__mutation_types = mutation_types

        return self

    def append_mutation_type(self, mutation_type):
        self.__mutation_types.append(mutation_type)

        return self

    def set_crossover_types(self, crossover_types: list):
        self.__crossover_types = crossover_types

        return self

    def append_crossover_type(self, crossover_type):
        self.__crossover_types.append(crossover_type)

        return self

    def create_optimizer_parameters(self) -> GPAlgorithmParameters:
        return GPAlgorithmParameters(
            max_pop_size=self.__max_population_size,
            pop_size=self.__population_size,
            crossover_prob=self.__crossover_probability,
            mutation_prob=self.__mutation_probability,
            genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
            selection_types=[SelectionTypesEnum.tournament],
            mutation_types=self.__mutation_types,
            crossover_types=self.__crossover_types
        )
