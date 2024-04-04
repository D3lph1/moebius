from datetime import timedelta

from golem.core.adapter import DirectAdapter
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.objective.objective import Objective
from golem.core.optimisers.objective.objective_eval import ObjectiveEvaluate
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams

from .graph.graph import create_generator_model_graph, GraphShuffler, GeneratorModel, GeneratorNode
from .graph.initializer import GMMParametersInitializer
from .metric import optimisation_metric, metric_olr_avg_discrete
from .parameters import EvolutionaryConfiguration

n_generation = 500
time_m = 60

class GMMOptimizer:
    __evol_config: EvolutionaryConfiguration
    __graph_shuffler: GraphShuffler
    __params_initializer: GMMParametersInitializer

    def __init__(
            self,
            evol_config: EvolutionaryConfiguration,
            graph_shuffler: GraphShuffler,
            params_initializer: GMMParametersInitializer
    ):
        self.__evol_config = evol_config
        self.__graph_shuffler = graph_shuffler
        self.__params_initializer = params_initializer

    def optimize(self, n_dim: int, n_comp: int, target_olr: float):
        graph = create_generator_model_graph(
            n_dim,
            n_comp,
            self.__params_initializer
        )

        graph = self.__graph_shuffler.shuffle(graph)

        objective = Objective(
            {'custom': lambda a: optimisation_metric(a, target_olr, metric_olr_avg_discrete)})
        objective_eval = ObjectiveEvaluate(objective)

        requirements = GraphRequirements(
            max_arity=100,
            max_depth=100,
            early_stopping_iterations=5,
            num_of_generations=n_generation,
            timeout=timedelta(minutes=time_m),
            history_dir=None,
            n_jobs=6
        )
        graph_generation_params = GraphGenerationParams(
            adapter=DirectAdapter(base_graph_class=GeneratorModel, base_node_class=GeneratorNode),
            rules_for_constraint=[]
        )

        optimiser = EvoGraphOptimizer(
            graph_generation_params=graph_generation_params,
            graph_optimizer_params=self.__evol_config.create_optimizer_parameters(),
            requirements=requirements,
            initial_graphs=[graph],
            objective=objective
        )

        return optimiser.optimise(objective_eval)
