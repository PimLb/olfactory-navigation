from ..agents.pbvi_agent import PBVI_Agent, TrainingHistory
from .model_based_util.value_function import ValueFunction
from .model_based_util.belief import Belief, BeliefSet

import numpy as np
gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')


class PBVI_RA_Agent(PBVI_Agent):
    
    def expand(self,
               belief_set:BeliefSet,
               value_function:ValueFunction,
               max_generation:int,
               use_gpu:bool=False
               ) -> BeliefSet:
        '''
        This expansion technique relies only randomness and will generate at most 'max_generation' beliefs.

        Parameters
        ----------
        model : pomdp.Model
            The POMDP model on which to expand the belief set on.
        belief_set : BeliefSet
            List of beliefs to expand on.
        max_generation : int, default=10
            The max amount of beliefs that can be added to the belief set at once.
        '''
        # GPU support
        if use_gpu:
            assert gpu_support, "GPU support is not enabled, Cupy might need to be installed..."

        xp = np if not use_gpu else cp
        model = self.model if not use_gpu else self.model.gpu_model

        # How many new beliefs to add
        generation_count = min(belief_set.belief_array.shape[0], max_generation)

        # Generation of the new beliefs at random
        new_beliefs = xp.random.random((generation_count, model.state_count))
        new_beliefs /= xp.sum(new_beliefs, axis=1)[:,None]

        return BeliefSet(model, new_beliefs)


    def train(self,
              expansions:int,
              update_passes:int=1,
              max_belief_growth:int=10,
              initial_belief:BeliefSet|Belief|None=None,
              initial_value_function:ValueFunction|None=None,
              prune_level:int=1,
              prune_interval:int=10,
              limit_value_function_size:int=-1,
              gamma:float=0.99,
              eps:float=1e-6,
              use_gpu:bool=False,
              history_tracking_level:int=1,
              force:bool=False,
              print_progress:bool=True
              ) -> TrainingHistory:

        return super().train(expansions = expansions,
                             full_backup = True,
                             update_passes = update_passes,
                             max_belief_growth = max_belief_growth,
                             initial_belief = initial_belief,
                             initial_value_function = initial_value_function,
                             prune_level = prune_level,
                             prune_interval = prune_interval,
                             limit_value_function_size = limit_value_function_size,
                             gamma = gamma,
                             eps = eps,
                             use_gpu = use_gpu,
                             history_tracking_level = history_tracking_level,
                             force = force,
                             print_progress = print_progress)