from ..environment import Environment
from ..agents.pbvi_agent import PBVI_Agent
from .model_based_util.pomdp import Model
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
               model:Model,
               belief_set:BeliefSet,
               max_generation:int=10
               ) -> BeliefSet:
        '''
        This expansion technique relies only randomness and will generate at most 'max_generation' beliefs.

        Parameters
        model : pomdp.Model
            The POMDP model on which to expand the belief set on.
        belief_set : BeliefSet
            List of beliefs to expand on.
        max_generation : int, default=10
            The max amount of beliefs that can be added to the belief set at once.
        '''
        xp = np if not gpu_support else cp.get_array_module(belief_set.belief_array)

        # How many new beliefs to add
        generation_count = min(belief_set.belief_array.shape[0], max_generation)

        # Generation of the new beliefs at random
        new_beliefs = xp.random.random((generation_count, model.state_count))
        new_beliefs /= xp.sum(new_beliefs, axis=1)[:,None]

        return BeliefSet(model, new_beliefs)


    def train(self,
              expansions:int,
              full_backup:bool=True,
              update_passes:int=1,
              max_belief_growth:int=10,
              initial_belief:BeliefSet|Belief|None=None,
              initial_value_function:ValueFunction|None=None,
              prune_level:int=1,
              prune_interval:int=10,
              limit_value_function_size:int=-1,
              use_gpu: bool = False,
              history_tracking_level: int = 1,
              force: bool = False,
              print_progress: bool = True) -> None:
        
        return super().train(expansions,
                             full_backup,
                             update_passes,
                             max_belief_growth,
                             initial_belief,
                             initial_value_function,
                             prune_level,
                             prune_interval,
                             limit_value_function_size,
                             use_gpu,
                             history_tracking_level,
                             force,
                             print_progress)