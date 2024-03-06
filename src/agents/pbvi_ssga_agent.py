from ..environment import Environment
from ..agents.pbvi_agent import PBVI_Agent
from .model_based_util.pomdp import Model
from .model_based_util.value_function import ValueFunction
from .model_based_util.belief import Belief, BeliefSet

from random import random

import numpy as np
gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')


class PBVI_SSGA_Agent(PBVI_Agent):

    def expand(self,
               model:Model,
               belief_set:BeliefSet,
               value_function:ValueFunction,
               epsilon:float=0.1,
               max_generation:int=10
               ) -> BeliefSet:
        '''
        Stochastic Simulation with Greedy Action.
        Simulates running a single-step forward from the beliefs in the "belief_set".
        The step forward is taking assuming we are in a random state s (weighted by the belief),
        then taking the best action a based on the belief with probability 'epsilon'.
        These lead to a new state s_p and a observation o.
        From this action a and observation o we can update our belief. 

        Parameters
        ----------
        model : pomdp.Model
            The POMDP model on which to expand the belief set on.
        belief_set : BeliefSet
            List of beliefs to expand on.
        value_function : ValueFunction
            Used to find the best action knowing the belief.
        eps : float
            Parameter tuning how often we take a greedy approach and how often we move randomly.
        max_generation : int, default=10
            The max amount of beliefs that can be added to the belief set at once.

        Returns
        -------
        belief_set_new : BeliefSet
            Union of the belief_set and the expansions of the beliefs in the belief_set
        '''
        xp = np if not gpu_support else cp.get_array_module(belief_set.belief_array)

        old_shape = belief_set.belief_array.shape
        to_generate = min(max_generation, old_shape[0])

        new_belief_array = xp.empty((to_generate, old_shape[1]))

        # Random previous beliefs
        rand_ind = np.random.choice(np.arange(old_shape[0]), to_generate, replace=False)

        for i, belief_vector in enumerate(belief_set.belief_array[rand_ind]):
            b = Belief(model, belief_vector)
            s = b.random_state()
            
            if random.random() < epsilon:
                a = random.choice(model.actions)
            else:
                best_alpha_index = xp.argmax(xp.dot(value_function.alpha_vector_array, b.values))
                a = value_function.actions[best_alpha_index]
            
            s_p = model.transition(s, a)
            o = model.observe(s_p, a)
            b_new = b.update(a, o)
            
            new_belief_array[i] = b_new.values
            
        return BeliefSet(model, new_belief_array)


    def train(self,
              expansions:int,
              full_backup:bool = True,
              update_passes: int = 1,
              max_belief_growth: int = 10,
              initial_belief: BeliefSet | Belief | None = None,
              initial_value_function: ValueFunction | None = None,
              prune_level: int = 1, prune_interval: int = 10,
              limit_value_function_size: int = -1,
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