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


class PBVI_SSEA_Agent(PBVI_Agent):

    def expand(self,
               model:Model,
               belief_set:BeliefSet,
               max_generation:int=10
               ) -> BeliefSet:
        '''
        Stochastic Simulation with Exploratory Action.
        Simulates running steps forward for each possible action knowing we are a state s, chosen randomly with according to the belief probability.
        These lead to a new state s_p and a observation o for each action.
        From all these and observation o we can generate updated beliefs. 
        Then it takes the belief that is furthest away from other beliefs, meaning it explores the most the belief space.

        Parameters
        ----------
        model : pomdp.Model
            The POMDP model on which to expand the belief set on.
        belief_set : BeliefSet
            List of beliefs to expand on.
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

        # Generation of successors
        successor_beliefs = xp.array([[[b.update(a,o).values for o in model.observations] for a in model.actions] for b in belief_set.belief_list])
        
        # Compute the distances between each pair and of successor are source beliefs
        diff = (belief_set.belief_array[:, None,None,None, :] - successor_beliefs)
        dist = xp.sqrt(xp.einsum('bnaos,bnaos->bnao', diff, diff))

        # Taking the min distance for each belief
        belief_min_dists = xp.min(dist,axis=0)

        # Taking the max distanced successors
        b_star, a_star, o_star = xp.unravel_index(xp.argsort(belief_min_dists, axis=None)[::-1][:to_generate], successor_beliefs.shape[:-1])

        # Selecting successor beliefs
        new_belief_array = successor_beliefs[b_star[:,None], a_star[:,None], o_star[:,None], model.states[None,:]]

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