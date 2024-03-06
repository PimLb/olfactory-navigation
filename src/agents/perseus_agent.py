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


class Perseus_Agent(PBVI_Agent):

    def expand(self,
               model:Model,
               b:Belief,
               max_generation:int=10
               ) -> BeliefSet:
        '''
        Function implementing the exploration process using the MDP policy in order to generate a sequence of Beliefs.
        It is a recursive function that is started by a initial state 's' and using the MDP policy, chooses the best action to take.
        Following this, a random next state 's_p' is being sampled from the transition probabilities and a random observation 'o' based on the observation probabilities.
        Then the given belief is updated using the chosen action and the observation received and the updated belief is added to the sequence.
        Once the state is a goal state, the recursion is done and the belief sequence is returned.

        Parameters
        ----------
        model : pomdp.Model
            The model in which the exploration process will happen.
        b : Belief
            A belief to be added to the returned belief sequence and updated for the next step of the recursion.
        max_generation : int, default=10
            The maximum recursion depth that can be reached before the generated belief sequence is returned.
        
        Returns
        -------
        belief_set : BeliefSet
            A new sequence of beliefs.
        '''
        xp = np if not gpu_support else cp.get_array_module(b.values)

        initial_belief = b
        belief_sequence = []

        for i in range(max_generation):
            # Choose random action
            a = int(xp.random.choice(model.actions, size=1)[0])

            # Choose random observation based on prob: P(o|b,a)
            obs_prob = xp.einsum('sor,s->o', model.reachable_transitional_observation_table[:,a,:,:], b.values)
            o = int(xp.random.choice(model.observations, size=1, p=obs_prob)[0])

            # Update belief
            bao = b.update(a,o)

            # Finalization
            belief_sequence.append(bao)
            b = bao

        return BeliefSet(model, belief_sequence)


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