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


class FSVI_Agent(PBVI_Agent):

    def expand(self,
               belief_set:BeliefSet,
               value_function:ValueFunction,
               max_generation:int,
               mdp_policy:ValueFunction,
               use_gpu:bool=False
               ) -> BeliefSet:
        '''
        # TODO: Not anymore a recursive function, to rework
        Function implementing the exploration process using the MDP policy in order to generate a sequence of Beliefs following the the Forward Search Value Iteration principles.
        It is a recursive function that is started by a initial state 's' and using the MDP policy, chooses the best action to take.
        Following this, a random next state 's_p' is being sampled from the transition probabilities and a random observation 'o' based on the observation probabilities.
        Then the given belief is updated using the chosen action and the observation received and the updated belief is added to the sequence.
        Once the state is a goal state, the recursion is done and the belief sequence is returned.

        Parameters
        ----------
        b0 : Belief
            The belief to start the sequence with.
            A random state will be chosen based on the probability distribution of the belief.
        mdp_policy : ValueFunction
            The mdp policy used to choose the action from with the given state 's'.
        max_generation : int, default=10
            The maximum recursion depth that can be reached before the generated belief sequence is returned.
        
        Returns
        -------
        belief_set : BeliefSet
            A new sequence of beliefs.
        '''
        # GPU support
        if use_gpu:
            assert gpu_support, "GPU support is not enabled, Cupy might need to be installed..."

        xp = np if not use_gpu else cp
        model = self.model if not use_gpu else self.model.gpu_model

        # Getting initial belief
        b0 = belief_set.belief_list[0]
        belief_list = [b0]

        # Choose a random starting state
        s = b0.random_state()

        # Setting the working belief
        b = b0

        for _ in range(max_generation - 1): #-1 due to a one belief already being present in the set
            # Choose action based on mdp value function
            a_star = xp.argmax(mdp_policy.alpha_vector_array[:,s])

            # Pick a random next state (weighted by transition probabilities)
            s_p = model.transition(s, a_star)
            
            # Pick a random observation weighted by observation probabilities in state s_p and after having done action a_star
            o = model.observe(s_p, a_star)
            
            # Generate a new belief based on a_star and o
            b_p = b.update(a_star, o)

            # Record new belief
            belief_list.append(b_p)

            # Updating s and b
            s = s_p
            b = b_p

            # Reset and belief if end state is reached
            if s in model.end_states:
                s = b0.random_state()
                b = b0

        return BeliefSet(model, belief_list)


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
              ) -> None:
        
        return super().train(expansions = expansions,
                             full_backup = False,
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