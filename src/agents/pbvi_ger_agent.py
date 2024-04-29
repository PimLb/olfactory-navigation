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


class PBVI_GER_Agent(PBVI_Agent):

    def expand(self,
               belief_set:BeliefSet,
               value_function:ValueFunction,
               max_generation:int,
               use_gpu:bool=False
               ) -> BeliefSet:
        '''
        Greedy Error Reduction.
        It attempts to choose the believes that will maximize the improvement of the value function by minimizing the error.
        The error is computed by the sum of the change between two beliefs and their two corresponding alpha vectors.

        Parameters
        ----------
        model : pomdp.Model
            The POMDP model on which to expand the belief set on.
        belief_set : BeliefSet
            List of beliefs to expand on.
        value_function : ValueFunction
            Used to find the best action knowing the belief.
        max_generation : int, default=10
            The max amount of beliefs that can be added to the belief set at once.

        Returns
        -------
        belief_set_new : BeliefSet
            Union of the belief_set and the expansions of the beliefs in the belief_set
        '''
        # GPU support
        if use_gpu:
            assert gpu_support, "GPU support is not enabled, Cupy might need to be installed..."

        xp = np if not use_gpu else cp
        model = self.model if not use_gpu else self.model.gpu_model

        old_shape = belief_set.belief_array.shape
        to_generate = min(max_generation, old_shape[0])

        new_belief_array = xp.empty((old_shape[0] + to_generate, old_shape[1]))
        new_belief_array[:old_shape[0]] = belief_set.belief_array

        # Finding the min and max rewards for computation of the epsilon
        r_min = model._min_reward / (1 - self.gamma)
        r_max = model._max_reward / (1 - self.gamma)

        # Generation of all potential successor beliefs
        successor_beliefs = xp.array([[[b.update(a,o).values for o in model.observations] for a in model.actions] for b in belief_set.belief_list])
        
        # Finding the alphas associated with each previous beliefs
        best_alpha = xp.argmax(xp.dot(belief_set.belief_array, value_function.alpha_vector_array.T), axis = 1)
        b_alphas = value_function.alpha_vector_array[best_alpha]

        # Difference between beliefs and their successors
        b_diffs = successor_beliefs - belief_set.belief_array[:,None,None,:]

        # Computing a 'next' alpha vector made of the max and min
        alphas_p = xp.where(b_diffs >= 0, r_max, r_min)

        # Difference between alpha vectors and their successors alpha vector
        alphas_diffs = alphas_p - b_alphas[:,None,None,:]

        # Computing epsilon for all successor beliefs
        eps = xp.einsum('baos,baos->bao', alphas_diffs, b_diffs)

        # Computing the probability of the b and doing action a and receiving observation o
        bao_probs = xp.einsum('bs,saor->bao', belief_set.belief_array, model.reachable_transitional_observation_table)

        # Taking the sumproduct of the probs with the epsilons
        res = xp.einsum('bao,bao->ba', bao_probs, eps)

        # Picking the correct amount of initial beliefs and ideal actions
        b_stars, a_stars = xp.unravel_index(xp.argsort(res, axis=None)[::-1][:to_generate], res.shape)

        # And picking the ideal observations
        o_star = xp.argmax(bao_probs[b_stars[:,None], a_stars[:,None], model.observations[None,:]] * eps[b_stars[:,None], a_stars[:,None], model.observations[None,:]], axis=1)

        # Selecting the successor beliefs
        new_belief_array = successor_beliefs[b_stars[:,None], a_stars[:,None], o_star[:,None], model.states[None,:]]

        return BeliefSet(model, new_belief_array)


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