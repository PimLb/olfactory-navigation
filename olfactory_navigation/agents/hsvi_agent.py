from ..agents.pbvi_agent import PBVI_Agent, TrainingHistory
from .model_based_util.value_function import ValueFunction
from .model_based_util.belief import Belief, BeliefSet
from .model_based_util.belief_value_mapping import BeliefValueMapping

from typing import Union

import numpy as np
gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')


class HSVI_Agent(PBVI_Agent):
    '''
    A flavor of the PBVI Agent. 
    
    # TODO: Do document of HSVI agent
    # TODO: FIX HSVI expand

    Parameters
    ----------
    environment : Environment
        The olfactory environment to train the agent with.
    threshold : float, optional, default=3e-6
        The olfactory sensitivity of the agent. Odor cues under this threshold will not be detected by the agent.
    name : str, optional
        A custom name to give the agent. If not provided is will be a combination of the class-name and the threshold.

    Attributes
    ---------
    environment : Environment
    threshold : float
    name : str
    model : pomdp.Model
        The environment converted to a POMDP model using the "from_environment" constructor of the pomdp.Model class.
    saved_at : str
        The place on disk where the agent has been saved (None if not saved yet).
    on_gpu : bool
        Whether the agent has been sent to the gpu or not.
    trained_at : str
        A string timestamp of when the agent has been trained (None if not trained yet).
    value_function : ValueFunction
        The value function used for the agent to make decisions.
    belief : BeliefSet
        Used only during simulations.
        Part of the Agent's status. Where the agent believes he is over the state space.
        It is a list of n belief points based on how many simulations are running at once.
    action_played : list[int]
        Used only during simulations.
        Part of the Agent's status. Records what action was last played by the agent.
        A list of n actions played based on how many simulations are running at once.
    '''
    def expand(self,
               belief_set:BeliefSet,
               value_function:ValueFunction,
               max_generation:int
               ) -> BeliefSet:
        '''
        The expand function of the  Heuristic Search Value Iteration (HSVI) technique.
        It is a redursive function attempting to minimize the bound between the upper and lower estimations of the value function.

        It is developped by Smith T. and Simmons R. and described in the paper "Heuristic Search Value Iteration for POMDPs".

        Parameters
        ----------
        belief_set : BeliefSet
            List of beliefs to expand on.
        value_function : ValueFunction
            The current value function. Used to compute the value at belief points.
        max_generation : int, default=10
            The max amount of beliefs that can be added to the belief set at once.

        Returns
        -------
        belief_set : BeliefSet
            A new sequence of beliefs.
        '''
        # GPU support
        xp = np if not self.on_gpu else cp
        model = self.model

        if conv_term is None:
            conv_term = self.eps

        # Update convergence term
        conv_term /= self.gamma

        # Find best a based on upper bound v
        max_qv = -xp.inf
        best_a = -1
        for a in model.actions:
            b_probs = xp.einsum('sor,s->o', model.reachable_transitional_observation_table[:,a,:,:], b.values)

            b_prob_val = 0
            for o in model.observations:
                b_prob_val += (b_probs[o] * upper_bound_belief_value_map.evaluate(b.update(a,o)))
            
            qva = float(xp.dot(model.expected_rewards_table[:,a], b.values) + (self.gamma * b_prob_val))

            # qva = upper_bound_belief_value_map.qva(b, a, gamma=self.gamma)
            if qva > max_qv:
                max_qv = qva
                best_a = a

        # Choose o that max gap between bounds
        b_probs = xp.einsum('sor,s->o', model.reachable_transitional_observation_table[:,best_a,:,:], b.values)

        max_o_val = -xp.inf
        best_v_diff = -xp.inf
        next_b = b

        for o in model.observations:
            bao = b.update(best_a, o)

            upper_v_bao = upper_bound_belief_value_map.evaluate(bao)
            lower_v_bao = xp.max(xp.dot(value_function.alpha_vector_array, bao.values))

            v_diff = (upper_v_bao - lower_v_bao)

            o_val = b_probs[o] * v_diff
            
            if o_val > max_o_val:
                max_o_val = o_val
                best_v_diff = v_diff
                next_b = bao

        # if bounds_split < conv_term or max_generation <= 0:
        if best_v_diff < conv_term or max_generation <= 1:
            return BeliefSet(model, [next_b])
        
        # Add the belief point and associated value to the belief-value mapping
        upper_bound_belief_value_map.add(b, max_qv)

        # Go one step deeper in the recursion
        b_set = self.expand_hsvi(model=model,
                                 b=next_b,
                                 value_function=value_function,
                                 upper_bound_belief_value_map=upper_bound_belief_value_map,
                                 conv_term=conv_term,
                                 max_generation=max_generation-1)
        
        # Append the nex belief of this iteration to the deeper beliefs
        new_belief_list = b_set.belief_list
        new_belief_list.append(next_b)

        return BeliefSet(model, new_belief_list)


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
              print_progress:bool=True,
              print_stats:bool=True
              ) -> TrainingHistory:
        '''
        Main loop of the Point-Based Value Iteration algorithm.
        It consists in 2 steps, Backup and Expand.
        1. Expand: Expands the belief set base with a expansion strategy given by the parameter expand_function
        2. Backup: Updates the alpha vectors based on the current belief set

        Heuristic Search Value Iteration:
        - By default it performs the backup only on set of beliefs generated by the expand function. (so it full_backup=False)

        Parameters
        ----------
        expansions : int
            How many times the algorithm has to expand the belief set. (the size will be doubled every time, eg: for 5, the belief set will be of size 32)
        update_passes : int, default=1
            How many times the backup function has to be run every time the belief set is expanded.
        max_belief_growth : int, default=10
            How many beliefs can be added at every expansion step to the belief set.
        initial_belief : BeliefSet or Belief, optional
            An initial list of beliefs to start with.
        initial_value_function : ValueFunction, optional
            An initial value function to start the solving process with.
        prune_level : int, default=1
            Parameter to prune the value function further before the expand function.
        prune_interval : int, default=10
            How often to prune the value function. It is counted in number of backup iterations.
        limit_value_function_size : int, default=-1
            When the value function size crosses this threshold, a random selection of 'max_belief_growth' alpha vectors will be removed from the value function
            If set to -1, the value function can grow without bounds.
        use_gpu : bool, default=False
            Whether to use the GPU with cupy array to accelerate solving.
        gamma : float, default=0.99
            The discount factor to value immediate rewards more than long term rewards.
            The learning rate is 1/gamma.
        eps : float, default=1e-6
            The smallest allowed changed for the value function.
            Bellow the amound of change, the value function is considered converged and the value iteration process will end early.
        history_tracking_level : int, default=1
            How thorough the tracking of the solving process should be. (0: Nothing; 1: Times and sizes of belief sets and value function; 2: The actual value functions and beliefs sets)
        force : bool, default=False
            Whether to force retraining if a value function already exists for this agent.
        print_progress : bool, default=True
            Whether or not to print out the progress of the value iteration process.
        print_stats : bool, default=True
            Whether or not to print out statistics at the end of the training run.

        Returns
        -------
        solver_history : SolverHistory
            The history of the solving process with some plotting options.
        '''
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
                             print_progress = print_progress,
                             print_stats = print_stats)