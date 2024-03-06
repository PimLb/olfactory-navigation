from ..environment import Environment
from ..agents.pbvi_agent import PBVI_Agent
from .model_based_util.pomdp import Model
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

    def expand(self,
               model:Model,
               b:Belief,
               value_function:ValueFunction,
               upper_bound_belief_value_map:BeliefValueMapping,
               conv_term:Union[float,None]=None,
               max_generation:int=10
               ) -> BeliefSet:
        '''
        The expand function of the  Heruistic Search Value Iteration (HSVI) technique.
        It is a redursive function attempting to minimize the bound between the upper and lower estimations of the value function.

        It is developped by Smith T. and Simmons R. and described in the paper "Heuristic Search Value Iteration for POMDPs".
        
        Parameters
        ----------
        model : pomdp.Model
            The model in which the exploration process will happen.
        b : Belief
            A belief to be added to the returned belief sequence and updated for the next step of the recursion.
        value_function : ValueFunction
            The lower bound of the value function.
        upper_bound_belief_value_map : BeliefValueMapping
            The upper bound of the value function.
            Initially it is define with the mdp policy of the model (run: "BeliefValueMapping(model, mdp_policy)").
            It is then refined through the expansion process by adding newly found belief and value pairs.
        max_generation : int, default=10
            The maximum recursion depth that can be reached before the generated belief sequence is returned.
        
        Returns
        -------
        belief_set : BeliefSet
            A new sequence of beliefs.
        '''
        xp = np if not gpu_support else cp.get_array_module(b.values)

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