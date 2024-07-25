from datetime import datetime
from inspect import signature

import random

import numpy as np
gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')

from olfactory_navigation.agents.model_based_util.mdp import log
from olfactory_navigation.agents.model_based_util.mdp import Model as MDP_Model


class Model(MDP_Model):
    '''
    POMDP Model class. Partially Observable Markov Decision Process Model.


    Parameters
    ----------
    states : int or list[str] or list[list[str]]
        A list of state labels or an amount of states to be used. Also allows to provide a matrix of states to define a grid model.
    actions : int or list
        A list of action labels or an amount of actions to be used.
    observations : int or list
        A list of observation labels or an amount of observations to be used
    transitions : array-like or function, optional
        The transitions between states, an array can be provided and has to be |S| x |A| x |S| or a function can be provided. 
        If a function is provided, it has be able to deal with np.array arguments.
        If none is provided, it will be randomly generated.
    reachable_states : array-like, optional
        A list of states that can be reached from each state and actions. It must be a matrix of size |S| x |A| x |R| where |R| is the max amount of states reachable from any given state and action pair.
        It is optional but useful for speedup purposes.
    rewards : array-like or function, optional
        The reward matrix, has to be |S| x |A| x |S|.
        A function can also be provided here but it has to be able to deal with np.array arguments.
        If provided, it will be use in combination with the transition matrix to fill to expected rewards.
    observation_table : array-like or function, optional
        The observation matrix, has to be |S| x |A| x |O|. If none is provided, it will be randomly generated.
    rewards_are_probabilistic: bool, default=False
        Whether the rewards provided are probabilistic or pure rewards. If probabilist 0 or 1 will be the reward with a certain probability.
    state_grid : array-like, optional
        If provided, the model will be converted to a grid model.
    start_probabilities : list, optional
        The distribution of chances to start in each state. If not provided, there will be an uniform chance for each state. It is also used to represent a belief of complete uncertainty.
    end_states : list, optional
        Entering either state in the list during a simulation will end the simulation.
    end_actions : list, optional
        Playing action of the list during a simulation will end the simulation.
    print_debug : bool, default=False
        Whether to print debug logs about the creation progress of the POMDP Model.
    seed : int, default=12131415
        For reproducible randomness.

    Attributes
    ----------
    states : np.ndarray
        A 1D array of states indices. Used to loop over states.
    state_labels : list[str]
        A list of state labels. (To be mainly used for plotting)
    state_count : int
        How many states are in the Model.
    state_grid : np.ndarray
        The state indices organized as a 2D grid. (Used for plotting purposes)
    actions : np.ndarry
        A 1D array of action indices. Used to loop over actions.
    action_labels : list[str]
        A list of action labels. (To be mainly used for plotting)
    action_count : int
        How many action are in the Model.
    observations : np.ndarray
        A 1D array of observation indices. Used to loop over obervations.
    observation_labels : list[str]
        A list of observation labels. (To be mainly used for plotting)
    observation_count : int
        How many observations can be made in the Model.
    transition_table : np.ndarray
        A 3D matrix of the transition probabilities.
        Can be None in the case a transition function is provided instead.
        Note: When possible, use reachable states and reachable probabilities instead.
    transition_function : function
        A callable function taking 3 arguments: s, a, s_p; and returning a float between 0.0 and 1.0.
        Can be None in the case a transition table is provided instead.
        Note: When possible, use reachable states and reachable probabilities instead.
    observation_table : np.ndarray
        A 3D matrix of shape S x A x O representing the probabilies of obsevating o when taking action a and leading to state s_p.
    reachable_states : np.ndarray
        A 3D array of the shape S x A x R, where R is max amount to states that can be reached from any state-action pair.
    reachable_probabilities : np.ndarray
        A 3D array of the same shape as reachable_states, the array represent the probability of reaching the state pointed by the reachable_states matrix.
    reachable_state_count : int
        The maximum of states that can be reached from any state-action combination.
    reachable_transitional_observation_table : np.ndarray
        A 4D array of shape S x A x O x R, representing the probabiliies of landing if each reachable state r, while observing o after having taken action a from state s.
        Mainly used to speedup repeated operations in solver.
    immediate_reward_table : np.ndarray
        A 3D matrix of shape S x A x S x O of the reward that will received when taking action a, in state s, landing in state s_p, and observing o.
        Can be None in the case an immediate rewards function is provided instead.
    immediate_reward_function : function
        A callable function taking 4 argments: s, a, s_p, o and returning the immediate reward the agent will receive.
        Can be None in the case an immediate rewards function is provided instead.
    expected_reward_table : np.ndarray
        A 2D array of shape S x A. It represents the rewards that is expected to be received when taking action a from state s.
        It is made by taking the weighted average of immediate rewards with the transitions and the observation probabilities.
    start_probabilities : np.ndarray
        A 1D array of length |S| containing the probility distribution of the agent starting in each state.
    rewards_are_probabilisitic : bool
        Whether the immediate rewards are probabilitic, ie: returning a 0 or 1 based on the reward that is considered to be a probability.
    end_states : list[int]
        A list of states that, when reached, terminate a simulation.
    end_actions : list[int]
        A list of actions that, when taken, terminate a simulation.
    is_on_gpu : bool
        Whether the numpy array of the model are stored on the gpu or not.
    gpu_model : mdp.Model
        An equivalent model with the np.ndarray objects on GPU. (If already on GPU, returns self)
    cpu_model : mdp.Model
        An equivalent model with the np.ndarray objects on CPU. (If already on CPU, returns self)
    seed : int
        The seed used for the random operations (to allow for reproducability).
    rnd_state : np.random.RandomState
        The random state variable used to generate random values.
    '''
    def __init__(self,
                 states: int | list[str] | list[list[str]],
                 actions: int | list,
                 observations: int | list,
                 transitions = None,
                 reachable_states = None,
                 rewards = None,
                 observation_table = None,
                 rewards_are_probabilistic: bool = False,
                 state_grid = None,
                 start_probabilities: list | None = None,
                 end_states: list[int] = [],
                 end_actions: list[int] = [],
                 print_debug: bool = False,
                 seed: int = 12131415
                 ) -> None:
        super().__init__(states=states,
                         actions=actions,
                         transitions=transitions,
                         reachable_states=reachable_states,
                         rewards=-1, # Defined here lower since immediate reward table has different shape for MDP is different than for POMDP
                         rewards_are_probabilistic=rewards_are_probabilistic,
                         state_grid=state_grid,
                         start_probabilities=start_probabilities,
                         end_states=end_states,
                         end_actions=end_actions,
                         print_debug=print_debug,
                         seed=seed)
        # Debug logger
        def logger(content: str):
            if print_debug:
                log(content=content)

        if print_debug:
            print()
            log('POMDP particular parameters:')

        # ------------------------- Observations -------------------------
        if isinstance(observations, int):
            self.observation_labels = [f'o_{i}' for i in range(observations)]
        else:
            self.observation_labels = observations
        self.observation_count = len(self.observation_labels)
        self.observations = np.arange(self.observation_count)

        if observation_table is None:
            # If no observation matrix given, generate random one
            random_probs = self.rnd_state.random((self.state_count, self.action_count, self.observation_count))
            # Normalization to have s_p probabilies summing to 1
            self.observation_table = random_probs / np.sum(random_probs, axis=2, keepdims=True)
        else:
            self.observation_table = np.array(observation_table)
            o_shape = self.observation_table.shape
            exp_shape = (self.state_count, self.action_count, self.observation_count)
            assert o_shape == exp_shape, f"Observations table doesnt have the right shape, it should be SxAxO (expected: {exp_shape}, received: {o_shape})."

        logger(f'- {self.observation_count} observations')

        # ------------------------- Reachable transitional observation probabilities -------------------------
        logger('- Starting of transitional observations for reachable states table')
        start_ts = datetime.now()

        reachable_observations = self.observation_table[self.reachable_states[:,:,None,:], self.actions[None,:,None,None], self.observations[None,None,:,None]] # SAOR
        self.reachable_transitional_observation_table = np.einsum('sar,saor->saor', self.reachable_probabilities, reachable_observations)
        
        duration = (datetime.now() - start_ts).total_seconds()
        logger(f'    > Done in {duration:.3f}s')

        # ------------------------- Rewards -------------------------
        self.immediate_reward_table = None
        self.immediate_reward_function = None
        
        if rewards is None:
            if (len(self.end_states) > 0) or (len(self.end_actions) > 0):
                logger('- [Warning] Rewards are not define but end states/actions are, reaching an end state or doing an end action will give a reward of 1.')
                self.immediate_reward_function = self._end_reward_function
            else:
                # If no reward matrix given, generate random one
                self.immediate_reward_table = self.rnd_state.random((self.state_count, self.action_count, self.state_count, self.observation_count))
        elif callable(rewards):
            # Rewards is a function
            logger('- [Warning] The rewards are provided as a function, if the model is saved, the rewards will need to be defined before loading model.')
            logger('    > Alternative: Setting end states/actions and leaving the rewards can be done to make the end states/action giving a reward of 1 by default.')
            self.immediate_reward_function = rewards
            assert len(signature(rewards).parameters) == 4, "Reward function should accept 4 parameters: s, a, sn, o..."
        else:
            # Array like
            self.immediate_reward_table = np.array(rewards)
            r_shape = self.immediate_reward_table.shape
            exp_shape = (self.state_count, self.action_count, self.state_count, self.observation_count)
            assert r_shape == exp_shape, f"Rewards table doesnt have the right shape, it should be SxAxSxO (expected: {exp_shape}, received {r_shape})"
        
        # ------------------------- Expected rewards -------------------------
        logger('- Starting generation of expected rewards table')
        start_ts = datetime.now()

        reachable_rewards = None
        if self.immediate_reward_table is not None:
            reachable_rewards = rewards[self.states[:,None,None,None], self.actions[None,:,None,None], self.reachable_states[:,:,:,None], self.observations[None,None,None,:]]
        else:
            def reach_reward_func(s,a,ri,o):
                s = s.astype(int)
                a = a.astype(int)
                ri = ri.astype(int)
                o = o.astype(int)
                return self.immediate_reward_function(s,a,self.reachable_states[s,a,ri],o)
            
            reachable_rewards = np.fromfunction(reach_reward_func, (*self.reachable_states.shape, self.observation_count))

        self._min_reward = float(np.min(reachable_rewards))
        self._max_reward = float(np.max(reachable_rewards))

        self.expected_rewards_table = np.einsum('saor,saro->sa', self.reachable_transitional_observation_table, reachable_rewards)

        duration = (datetime.now() - start_ts).total_seconds()
        logger(f'    > Done in {duration:.3f}s')


    def _end_reward_function(self, s, a, sn, o):
        '''
        The default reward function.
        Returns 1 if the next state sn is in the end states or if the action is in the end actions (terminating actions)
        '''
        return (np.isin(sn, self.end_states) | np.isin(a, self.end_actions)).astype(int)


    def reward(self,
               s: int,
               a: int,
               s_p: int,
               o: int
               ) -> int | float:
        '''
        Returns the rewards of playing action a when in state s and landing in state s_p.
        If the rewards are probabilistic, it will return 0 or 1.

        Parameters
        ----------
        s : int
            The current state.
        a : int
            The action taking in state s.
        s_p : int
            The state landing in after taking action a in state s
        o : int
            The observation that is done after having played action a in state s and landing in s_p

        Returns
        -------
        reward : int or float
            The reward received.
        '''
        reward = float(self.immediate_reward_table[s,a,s_p,o] if self.immediate_reward_table is not None else self.immediate_reward_function(s,a,s_p,o))
        if self.rewards_are_probabilistic:
            rnd = random.random()
            return 1 if rnd < reward else 0
        else:
            return reward
    

    def observe(self,
                s_p: int,
                a: int
                ) -> int:
        '''
        Returns a random observation knowing action a is taken from state s, it is weighted by the observation probabilities.

        Parameters
        ----------
        s_p : int
            The state landed on after having done action a.
        a : int
            The action to take.

        Returns
        -------
        o : int
            A random observation.
        '''
        xp = cp if self.is_on_gpu else np
        o = int(self.rnd_state.choice(a=self.observations, size=1, p=self.observation_table[s_p,a])[0])
        return o