from datetime import datetime
from inspect import signature
from matplotlib import colors, patches
from matplotlib import pyplot as plt

import os
import pandas as pd
import pickle
import random

import numpy as np
gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')


COLOR_LIST = [{
    'name': item.replace('tab:',''),
    'id': item,
    'hex': value,
    'rgb': [int(value.lstrip('#')[i:i + (len(value)-1) // 3], 16) for i in range(0, (len(value)-1), (len(value)-1) // 3)]
    } for item, value in colors.TABLEAU_COLORS.items()] # type: ignore

COLOR_ARRAY = np.array([c['rgb'] for c in COLOR_LIST])


def log(content:str) -> None:
    '''
    Function to print a log line with a timestamp.

    Parameters
    ----------
    content : str
        The content to be printed as a log.
    '''
    print(f'[{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}] ' + content)


class Model:
    '''
    MDP Model class.


    Parameters
    ----------
    states : int or list[str] or list[list[str]]
        A list of state labels or an amount of states to be used. Also allows to provide a matrix of states to define a grid model.
    actions : int or list
        A list of action labels or an amount of actions to be used.
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
    rewards_are_probabilistic : bool, default=False
        Whether the rewards provided are probabilistic or pure rewards. If probabilist 0 or 1 will be the reward with a certain probability.
    state_grid : array-like, optional
        If provided, the model will be converted to a grid model.
    start_probabilities : list, optional
        The distribution of chances to start in each state. If not provided, there will be an uniform chance for each state.
    end_states : list, optional
        Entering either state in the list during a simulation will end the simulation.
    end_actions : list, optional
        Playing action of the list during a simulation will end the simulation.
    print_debug : bool, default=False
        Whether to print debug logs about the creation progress of the MDP Model.
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
    transition_table : np.ndarray
        A 3D matrix of the transition probabilities.
        Can be None in the case a transition function is provided instead.
        Note: When possible, use reachable states and reachable probabilities instead.
    transition_function : function
        A callable function taking 3 arguments: s, a, s_p; and returning a float between 0.0 and 1.0.
        Can be None in the case a transition table is provided instead.
        Note: When possible, use reachable states and reachable probabilities instead.
    reachable_states : np.ndarray
        A 3D array of the shape S x A x R, where R is max amount to states that can be reached from any state-action pair.
    reachable_probabilities : np.ndarray
        A 3D array of the same shape as reachable_states, the array represent the probability of reaching the state pointed by the reachable_states matrix.
    reachable_state_count : int
        The maximum of states that can be reached from any state-action combination.
    immediate_reward_table : np.ndarray
        A 3D matrix of shape S x A x S of the reward that will received when taking action a, in state s and landing in state s_p.
        Can be None in the case an immediate rewards function is provided instead.
    immediate_reward_function : function
        A callable function taking 3 argments: s, a, s_p and returning the immediate reward the agent will receive.
        Can be None in the case an immediate rewards function is provided instead.
    expected_reward_table : np.ndarray
        A 2D array of shape S x A. It represents the rewards that is expected to be received when taking action a from state s.
        It is made by taking the weighted average of immediate rewards and the transitions.
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
                 transitions = None,
                 reachable_states = None,
                 rewards = None,
                 rewards_are_probabilistic: bool = False,
                 state_grid = None,
                 start_probabilities: list | None = None,
                 end_states: list[int] = [],
                 end_actions: list[int] = [],
                 print_debug: bool = False,
                 seed: int = 12131415
                 ) -> None:
        # Debug logger
        def logger(content: str):
            if print_debug:
                log(content=content)

        # Empty variable
        self._alt_model = None
        self.is_on_gpu = False

        # Random variable
        self.seed = seed
        self.rnd_state = np.random.RandomState(seed = seed)
        
        logger('Instantiation of MDP Model:')
        
        # ------------------------- States -------------------------
        self.state_grid = None
        if isinstance(states, int): # State count
            self.state_labels = [f's_{i}' for i in range(states)]

        elif isinstance(states, list) and all(isinstance(item, list) for item in states): # 2D list of states
            dim1 = len(states)
            dim2 = len(states[0])
            assert all(len(state_dim) == dim2 for state_dim in states), "All sublists of states must be of equal size"
            
            self.state_labels = []
            for state_dim in states:
                for state in state_dim:
                    self.state_labels.append(state)

            self.state_grid = np.arange(dim1 * dim2).reshape(dim1, dim2)

        else: # Default: single of list of string items
            self.state_labels = [item for item in states if isinstance(item, str)]

        self.state_count = len(self.state_labels)
        self.states = np.arange(self.state_count)

        logger(f'- {self.state_count} states')

        # ------------------------- Actions -------------------------
        if isinstance(actions, int):
            self.action_labels = [f'a_{i}' for i in range(actions)]
        else:
            self.action_labels = actions
        self.action_count = len(self.action_labels)
        self.actions = np.arange(self.action_count)

        logger(f'- {self.action_count} actions')

        # ------------------------- Reachable states provided -------------------------
        self.reachable_states = None
        if reachable_states is not None:
            self.reachable_states = np.array(reachable_states)
            assert self.reachable_states.shape[:2] == (self.state_count, self.action_count), f"Reachable states provided is not of the expected shape (received {self.reachable_states.shape}, expected ({self.state_count}, {self.action_count}, :))"
            self.reachable_state_count = self.reachable_states.shape[2]

            logger(f'- At most {self.reachable_state_count} reachable states per state-action pair')

        # ------------------------- Transitions -------------------------
        logger('- Starting generation of transitions table')
        start_ts = datetime.now()

        self.transition_table = None
        self.transition_function = None
        if transitions is None:
            if reachable_states is None:
                # If no transitiong matrix and no reachable states given, generate random one
                logger('    > [Warning] No transition matrix and no reachable states have provided so a random transition matrix is generated...')
                random_probs = self.rnd_state.random((self.state_count, self.action_count, self.state_count))

                # Normalization to have s_p probabilies summing to 1
                self.transition_table = random_probs / np.sum(random_probs, axis=2, keepdims=True)
            else:
                # Make uniform transition probabilities over reachable states
                logger(f'    > [Warning] No transition matrix or function provided but reachable states are, so probability to reach any reachable states will "1 / reachable state count" so here: {1/self.reachable_state_count:.3f}.')

        elif callable(transitions): # Transition function
            self.transition_function = transitions
            # Attempt to create transition table in memory
            t_arr = None
            try:
                t_arr = np.fromfunction(self.transition_function, (self.state_count, self.action_count, self.state_count))
            except MemoryError:
                logger('    > [Warning] Not enough memory to store transition table, using transition function provided...')
            else:
                self.transition_table = t_arr

        else: # Array like
            self.transition_table = np.array(transitions)
            t_shape = self.transition_table.shape
            exp_shape = (self.state_count, self.action_count, self.state_count)
            assert t_shape == exp_shape, f"Transitions table provided doesnt have the right shape, it should be SxAxS (expected {exp_shape}, received {t_shape})"

        duration = (datetime.now() - start_ts).total_seconds()
        logger(f'    > Done in {duration:.3f}s')
        if duration > 1:
            logger(f'    > [Warning] Transition table generation took long, if not done already, try to use the reachable_states parameter to speedup the process.')

        # ------------------------- Rewards are probabilistic toggle -------------------------
        self.rewards_are_probabilistic = rewards_are_probabilistic

        # ------------------------- State grid -------------------------
        logger('- Generation of state grid')
        if state_grid is None and self.state_grid is None:
            self.state_grid = np.arange(self.state_count).reshape((1,self.state_count))
        
        elif state_grid is not None:
            assert all(isinstance(l, list) for l in state_grid), "The provided states grid must be a list of lists."

            grid_shape = (len(state_grid), len(state_grid[0]))
            assert all(len(l) == grid_shape[1] for l in state_grid), "All rows must have the same length."

            if all(all(isinstance(e, int) for e in l) for l in state_grid):
                state_grid = np.array(state_grid)
                try:
                    self.states[state_grid]
                except:
                    raise Exception('An error occured with the list of state indices provided...')
                else:
                    self.state_grid = state_grid

            else:
                logger('    > [Warning] Looping through all grid states provided to find the corresponding states, can take a while...')
                
                np_state_grid = np.zeros(grid_shape, dtype=int)
                states_covered = 0
                for i, row in enumerate(state_grid):
                    for j, element in enumerate(state_grid):
                        if isinstance(element, str) and (element in self.state_labels):
                            states_covered += 1
                            np_state_grid[i,j] = self.state_labels.index(element)
                        elif isinstance(element, int) and (element < self.state_count):
                            np_state_grid[i,j] = element
                        
                        else:
                            raise Exception(f'Countains a state (\'{state}\') not in the list of states...')

                assert states_covered == self.state_count, "Some states of the state list are missing..."

        # ------------------------- Start state probabilities -------------------------
        logger('- Generating start probabilities table')
        if start_probabilities is not None:
            assert len(start_probabilities) == self.state_count
            self.start_probabilities = np.array(start_probabilities,dtype=float)
        else:
            self.start_probabilities = np.full((self.state_count), 1/self.state_count)

        # ------------------------- End state conditions -------------------------
        self.end_states = end_states
        self.end_actions = end_actions
        
        # ------------------------- Reachable states -------------------------
        # If not set yet
        if self.reachable_states is None:
            logger('- Starting computation of reachable states from transition data')
            
            if self.state_count > 1000:
                logger('-    > [Warning] For models with large amounts of states, this operation can take time. Try generating it advance and use the parameter \'reachable_states\'...')
            
            start_ts = datetime.now()

            self.reachable_states = []
            self.reachable_state_count = 0
            for s in self.states:
                reachable_states_for_action = []
                for a in self.actions:
                    reachable_list = []
                    if self.transition_table is not None:
                        reachable_list = np.argwhere(self.transition_table[s,a,:] > 0)[:,0].tolist()
                    else:
                        for sn in self.states:
                            if self.transition_function(s,a,sn) > 0:
                                reachable_list.append(sn)
                    reachable_states_for_action.append(reachable_list)
                    
                    if len(reachable_list) > self.reachable_state_count:
                        self.reachable_state_count = len(reachable_list)

                self.reachable_states.append(reachable_states_for_action)

            # In case some state-action pairs lead to more states than other, we fill with the 1st non states not used
            for s in self.states:
                for a in self.actions:
                    to_add = 0
                    while len(self.reachable_states[s][a]) < self.reachable_state_count:
                        if to_add not in self.reachable_states[s][a]:
                            self.reachable_states[s][a].append(to_add)
                        to_add += 1

            # Converting to ndarray
            self.reachable_states = np.array(self.reachable_states, dtype=int)

            duration = (datetime.now() - start_ts).total_seconds()
            logger(f'    > Done in {duration:.3f}s')
            logger(f'- At most {self.reachable_state_count} reachable states per state-action pair')

        # ------------------------- Reachable state probabilities -------------------------
        logger('- Starting computation of reachable state probabilities from transition data')
        start_ts = datetime.now()

        if self.transition_function is None and self.transition_table is None:
            self.reachable_probabilities = np.full(self.reachable_states.shape, 1/self.reachable_state_count)
        elif self.transition_table is not None:
            self.reachable_probabilities = self.transition_table[self.states[:,None,None], self.actions[None,:,None], self.reachable_states]
        else:
            self.reachable_probabilities = np.fromfunction((lambda s,a,ri: self.transition_function(s.astype(int), a.astype(int), self.reachable_states[s.astype(int), a.astype(int), ri.astype(int)])), self.reachable_states.shape)
            
        duration = (datetime.now() - start_ts).total_seconds()
        logger(f'    > Done in {duration:.3f}s')

        # ------------------------- Rewards -------------------------
        self.immediate_reward_table = None
        self.immediate_reward_function = None
        if rewards == -1: # If -1 is set, it means the rewards are defined in the superclass POMDP
            pass
        elif rewards is None:
            if (len(self.end_states) > 0) or (len(self.end_actions) > 0):
                logger('- [Warning] Rewards are not define but end states/actions are, reaching an end state or doing an end action will give a reward of 1.')
                self.immediate_reward_function = self._end_reward_function
            else:
                # If no reward matrix given, generate random one
                self.immediate_reward_table = self.rnd_state.random((self.state_count, self.action_count, self.state_count))
        elif callable(rewards):
            # Rewards is a function
            logger('- [Warning] The rewards are provided as a function, if the model is saved, the rewards will need to be defined before loading model.')
            logger('    > Alternative: Setting end states/actions and leaving the rewards can be done to make the end states/action giving a reward of 1 by default.')
            self.immediate_reward_function = rewards
            assert len(signature(rewards).parameters) == 3, "Reward function should accept 3 parameters: s, a, sn..."
        else:
            # Array like
            self.immediate_reward_table = np.array(rewards)
            r_shape = self.immediate_reward_table.shape
            exp_shape = (self.state_count, self.action_count, self.state_count)
            assert r_shape == exp_shape, f"Rewards table doesnt have the right shape, it should be SxAxS (expected: {exp_shape}, received {r_shape})"

        # ------------------------- Min and max rewards -------------------------
        self._min_reward = None
        self._max_reward = None

        # ------------------------- Expected rewards -------------------------
        self.expected_rewards_table = None
        if rewards != -1:
            logger('- Starting generation of expected rewards table')
            start_ts = datetime.now()

            reachable_rewards = None
            if self.immediate_reward_table is not None:
                reachable_rewards = self.immediate_reward_table[self.states[:,None,None], self.actions[None,:,None], self.reachable_states]
            else:
                def reach_reward_func(s,a,ri):
                    s = s.astype(int)
                    a = a.astype(int)
                    ri = ri.astype(int)
                    return self.immediate_reward_function(s,a,self.reachable_states[s,a,ri])
                
                reachable_rewards = np.fromfunction(reach_reward_func, self.reachable_states.shape)
            
            self._min_reward = float(np.min(reachable_rewards))
            self._max_reward = float(np.max(reachable_rewards))

            self.expected_rewards_table = np.einsum('sar,sar->sa', self.reachable_probabilities, reachable_rewards)

            duration = (datetime.now() - start_ts).total_seconds()
            logger(f'    > Done in {duration:.3f}s')


    def _end_reward_function(self, s, a, sn):
        return (np.isin(sn, self.end_states) | np.isin(a, self.end_actions)).astype(int)
    
    
    def transition(self,
                   s: int,
                   a: int
                   ) -> int:
        '''
        Returns a random posterior state knowing we take action a in state t and weighted on the transition probabilities.

        Parameters
        ----------
        s : int 
            The current state
        a : int
            The action to take

        Returns
        -------
        s_p : int
            The posterior state
        '''
        xp = cp if self.is_on_gpu else np

        # Shortcut for deterministic systems
        if self.reachable_state_count == 1:
            return int(self.reachable_states[s,a,0])

        s_p = int(self.rnd_state.choice(a=self.reachable_states[s,a], size=1, p=self.reachable_probabilities[s,a])[0])
        return s_p
    

    def reward(self,
               s: int,
               a: int,
               s_p: int
               ) -> int | float:
        '''
        Returns the rewards of playing action a when in state s and landing in state s_p.
        If the rewards are probabilistic, it will return 0 or 1.

        Parameters
        ----------
        s : int
            The current state
        a : int
            The action taking in state s
        s_p : int
            The state landing in after taking action a in state s

        Returns
        -------
        reward : int or float
            The reward received.
        '''
        reward = float(self.immediate_reward_table[s,a,s_p] if self.immediate_reward_table is not None else self.immediate_reward_function(s,a,s_p))
        if self.rewards_are_probabilistic:
            rnd = random.random()
            return 1 if rnd < reward else 0
        else:
            return reward
    

    def get_coords(self,
                   items: int | list
                   ) -> list[list[int]] | list[int]:
        '''
        Function to get the coordinate (on the state_grid) for the provided state index or indices.

        Parameters
        ----------
        items : int or list[int]
            The states ids or id get convert to a 2D coordinate.

        Returns
        -------
        item_coords : list[int] or list[list[int]]
            The 2D positions of the provided item ids.
        '''
        item_list = [items] if isinstance(items, int) else items
        item_coords = [np.argwhere(self.cpu_model.state_grid == s)[0] for s in item_list]

        return item_coords[0] if isinstance(items, int) else item_coords


    def save(self,
             file_name: str,
             path: str = './Models'
             ) -> None:
        '''
        Function to save the current model in a pickle file.
        By default, the model will be saved in 'Models' directory in the current working directory but this can be changed using the 'path' parameter.

        Parameters
        ----------
        file_name : str
            The name of the json file the model will be saved in.
        path : str, default='./Models'
            The path at which the model will be saved.
        '''
        if not os.path.exists(path):
            print('Folder does not exist yet, creating it...')
            os.makedirs(path)

        if not file_name.endswith('.pck'):
            file_name += '.pck'

        # Writing the cpu version of the file to a pickle file
        with open(path + '/' + file_name, 'wb') as f:
            pickle.dump(self.cpu_model, f)


    @classmethod
    def load_from_file(cls,
                       file: str
                       ) -> 'Model':
        '''
        Function to load a MDP model from a pickle file. The json structure must contain the same items as in the constructor of this class.

        Parameters
        ----------
        file : str
            The file and path of the model to be loaded.
                
        Returns
        -------
        loaded_model : mdp.Model
            An instance of the loaded model.
        '''
        with open(file, 'rb') as openfile:
            loaded_model = pickle.load(openfile)

        return loaded_model


    @property
    def gpu_model(self) -> 'Model':
        '''
        The same model but on the GPU instead of the CPU. If already on the GPU, the current model object is returned.
        '''
        if self.is_on_gpu:
            return self
        
        assert gpu_support, "GPU Support is not available, try installing cupy..."
        
        if self._alt_model is None:
            log('Sending Model to GPU...')
            start = datetime.now()

            # Setting all the arguments of the new class and convert to cupy if numpy array
            new_model = super().__new__(self.__class__)
            for arg, val in self.__dict__.items():
                if isinstance(val, np.ndarray):
                    new_model.__setattr__(arg, cp.array(val))
                elif arg == 'rnd_state':
                    new_model.__setattr__(arg, cp.random.RandomState(self.seed))
                else:
                    new_model.__setattr__(arg, val)

            # GPU/CPU variables
            new_model.is_on_gpu = True
            new_model._alt_model = self
            self._alt_model = new_model
            
            duration = (datetime.now() - start).total_seconds()
            log(f'    > Done in {duration:.3f}s')

        return self._alt_model


    @property
    def cpu_model(self) -> 'Model':
        '''
        The same model but on the CPU instead of the GPU. If already on the CPU, the current model object is returned.
        '''
        if not self.is_on_gpu:
            return self
        
        assert gpu_support, "GPU Support is not available, try installing cupy..."

        if self._alt_model is None:
            log('Sending Model to CPU...')
            start = datetime.now()

            # Setting all the arguments of the new class and convert to numpy if cupy array
            new_model = super().__new__(self.__class__)
            for arg, val in self.__dict__.items():
                new_model.__setattr__(arg, cp.asnumpy(val) if isinstance(val, cp.ndarray) else val)
            
            # GPU/CPU variables
            new_model.is_on_gpu = False
            new_model._alt_model = self
            self._alt_model = new_model
            
            duration = (datetime.now() - start).total_seconds()
            log(f'    > Done in {duration:.3f}s')

        return self._alt_model