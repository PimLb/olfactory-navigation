import inspect
import numpy as np
import os
import pickle
import shutil

from olfactory_navigation.environment import Environment

import numpy as np

gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')


class AgentState:
    def __init__(self) -> None:
        pass


class Agent:
    '''
    A generic agent class.

    It is meant to define the general structure for an agent meant to evolve in a environment of olfactory cues.
    To define such agent, a set of methods need to be implemented. This methods can be seperated into 3 categories:

    1. Training methods
    2. Simulation methods
    3. General methods

    The training methods are meant to train the agent before testing their performance in a simulation. A single method is needed for this:

    - train()

    The simulation methods are meant for the agent to make choices and receiving observations during a simulation. The following methods are required for this:

    - initialize_state(): This method is meant for the state of the agent(s) to be initialized before the simulation loop starts. The state of the agent can be an internal clock, a belief or something else arbitrary.
    - choose_action(): Here the agent(s) is asked to choose an action to play based on its internal state.
    - update_state(): Then, after the agent(s) has taken an action, the observation it makes along with whether he reached the source or not is returned to him using this method. This allows the agent to update its internal state.
    - kill(): Finally, the method asks for a set of agents to be terminated. The basic case happens when the agent reaches the source but it can also be asked to terminate if it has reached the end of the simulation without success.

    The general methods are methods to perform general actions with the agent. These methods are:

    - save(): To save the agent to long term storage.
    - load(): To load the agent from long term storage.
    - modify_environment(): To provide an equivalent agent with a different environment linked to it. If the agent has previously been trained, the trained components needs to be adapted to this new environment.
    - to_gpu(): To create an alternative version of the agent whether the array instances are stored on the GPU memory instead of the CPU memory.
    - to_cpu(): To create an alternative version of the agent whether the array instances are stored on the CPU memory instead of the GPU memory.

    For a user to implement an agent, the main methods to define are the Simulation methods! The training method is, as stated, optional, as some agent definitions do not require it.
    And the General methods all have some default behavior and are therefore only needed to be overwritten in specific cases.


    Parameters
    ----------
    environment : Environment
        The olfactory environment the agent is meant to evolve in.

    thresholds : float or list[float] or dict[str, float] or dict[str, list[float]], default=3e-6
        The olfactory thresholds. If an odor cue above this threshold is detected, the agent detects it, else it does not.
        If a list of thresholds is provided, he agent should be able to detect |thresholds|+1 levels of odor.
        A dictionary of (list of) thresholds can also be provided when the environment is layered.
        In such case, the number of layers provided must match the environment's layers and their labels must match.
        The thresholds provided will be converted to an array where the levels start with -inf and end with +inf.
    space_aware : bool, default=False
        Whether the agent is aware of it's own position in space.
        This is to be used in scenarios where, for example, the agent is an enclosed container and the source is the variable.
        Note: The observation array will have a different shape when returned to the update_state function!
    spacial_subdivisions : np.ndarray, optional
        How many spacial compartments the agent has to internally represent the space it lives in.
        By default, it will be as many as there are grid points in the environment.
    actions : dict or np.ndarray, optional
        The set of action available to the agent. It should match the type of environment (ie: if the environment has layers, it should contain a layer component to the action vector, and similarly for a third dimension).
        Else, a dict of strings and action vectors where the strings represent the action labels.
        If none is provided, by default, all unit movement vectors are included and shuch for all layers (if the environment has layers.)
    name : str, optional
        A custom name for the agent. If it is not provided it will be named like "<class_name>-thresh_<threshold>".
    seed : int, default=12131415
        For reproducible randomness.

    Attributes
    ----------
    environment : Environment
    thresholds : np.ndarray
        An array of the thresholds of detection, starting with -inf and ending with +inf.
        In the case of a 2D array of thresholds, the rows of thresholds apply to the different layers of the environment.
    space_aware : bool
    spacial_subdivisions : np.ndarray
    name : str
    action_set : np.ndarray
        The actions allowed of the agent. Formulated as movement vectors as [(layer,) (dz,) dy, dx].
    action_labels : list[str]
        The labels associated to the action vectors present in the action set.
    saved_at : str
        If the agent has been saved, the path at which it is saved is recorded in this variable.
    on_gpu : bool
        Whether the arrays are on the GPU memory or not. For this, the support for Cupy needs to be enabled and the agent needs to have been moved to the GPU using the to_gpu() function.
    class_name : str
        The name of the class of the agent.
    seed : int
        The seed used for the random operations (to allow for reproducability).
    rnd_state : np.random.RandomState
        The random state variable used to generate random values.
    cpu_version : Agent
        An instance of the agent on the CPU. If it already is, it returns itself.
    gpu_version : Agent
        An instance of the agent on the CPU. If it already is, it returns itself.
    '''
    def __init__(self,
                 environment: Environment,
                 thresholds: float | list[float] | dict[str, float] | dict[str, list[float]] = 3e-6,
                 space_aware: bool = False,
                 spacial_subdivisions: np.ndarray | None = None,
                 actions: dict[str, np.ndarray] | np.ndarray | None = None,
                 name: str | None = None,
                 seed: int = 12131415
                 ) -> None:
        self.environment = environment
        self.space_aware = space_aware

        # Handle thresholds
        if isinstance(thresholds, float):
            thresholds = [thresholds]

        if isinstance(thresholds, list):
            # Ensure -inf and +inf begin and end the thresholds list
            if -np.inf not in thresholds:
                thresholds = [-np.inf] + thresholds

            if np.inf not in thresholds:
                thresholds = thresholds + [np.inf]

            self.thresholds = np.array(thresholds)

        # Handle the case where a dict of thresholds is provided in the case of a layered environment and ensuring they are ordered.
        elif isinstance(thresholds, dict):
            assert self.environment.has_layers, "Thresholds can only be provided as a dict if the environment is layered."
            assert len(thresholds) == len(self.environment.layer_labels), "Thresholds must be given for each layers."

            if all(isinstance(layer_thresholds, list) for layer_thresholds in thresholds.values()):
                layer_thresholds_count = len(thresholds.values()[0])
                assert all(len(layer_thresholds) == layer_thresholds_count for layer_thresholds in thresholds.values()), "If provided as lists, the threshold lists must all be of the same length."
            else:
                assert all(isinstance(layer_threshold, float) for layer_threshold in thresholds.values()), "The thresholds provided in a dictionary must all be list or float, not a combination of both."

            # Processing layers of thresholds
            layer_thresholds_list = []
            for layer_label in enumerate(self.environment.layer_labels):
                assert layer_label in thresholds, f"Environment's layer [{layer_label}] was not matched to any layers of the environment."
                layer_thresholds = thresholds[layer_label]

                if not isinstance(layer_thresholds, list):
                    layer_thresholds = [layer_thresholds]

                # Ensure -inf and +inf begin and end the thresholds list
                if -np.inf not in layer_thresholds:
                    layer_thresholds = [-np.inf] + layer_thresholds

                if np.inf not in layer_thresholds:
                    layer_thresholds = layer_thresholds + [np.inf]

                layer_thresholds_list.append(layer_thresholds)

            # Building an array with the layers of thresholds
            self.thresholds = np.array(layer_thresholds_list)

        # Ensuring the thresholds are in ascending order
        self.thresholds.sort()

        # Spacial subdivisions
        if spacial_subdivisions is None:
            self.spacial_subdivisions = np.array(self.environment.shape)
        else:
            self.spacial_subdivisions = np.array(spacial_subdivisions)
        assert len(self.spacial_subdivisions) == self.environment.dimensions, "The amount of spacial divisions must match the amount of dimensions of the environment."

        # Mapping environment to spacial subdivisions
        env_shape = np.array(self.environment.shape)

        std_size = (env_shape / self.spacial_subdivisions).astype(int)
        overflows = (env_shape % self.spacial_subdivisions)

        cell_sizes = [(np.repeat(size, partitions) + np.array([np.floor(overflow / 2), *np.zeros(partitions-2), np.ceil(overflow / 2)])).astype(int)
                    for partitions, size, overflow in zip(self.spacial_subdivisions, std_size, overflows)]

        # Finding the edges of the cells and filling a grid with ids
        cell_edges = [np.concatenate(([0], np.cumsum(ax_sizes))) for ax_sizes in cell_sizes]

        lower_bounds = np.array([ax_arr.ravel() for ax_arr in np.meshgrid(*[bounds_arr[:-1] for bounds_arr in cell_edges], indexing='ij')]).T
        upper_bounds = np.array([ax_arr.ravel() for ax_arr in np.meshgrid(*[bounds_arr[1 :] for bounds_arr in cell_edges], indexing='ij')]).T

        self._environment_to_subdivision_mapping = np.full(self.environment.shape, -1)
        for i, (lower_b, upper_b) in enumerate(zip(lower_bounds, upper_bounds)):
            slices = [slice(ax_lower, ax_upper) for ax_lower, ax_upper in zip(lower_b, upper_b)]

            # Grid to cell mapping
            self._environment_to_subdivision_mapping[*slices] = i

        # Allowed actions
        self.action_labels = None
        if actions is None:
            if environment.dimensions == 2:
                self.action_set = np.array([
                    [-1,  0], # North
                    [ 0,  1], # East
                    [ 1,  0], # South
                    [ 0, -1]  # West
                ])
                self.action_labels = [
                    'North',
                    'East',
                    'South',
                    'West'
                ]
            elif environment.dimensions == 3:
                self.action_set = np.array([
                    [ 0, -1,  0], # North
                    [ 0,  0,  1], # East
                    [ 0,  1,  0], # South
                    [ 0,  0, -1], # West
                    [ 1,  0,  0], # Up
                    [-1,  0,  0]  # Down
                ])
                self.action_labels = [
                    'North',
                    'East',
                    'South',
                    'West',
                    'Up',
                    'Down'
                ]
            else: # ND
                self.action_set = np.zeros((2*environment.dimensions, environment.dimensions))
                self.action_labels = []
                for dim in range(environment.dimensions):
                    # Increase in dimension 'dim'
                    self.action_set[dim*2, -dim-1] = 1
                    self.action_labels.append(f'd{dim}+1')

                    # Decrease in dimension 'dim'
                    self.action_set[(dim*2) + 1, -dim-1] = -1
                    self.action_labels.append(f'd{dim}-1')

            # Layered
            if environment.has_layers:
                self.action_set = np.array([[layer, *action_vector] for layer in environment.layers for action_vector in self.action_set])
                self.action_labels = [f'l_{layer}_{action}' for  layer in environment.layer_labels for action in self.action_labels]

        # Actions provided as numpy array
        elif isinstance(actions, np.ndarray):
            self.action_set = actions
            self.action_labels = ['a_' + '_'.join([str(dim_a) for dim_a in action_vector]) for action_vector in self.action_set]

        # Actions provided as dict
        else:
            self.action_set = np.ndarray(list(actions.values()))
            self.action_labels = list(actions.keys())

        # Asertion that the shape of the actions set if right
        layered = 0 if not environment.has_layers else 1
        assert self.action_set.shape[1] == (layered + environment.dimensions), f"The shape of the action_set provided is not right. (Found {self.action_set.shape}; expected (., {layered + environment.dimensions}))"

        # setup name
        if name is None:
            self.name = self.class_name
            if len(self.thresholds.shape) == 2:
                thresh_string = '-'.join(['_'.join([f'l{row_i}'] + [str(item) for item in row]) for row_i, row in enumerate(self.thresholds[:,1:-1])])
            else:
                thresh_string = '_'.join([str(item) for item in self.thresholds])
            self.name += f'-tresholds-' + thresh_string
            self.name += '-space_aware' if self.space_aware else ''
        else:
            self.name = name

        # Other variables
        self.saved_at = None

        self.on_gpu = False
        self._alternate_version = None

        # random state
        self.seed = seed
        self.rnd_state = np.random.RandomState(seed = seed)


    @property
    def class_name(self):
        '''
        The name of the class of the agent.
        '''
        return self.__class__.__name__


    # ----------------
    # Training methods
    # ----------------
    def train(self) -> None:
        '''
        Optional function to train the agent in the olfactory environment it is in.
        This function is optional as some agents have some fixed behavior and therefore dont require training.
        '''
        raise NotImplementedError('The train function is not implemented, make an agent subclass to implement the method')


    # ------------------
    # Simulation methods
    # ------------------
    def initialize_state(self,
                         n: int = 1
                         ) -> None:
        '''
        Function to initialize the internal state of the agent(s) for the simulation process. The internal state can be concepts such as the "memory" or "belief" of the agent.
        The n parameter corresponds to how many "instances" need to instanciated. This is meant so that we work with a "group" of agents instead of individual instances.

        This is done with the purpose that the state of the group of agents be stored in (Numpy) arrays to allow vectorization instead of sequential loops.

        Parameters
        ----------
        n : int, default=1
            How many agents to initialize.
        '''
        raise NotImplementedError('The initialize_state function is not implemented, make an agent subclass to implement the method')


    def choose_action(self) -> np.ndarray:
        '''
        Function to allow for the agent(s) to choose an action to take based on its current state.

        It should return a 2D array of shape n by 2 (or 3, or 4 depending of whether the environment has layers and/or a 3rd dimension),
        where n is how many agents are to choose an action. It should be n 2D vectors of (the layer and) the change in the (z,) y, and x positions.

        Returns
        -------
        movement_vector : np.ndarray
            An array of n vectors in 2D space of the movement(s) the agent(s) will take.
        '''
        raise NotImplementedError('The choose_action function is not implemented, make an agent subclass to implement the method')


    def discretize_observations(self,
                                observation: np.ndarray,
                                action: np.ndarray,
                                source_reached: np.ndarray
                                ) -> np.ndarray:
        '''
        Function to convert a set of observations to discrete observation ids.
        It uses the olfaction thresholds of the agent to discretize the odor concentrations.

        In the case where the agent is also aware of it's own position in space, in which case the observation matrix is of size n by 1 + environment.dimensions,
        the agent first converts the points on the grid to position ids and then multiply the id by olfactory observation id.

        Parameters
        ----------
        observation : np.ndarray
            The observations the agent receives, in the case the agent is space_aware, the position point is also included.
        action: np.ndarray
            The action the agent did. This parameter is used in the particular case where the environment has layers and the odor thresholds are layer dependent.
        source_reached : np.ndarray
            A 1D array of boolean values signifying whether each agent reached or not the source.

        Returns
        -------
        discrete_observations: np.ndarray
            An integer array of discrete observations
        '''
        # GPU support
        xp = np if not self.on_gpu else cp

        n = len(observation)
        discrete_observations = xp.ones(n, dtype=int) * -1

        # Gather olfactory observation
        olfactory_observation = observation if not self.space_aware else observation[:,0]

        # Compute the amount of observations available
        observation_count = self.thresholds.shape[-1] - 1 # |thresholds| - 1 observation buckets

        # Handle the special case where we have a layered environment and different thresholds for these layers
        if self.environment.has_layers and len(self.thresholds.shape) == 2:
            layer_ids = action[:,0] # First column of actions is the layer
            action_layer_thresholds = self.thresholds[layer_ids]
            observation_ids = xp.argwhere((olfactory_observation[:,None] >= action_layer_thresholds[:,:-1]) & (olfactory_observation[:,None] < action_layer_thresholds[:,1:]))[:,1]
        else:
            # Setting observation ids
            observation_ids = xp.argwhere((olfactory_observation[:,None] >= self.thresholds[:-1][None,:]) & (olfactory_observation[:,None] < self.thresholds[1:][None,:]))[:,1]

        # If needed, multiply observation indices by position indices
        if not self.space_aware:
            discrete_observations = observation_ids
        else:
            position_clipped = xp.clip(observation[:,1:], a_min=0, a_max=(xp.array(self.environment.shape)-1))
            position_count = int(xp.prod(self.spacial_subdivisions))
            position_ids = self._environment_to_subdivision_mapping[*position_clipped.astype(int).T]

            # Add the amount of possible positions to the observation count
            observation_count *= position_count

            # Combine with observation ids
            discrete_observations = (observation_ids * position_count) + position_ids

        # Adding the goal observation
        discrete_observations[source_reached] = observation_count

        return discrete_observations


    def update_state(self,
                     action: np.ndarray,
                     observation: np.ndarray,
                     source_reached: np.ndarray
                     ) -> None | np.ndarray:
        '''
        Function to update the internal state(s) of the agent(s) based on the action(s) taken and the observation(s) received.
        The observations are then compared with the thresholds to decide whether something was sensed or not or to what level.

        Parameters
        ----------
        action : np.ndarray
            A 2D array of n movement vectors. If the environment is layered, the 1st component should be the layer.
        observation : np.ndarray
            A n by 1 (or 1 + environment.dimensions if space_aware) array of odor cues (float values) retrieved from the environment.
        source_reached : np.array
            A 1D array of boolean values signifying whether each agent reached or not the source.

        Returns
        -------
        update_successfull : np.ndarray, optional
            If nothing is returned, it means all the agent's state updates have been successfull.
            Else, a boolean np.ndarray of size n can be returned confirming for each agent whether the update has been successful or not.
        '''
        raise NotImplementedError('The update_state function is not implemented, make an agent subclass to implement the method')


    def kill(self,
             simulations_to_kill: np.ndarray
             ) -> None:
        '''
        Function to kill any agents that either reached the source or failed by not reaching the source before the horizon or failing to update its own state.
        The agents where the simulations_to_kill paramater is True have to removed from the list of agents.
        It is necessary because their reference will also be removed from the simulation loop. Therefore, if they are not removed, the array sizes will not match anymore.

        Parameters
        ----------
        simulations_to_kill : np.ndarray
            An array of size n containing boolean values of whether or not agent's simulations are terminated and therefore should be removed.
        '''
        raise NotImplementedError('The kill function is not implemented, make an agent subclass to implement the method')


    # ---------------
    # General methods
    # ---------------
    def save(self,
             folder: str | None = None,
             force: bool = False,
             save_environment: bool = False
             ) -> None:
        '''
        Function to save a trained agent to long term storage.
        By default, the agent is saved in its entirety using pickle.

        However, it is strongly advised to overwrite this method to only save save the necessary components of the agents in order to be able to load it and reproduce its behavior.
        For instance, if the agent is saved after the simulation is run, the state would also be saved within the pickle which is not wanted.

        Parameters
        ----------
        folder : str, optional
            The folder in which the agent's data should be saved.
        force : bool, default=False
            If the agent is already saved at the folder provided, the saving should fail.
            If the already saved agent should be overwritten, this parameter should be toggled to True.
        save_environment : bool, default=False
            Whether to save the agent's linked environment alongside the agent itself.
        '''
        if self.on_gpu:
            cpu_agent = self.to_cpu()
            cpu_agent.save(folder=folder, force=force, save_environment=save_environment)
            return

        # Adding env name to folder path
        if folder is None:
            folder = f'./Agent-{self.name}'
        else:
            folder += f'/Agent-{self.name}'

        # Checking the folder exists or creates it
        if not os.path.exists(folder):
            os.mkdir(folder)
        elif len(os.listdir(folder)) > 0:
            if force:
                shutil.rmtree(folder)
                os.mkdir(folder)
            else:
                raise Exception(f'{folder} is not empty. If you want to overwrite the saved agent, enable "force".')

        # Send self to pickle
        with open(folder + '/binary.pkl', 'wb') as f:
            pickle.dump(self, f)

        # Save environment in folder too if requested
        if save_environment:
            self.environment.save(folder=(folder + f'/Env-{self.environment.name}'))


    @classmethod
    def load(cls,
             folder: str
             ) -> 'Agent':
        '''
        Function to load a trained agent from long term storage.
        By default, as for the save function, it will load the agent from the folder assuming it is a pickle file.

        Parameters
        ----------
        folder : str
            The folder in which the agent was saved.

        Returns
        -------
        loaded_agent : Agent
            The agent loaded from the folder.
        '''
        from olfactory_navigation import agents

        for name, obj in inspect.getmembers(agents):
            if inspect.isclass(obj) and (name in folder) and issubclass(obj, cls) and (obj != cls):
                return obj.load(folder)

        # Default loading with pickle
        with open(folder + '/binary.pkl', 'rb') as f:
            return pickle.load(f)


    def modify_environment(self,
                           new_environment: Environment
                           ) -> 'Agent':
        '''
        Function to modify the environment of the agent.

        Note: By default, a new agent is created with the same thresholds and name but with a this new environment!
        If there are any trained elements to the agent, they are to be modified in this method to be adapted to this new environment.

        Parameters
        ----------
        new_environment : Environment
            The new environment to replace the agent in an equivalent agent.

        Returns
        -------
        modified_agent : Agent
            A new Agent whose environment has been replaced.
        '''
        # TODO: Fix this to account for other init parameters
        modified_agent = self.__class__(environment=new_environment,
                                        thresholds=self.thresholds,
                                        name=self.name)
        return modified_agent


    def to_gpu(self) -> 'Agent':
        '''
        Function to send the numpy arrays of the agent to the gpu.
        It returns a new instance of the Agent class with the arrays on the gpu.

        Returns
        -------
        gpu_agent : Agent
            A new environment instance where the arrays are on the gpu memory.
        '''
        # Check whether the agent is already on the gpu or not
        if self.on_gpu:
            return self

        # Warn and overwrite alternate_version in case it already exists
        if self._alternate_version is not None:
            print('[warning] A GPU instance already existed and is being recreated.')
            self._alternate_version = None

        assert gpu_support, "GPU support is not enabled, Cupy might need to be installed..."

        # Generating a new instance
        cls = self.__class__
        gpu_agent = cls.__new__(cls)

        # Copying arguments to gpu
        for arg, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                setattr(gpu_agent, arg, cp.array(val))
            elif arg == 'rnd_state':
                setattr(gpu_agent, arg, cp.random.RandomState(self.seed))
            else:
                setattr(gpu_agent, arg, val)

        # Self reference instances
        self._alternate_version = gpu_agent
        gpu_agent._alternate_version = self

        gpu_agent.on_gpu = True
        return gpu_agent


    def to_cpu(self) -> 'Agent':
        '''
        Function to send the numpy arrays of the agent to the cpu.
        It returns a new instance of the Agent class with the arrays on the cpu.

        Returns
        -------
        cpu_agent : Agent
            A new environment instance where the arrays are on the cpu memory.
        '''
        # Check whether the agent is already on the cpu or not
        if not self.on_gpu:
            return self

        if self._alternate_version is not None:
            print('[warning] A CPU instance already existed and is being recreated.')
            self._alternate_version = None

        # Generating a new instance
        cls = self.__class__
        cpu_agent = cls.__new__(cls)

        # Copying arguments to gpu
        for arg, val in self.__dict__.items():
            if isinstance(val, cp.ndarray):
                setattr(cpu_agent, arg, cp.asnumpy(val))
            elif arg == 'rnd_state':
                setattr(cpu_agent, arg, np.random.RandomState(self.seed))
            else:
                setattr(cpu_agent, arg, val)

        # Self reference instances
        self._alternate_version = cpu_agent
        cpu_agent._alternate_version = self

        cpu_agent.on_gpu = True
        return cpu_agent


    @property
    def gpu_version(self):
        '''
        A version of the Agent on the GPU.
        If the agent is already on the GPU it returns itself, otherwise the to_gpu function is called to generate a new one.
        '''
        if self.on_gpu:
            return self
        else:
            if self._alternate_version is not None: # Check if an alternate version already exists
                return self._alternate_version
            else: # Generate an alternate version on the gpu
                return self.to_gpu()


    @property
    def cpu_version(self):
        '''
        A version of the Agent on the CPU.
        If the agent is already on the CPU it returns itself, otherwise the to_cpu function is called to generate a new one.
        '''
        if not self.on_gpu:
            return self
        else:
            if self._alternate_version is not None: # Check if an alternate version already exists
                return self._alternate_version
            else: # Generate an alternate version on the cpu
                return self.to_cpu()