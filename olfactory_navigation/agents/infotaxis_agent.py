import warnings

from typing import Callable

from olfactory_navigation.environment import Environment
from olfactory_navigation.agent import Agent
from olfactory_navigation.agents.model_based_util.pomdp import Model
from olfactory_navigation.agents.model_based_util.belief import Belief, BeliefSet
from olfactory_navigation.agents.model_based_util.environment_converter import exact_converter

import numpy as np
gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')


class Infotaxis_Agent(Agent):
    '''
    An agent following the Infotaxis principle.
    It is a Model-Based approach that aims to make steps towards where the agent has the greatest likelihood to minimize the entropy of the belief.
    The belief is (as for the PBVI agent) a probability distribution over the state space of how much the agent is to be confident in each state.
    The technique was developped and described in the following article: Vergassola, M., Villermaux, E., & Shraiman, B. I. (2007). 'Infotaxis' as a strategy for searching without gradients.

    It does not need to be trained to the train(), save() and load() function are not implemented.


    Parameters
    ----------
    environment : Environment
        The olfactory environment to train the agent with.
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
        A custom name to give the agent. If not provided is will be a combination of the class-name and the threshold.
    seed : int, default=12131415
        For reproducible randomness.
    model : Model, optional
        A POMDP model to use to represent the olfactory environment.
        If not provided, the environment_converter parameter will be used.
    environment_converter : Callable, default=exact_converter
        A function to convert the olfactory environment instance to a POMDP Model instance.
        By default, we use an exact convertion that keeps the shape of the environment to make the amount of states of the POMDP Model.
        This parameter will be ignored if the model parameter is provided.
    converter_parameters : dict, optional
        A set of additional parameters to be passed down to the environment converter.

    Attributes
    ---------
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
    model : pomdp.Model
        The environment converted to a POMDP model using the "from_environment" constructor of the pomdp.Model class.
    saved_at : str
        The place on disk where the agent has been saved (None if not saved yet).
    on_gpu : bool
        Whether the agent has been sent to the gpu or not.
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
    belief : BeliefSet
        Used only during simulations.
        Part of the Agent's status. Where the agent believes he is over the state space.
        It is a list of n belief points based on how many simulations are running at once.
    action_played : list[int]
        Used only during simulations.
        Part of the Agent's status. Records what action was last played by the agent.
        A list of n actions played based on how many simulations are running at once.
    '''
    def __init__(self,
                 environment: Environment,
                 thresholds: float | list[float] | dict[str, float] | dict[str, list[float]] = 3e-6,
                 space_aware: bool = False,
                 spacial_subdivisions: np.ndarray | None = None,
                 actions: dict[str, np.ndarray] | np.ndarray | None = None,
                 name: str | None=None,
                 seed: int = 12131415,
                 model: Model | None = None,
                 environment_converter: Callable | None = None,
                 **converter_parameters
                 ) -> None:
        super().__init__(
            environment = environment,
            thresholds = thresholds,
            space_aware = space_aware,
            spacial_subdivisions = spacial_subdivisions,
            actions = actions,
            name = name,
            seed = seed
        )

        # Converting the olfactory environment to a POMDP Model
        if model is not None:
            loaded_model = model
        elif callable(environment_converter):
            loaded_model = environment_converter(agent=self, **converter_parameters)
        else:
            # Using the exact converter
            loaded_model = exact_converter(agent=self)
        self.model:Model = loaded_model

        # Status variables
        self.belief = None
        self.action_played = None


    def to_gpu(self) -> Agent:
        '''
        Function to send the numpy arrays of the agent to the gpu.
        It returns a new instance of the Agent class with the arrays on the gpu.

        Returns
        -------
        gpu_agent
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
            elif isinstance(val, Model):
                setattr(gpu_agent, arg, val.gpu_model)
            elif isinstance(val, BeliefSet) or isinstance(val, Belief):
                setattr(gpu_agent, arg, val.to_gpu())
            else:
                setattr(gpu_agent, arg, val)

        # Self reference instances
        self._alternate_version = gpu_agent
        gpu_agent._alternate_version = self

        gpu_agent.on_gpu = True
        return gpu_agent


    def initialize_state(self,
                         n: int = 1,
                         belief: BeliefSet | None = None
                         ) -> None:
        '''
        To use an agent within a simulation, the agent's state needs to be initialized.
        The initialization consists of setting the agent's initial belief.
        Multiple agents can be used at once for simulations, for this reason, the belief parameter is a BeliefSet by default.

        Parameters
        ----------
        n : int, default=1
            How many agents are to be used during the simulation.
        belief : BeliefSet, optional
            An optional set of beliefs to initialize the simulations with.
        '''
        if belief is None:
            self.belief = BeliefSet(self.model, [Belief(self.model) for _ in range(n)])
        else:
            assert len(belief) == n, f"The amount of beliefs provided ({len(belief)}) to initialize the state need to match the amount of stimulations to initialize (n={n})."

            if self.on_gpu and not belief.is_on_gpu:
                self.belief = belief.to_gpu()
            elif not self.on_gpu and belief.is_on_gpu:
                self.belief = belief.to_cpu()
            else:
                self.belief = belief


    def choose_action(self) -> np.ndarray:
        '''
        Function to let the agent or set of agents choose an action based on their current belief.
        Following the Infotaxis principle, it will choose an action that will minimize the sum of next entropies.

        Returns
        -------
        movement_vector : np.ndarray
            A single or a list of actions chosen by the agent(s) based on their belief.
        '''
        xp = np if not self.on_gpu else cp

        n = len(self.belief)

        best_entropy = xp.ones(n) * -1
        best_action = xp.ones(n, dtype=int) * -1

        current_entropy = self.belief.entropies

        for a in self.model.actions:
            total_entropy = xp.zeros(n)

            for o in self.model.observations:
                b_ao = self.belief.update(actions=xp.ones(n, dtype=int)*a,
                                           observations=xp.ones(n, dtype=int)*o,
                                           throw_error=False)

                # Computing entropy
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    b_ao_entropy = b_ao.entropies

                b_prob = xp.dot(self.belief.belief_array, xp.sum(self.model.reachable_transitional_observation_table[:,a,o,:], axis=1))

                total_entropy += (b_prob * (current_entropy - b_ao_entropy))

            # Checking if action is superior to previous best
            superiority_mask = best_entropy < total_entropy
            best_action[superiority_mask] = a
            best_entropy[superiority_mask] = total_entropy[superiority_mask]

        # Recording the action played
        self.action_played = best_action

        # Converting action indexes to movement vectors
        movemement_vector = self.action_set[best_action,:]

        return movemement_vector


    def update_state(self,
                     action: np.ndarray,
                     observation: np.ndarray,
                     source_reached: np.ndarray
                     ) -> None | np.ndarray:
        '''
        Function to update the internal state(s) of the agent(s) based on the previous action(s) taken and the observation(s) received.

        Parameters
        ----------
        action : np.ndarray
            A 2D array of n movement vectors. If the environment is layered, the 1st component should be the layer.
        observation : np.ndarray
            The observation(s) the agent(s) made.
        source_reached : np.ndarray
            A boolean array of whether the agent(s) have reached the source or not.

        Returns
        -------
        update_successfull : np.ndarray, optional
            If nothing is returned, it means all the agent's state updates have been successfull.
            Else, a boolean np.ndarray of size n can be returned confirming for each agent whether the update has been successful or not.
        '''
        assert self.belief is not None, "Agent was not initialized yet, run the initialize_state function first"


        # Discretizing observations
        observation_ids = self.discretize_observations(observation=observation, action=action, source_reached=source_reached)

        # Update the set of belief
        self.belief = self.belief.update(actions=self.action_played, observations=observation_ids)

        # Check for failed updates
        update_successful = (self.belief.belief_array.sum(axis=1) != 0.0)

        return update_successful


    def kill(self,
             simulations_to_kill: np.ndarray
             ) -> None:
        '''
        Function to kill any simulations that have not reached the source but can't continue further

        Parameters
        ----------
        simulations_to_kill : np.ndarray
            A boolean array of the simulations to kill.
        '''
        if all(simulations_to_kill):
            self.belief = None
        else:
            self.belief = BeliefSet(self.belief.model, self.belief.belief_array[~simulations_to_kill])