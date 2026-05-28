from __future__ import annotations

from typing import Callable

from olfactory_navigation.environment import Environment
from olfactory_navigation.agent import Agent
from olfactory_navigation.agents.model_based_util.environment_converter import exact_converter
from pomdp_toolkit import POMDP, ValueFunction, Belief, BeliefSet

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
    The technique was developed and described in the following article: Vergassola, M., Villermaux, E., & Shraiman, B. I. (2007). 'Infotaxis' as a strategy for searching without gradients.

    This agent does not require training; train(), save() and load() function are not implemented.


    Parameters
    ----------
    environment : Environment
        The olfactory environment to train the agent with.
    thresholds : float or list[float] or dict[str, float] or dict[str, list[float]], default = 3e-6
        The olfactory thresholds. If an odor cue above this threshold is detected, the agent detects it, else it does not.
        If a list of thresholds is provided, the agent should be able to detect |thresholds|+1 levels of odor.
        A dictionary of (list of) thresholds can also be provided when the environment is layered.
        In such case, the number of layers provided must match the environment's layers and their labels must match.
        The thresholds provided will be converted to an array where the levels start with -inf and end with +inf.
    space_aware : bool, default = False
        Whether the agent is aware of its own position in space.
        This is to be used in scenarios where, for example, the agent is an enclosed container and the source is the variable.
        Note: The observation array will have a different shape when returned to the update_state function!
    spacial_subdivisions : np.ndarray, optional
        How many spacial compartments the agent has to internally represent the space it lives in.
        By default, it will be as many as there are grid points in the environment.
    actions : dict or np.ndarray, optional
        The set of action available to the agent. It should match the type of environment (ie: if the environment has layers, it should contain a layer component to the action vector, and similarly for a third dimension).
        Else, a dict of strings and action vectors where the strings represent the action labels.
        If none is provided, by default, all unit steps in all cardinal directions are included and such for all layers (if the environment has layers.)
    name : str, optional
        A custom name to give the agent. If not provided is will be a combination of the class-name and the threshold.
    rng : int or np.random.Generator, default = np.random.default_rng()
        A seed for random generation or directly a numpy random generator.
    model : Model, optional
        A POMDP model to use to represent the olfactory environment.
        If not provided, the environment_converter parameter will be used.
    environment_converter : Callable, default = exact_converter
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
    trained : bool
        Whether or not the agent needs to be trained. If an agent doesnt need training this parameter is set to True by default.
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
    rng : np.random.Generator
        A random number generator.
    on_cpu : Agent
        An instance of the agent on the CPU. If it already is, it returns itself.
    on_gpu : Agent
        An instance of the agent on the GPU. If it already is, it returns itself.
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
                 spacial_subdivisions: np.ndarray = None,
                 actions: dict[str, np.ndarray] | np.ndarray = None,
                 name: str = None,
                 model: POMDP = None,
                 environment_converter: Callable = None,
                 use_reachability: bool = False,
                 **converter_parameters
                 ) -> None:
        super().__init__(
            environment = environment,
            thresholds = thresholds,
            space_aware = space_aware,
            spacial_subdivisions = spacial_subdivisions,
            actions = actions,
            name = name
        )

        # Converting the olfactory environment to a POMDP Model
        if model is not None:
            loaded_model = model
        elif callable(environment_converter):
            loaded_model = environment_converter(agent=self, **converter_parameters)
        else:
            # Using the exact converter
            loaded_model = exact_converter(agent=self)
        self.model:POMDP = loaded_model

        self.use_reachability = use_reachability

        # Status variables
        self.belief: BeliefSet = None
        self.action_played = None
        self.trained = True


    @property
    def on_gpu(self) -> Infotaxis_Agent:
        '''
        A version of the Agent on the GPU.
        If the agent is already on the GPU it returns itself, otherwise a new one is generated.
        '''
        # Check whether the agent is already on the gpu or not
        if self.is_on_gpu:
            return self

        assert gpu_support, "GPU support is not enabled, Cupy might need to be installed..."

        # Warn and overwrite alternate_version in case it already exists
        if self._alternate_version is None:
            # Generating a new instance
            cls = self.__class__
            gpu_agent = cls.__new__(cls)

            # Copying arguments to gpu
            for arg, val in self.__dict__.items():
                if isinstance(val, np.ndarray):
                    setattr(gpu_agent, arg, cp.array(val))
                elif isinstance(val, POMDP):
                    setattr(gpu_agent, arg, val.on_gpu)
                elif isinstance(val, BeliefSet) or isinstance(val, Belief):
                    setattr(gpu_agent, arg, val.on_gpu)
                else:
                    setattr(gpu_agent, arg, val)

            # Self reference instances
            self._alternate_version = gpu_agent
            gpu_agent._alternate_version = self
            gpu_agent.is_on_gpu = True

        return self._alternate_version


    @property
    def on_cpu(self) -> Infotaxis_Agent:
        '''
        A version of the Agent on the CPU.
        If the agent is already on the CPU it returns itself, otherwise a new one is generated.
        '''
        # Check whether the agent is already on the cpu or not
        if not self.is_on_gpu:
            return self

        # Check if an alternate version doesnt exists create a new one
        if self._alternate_version is None:
            # Generating a new instance
            cls = self.__class__
            cpu_agent = cls.__new__(cls)

            # Copying arguments to gpu
            for arg, val in self.__dict__.items():
                if isinstance(val, cp.ndarray):
                    setattr(cpu_agent, arg, cp.asnumpy(val))
                elif isinstance(val, POMDP):
                    setattr(cpu_agent, arg, val.on_cpu)
                elif isinstance(val, ValueFunction):
                    setattr(cpu_agent, arg, val.on_cpu)
                elif isinstance(val, BeliefSet) or isinstance(val, Belief):
                    setattr(cpu_agent, arg, val.on_cpu)
                else:
                    setattr(cpu_agent, arg, val)

            # Self reference instances
            self._alternate_version = cpu_agent
            cpu_agent._alternate_version = self
            cpu_agent.is_on_gpu = False

        return self._alternate_version


    def initialize_state(self,
                         n: int = 1,
                         belief: BeliefSet = None
                         ) -> None:
        '''
        To use an agent within a simulation, the agent's state needs to be initialized.
        The initialization consists in setting the agent's initial belief.
        Multiple agents can be used at once for simulations, for this reason, the belief parameter is a BeliefSet by default.

        Parameters
        ----------
        n : int, default = 1
            How many agents are to be used during the simulation.
        belief : BeliefSet, optional
            An optional set of beliefs to initialize the simulations with.
        '''
        if belief is None:
            self.belief = BeliefSet(self.model, [Belief(self.model) for _ in range(n)])
        else:
            assert len(belief) == n, f"The amount of beliefs provided ({len(belief)}) to initialize the state need to match the amount of stimulations to initialize (n={n})."

            if self.is_on_gpu and not belief.is_on_gpu:
                self.belief = belief.on_gpu
            elif not self.is_on_gpu and belief.is_on_gpu:
                self.belief = belief.on_cpu
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

        # Computing the entropies of the current beliefs
        current_entropy = self.belief.entropies

        # Compute the possible successors for each belief
        b_ao, provenance = self.belief.generate_all_successors(use_reachability=self.use_reachability,
                                                               raise_on_impossible_belief=False,
                                                               return_provenance=True)

        # Splitting the provenance
        provenance_b = provenance[:,0]
        provenance_a = provenance[:,1]
        provenance_o = provenance[:,2]

        # Computing entropies and P(o|b,a) to compute H
        b_ao_entropies = b_ao.entropies
        if self.use_reachability:
            b_ao_probs = xp.einsum('ns,nsr->n', self.belief.belief_array[provenance_b,:], self.model.reachable_transition_observation_table[:, provenance_a, :, provenance_o])
        else:
            b_ao_probs = xp.einsum('ns,nsp->n', self.belief.belief_array[provenance_b,:], self.model.transition_observation_table[:, provenance_a, :, provenance_o])

        b_ao_H = b_ao_probs * b_ao_entropies

        # Computing best_actions for each belief
        best_a = xp.zeros(len(self.belief), dtype=int)
        for b in xp.unique(provenance_b):
            b_entropy = current_entropy[b]

            current_best_delta_H = -xp.inf
            current_best_a = -1
            for a in self.model.actions:
                H_a = xp.sum(b_ao_H[(provenance_b == b) & (provenance_a == a)])
                delta_H = b_entropy - H_a

                if current_best_delta_H < delta_H:
                    current_best_delta_H = delta_H
                    current_best_a = a

            best_a[b] = current_best_a

        # Recording the action played
        self.action_played = best_a

        # Converting action indexes to movement vectors
        movemement_vector = self.action_set[best_a,:]

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
        # GPU support
        xp = np if not self.is_on_gpu else cp

        # Discretizing observations
        observation_ids = self.discretize_observations(observation=observation, action=action, source_reached=source_reached)

        # Update the set of beliefs
        self.belief, provenance = self.belief.update(actions = self.action_played,
                                                     observations = observation_ids,
                                                     raise_on_impossible_belief = False,
                                                     use_reachability = self.use_reachability,
                                                     return_provenance = True)

        # Check for failed updates
        update_successful = xp.isin(xp.arange(len(self.belief)), provenance[:,0])
        self.succeeded_update = update_successful

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
            filtered_simulations_to_kill = simulations_to_kill[self.succeeded_update]
            self.belief = BeliefSet(self.belief.model, self.belief.belief_array[~filtered_simulations_to_kill])
