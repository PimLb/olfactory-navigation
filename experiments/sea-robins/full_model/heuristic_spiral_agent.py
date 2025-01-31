import sys
sys.path.append('../../..')

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


class HeuristicSpiralAgent(Agent):
    def __init__(self,
                 environment: Environment,
                 thresholds: float | list[float] | dict[str, float] | dict[str, list[float]] = 3e-6,
                 space_aware: bool = False,
                 spacial_subdivisions: np.ndarray | None = None,
                 actions: dict[str, np.ndarray] | np.ndarray | None = None,
                 name: str | None=None,
                 seed: int = 12131415,
                 model: Model | None = None,
                 spiral_step: int = 1,
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

        self.spiral_step = spiral_step

        self.clock = None
        self.spiral_width = None
        self.spiral_direction = None
        self.flip_step = None
        self.steps_before_turnaround = None
        self.ignore_invalid_steps = None


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

        n = len(self.belief)

        xp = np if not self.on_gpu else cp

        # Spiral initialization
        self.clock = xp.ones(n, dtype=int)
        self.action_played = xp.random.randint(0, len(self.action_set), n) # Randomly choose an initial action
        self.spiral_width = xp.zeros(n, dtype=int)
        self.spiral_direction = (xp.random.randint(0, 2, n) * 2) - 1 # Randomly choose a direction (+1 = clockwise, -1 = counterclockwise)
        self.flip_step = xp.zeros(n, dtype=int)
        self.steps_before_turnaround = xp.zeros(n, dtype=int)
        self.ignore_invalid_steps = xp.zeros(n, dtype=int)

        self.action_played = xp.ones(n, dtype=int) # Start with the action 1 (right)
        self.spiral_direction = xp.ones(n, dtype=int) # Start with the direction 1 (clockwise)



    def choose_action(self) -> np.ndarray:
        '''
        Choose actions to follow a spiral trajectory. If it reaches a un-believable state, it will turn around.

        Returns
        -------
        movement_vector : np.ndarray
            A single or a list of actions chosen by the agent(s) based on their belief.
        '''
        xp = np if not self.on_gpu else cp

        next_action = self.action_played.copy() # Copy previous action

        forbidden_turn = (self.steps_before_turnaround > 0) & (self.steps_before_turnaround < self.spiral_step)

        # If spiral width is reached, change direction along the direction
        reached_width = (self.clock % (self.spiral_width) == 0) & ~forbidden_turn
        next_action[reached_width] = (next_action[reached_width] + self.spiral_direction[reached_width]) % len(self.action_set)

        # If 2 widths are reached, increase the width
        reached_double_width = (self.clock % ((self.spiral_width * 2)) == 0) & ~forbidden_turn
        self.spiral_width[reached_double_width] += self.spiral_step
        self.clock[reached_double_width] = 0

        # Check the available actions
        valid_actions = xp.zeros((len(self.action_set), len(self.belief)), dtype=bool)
        for a in range(len(self.action_set)):
            action_vect = xp.full(len(self.belief), a, dtype=int)
            tot_beliefs = xp.zeros(self.belief.belief_array.shape)

            for o in self.model.observations:
                observation_vect = xp.full(len(self.belief), o, dtype=int)
                next_beliefs = self.belief.update(actions=action_vect, observations=observation_vect, throw_error=False)
                tot_beliefs += next_beliefs.belief_array

            valid_actions[a] = xp.any(tot_beliefs[:,self.model.end_states] > 0, axis=1)

        # Deal with invalid actions
        next_action_invalid = ~(self.ignore_invalid_steps > 0) & ~valid_actions[next_action, xp.arange(len(self.belief))] # If ignore stick with the next action

        opposite_action = (next_action + (self.spiral_direction * -1)) % len(self.action_set)
        opposite_action_valid = valid_actions[opposite_action, xp.arange(len(self.belief))]

        turn_action = (next_action + self.spiral_direction) % len(self.action_set)
        turn_action_valid = valid_actions[turn_action, xp.arange(len(self.belief))]

        turn_around_available = next_action_invalid & opposite_action_valid & (self.flip_step == 0)
        turn_around_cut = next_action_invalid & (self.flip_step > 0) & turn_action_valid
        back_on_its_track = next_action_invalid & ~opposite_action_valid & ~turn_around_cut

        # option 1) Apply turnaround
        self.spiral_direction[turn_around_available] *= -1
        next_action[turn_around_available] = (next_action[turn_around_available] + self.spiral_direction[turn_around_available]) % len(self.action_set)

        self.flip_step[turn_around_available] = self.clock[turn_around_available] % self.spiral_width[turn_around_available]
        self.clock[turn_around_available] = xp.where(~(self.clock // self.spiral_width == 0)[turn_around_available], -self.spiral_step, (self.spiral_width[turn_around_available] - self.spiral_step))
        self.steps_before_turnaround[turn_around_available] = 2 * self.spiral_step

        # Apply cut turnaround cut short
        self.steps_before_turnaround[turn_around_cut] = self.spiral_step
        next_action[turn_around_cut] = turn_action[turn_around_cut]
        self.spiral_width[turn_around_cut] += xp.where(~(self.clock // self.spiral_width == 0)[turn_around_cut], self.spiral_step, 0)

        # Apply the flip steps to the clock to finish the turnaround
        finished_turnaround = (turn_around_cut | (~next_action_invalid & (self.flip_step > 0) & (self.steps_before_turnaround == self.spiral_step)))
        self.clock[finished_turnaround] = (self.spiral_width[finished_turnaround] * 2) - self.flip_step[finished_turnaround] - self.spiral_step
        self.flip_step[finished_turnaround] = 0

        # option 2) Apply the back-on-its-track manoeuver
        self.spiral_direction[back_on_its_track] *= -1
        next_action[back_on_its_track] = (next_action[back_on_its_track] + 2) % len(self.action_set)
        self.clock[back_on_its_track] = (self.spiral_width[back_on_its_track] * 2) - self.clock[back_on_its_track]
        self.spiral_width[back_on_its_track] += self.spiral_step
        self.ignore_invalid_steps[back_on_its_track] = self.spiral_width[back_on_its_track]

        # Increase the clock
        self.clock += 1
        self.steps_before_turnaround[self.steps_before_turnaround > 0] -= 1
        self.ignore_invalid_steps[self.ignore_invalid_steps > 0] -= 1

        # Recording the action played
        self.action_played = next_action

        # Converting action indexes to movement vectors
        movemement_vector = self.action_set[next_action,:]

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

        self.clock = self.clock[~simulations_to_kill]
        self.action_played = self.action_played[~simulations_to_kill]
        self.spiral_width = self.spiral_width[~simulations_to_kill]
        self.spiral_direction = self.spiral_direction[~simulations_to_kill]
        self.flip_step = self.flip_step[~simulations_to_kill]
        self.steps_before_turnaround = self.steps_before_turnaround[~simulations_to_kill]
        self.ignore_invalid_steps = self.ignore_invalid_steps[~simulations_to_kill]
