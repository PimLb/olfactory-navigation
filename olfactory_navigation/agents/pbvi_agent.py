import cv2
import inspect
import json
import os
import shutil

from datetime import datetime
from typing import Callable, Self

from olfactory_navigation.environment import Environment
from olfactory_navigation.agent import Agent
from olfactory_navigation.agents.model_based_util.environment_converter import exact_converter
from olfactory_navigation.simulation import SimulationHistory
from pomdp_toolkit import POMDP, ValueFunction, Belief, BeliefSet

import numpy as np

gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')


class TrainingHistory:
    '''
    Class to represent the history of a solver for a POMDP solver.
    It has mainly the purpose to have visualizations for the solution, belief set and the whole solving history.
    The visualizations available are:
        - Belief set plot
        - Solution plot
        - Video of value function and belief set evolution over training.


    Parameters
    ----------
    tracking_level : int
        The tracking level of the solver.
    model : Model
        The model the solver has solved.
    gamma : float
        The gamma parameter used by the solver (learning rate).
    eps : float
        The epsilon parameter used by the solver (covergence bound).
    expand_append : bool
        Whether the expand function appends new belief points to the belief set of reloads it all.
    initial_value_function : ValueFunction
        The initial value function the solver will use to start the solving process.
    initial_belief_set : BeliefSet
        The initial belief set the solver will use to start the solving process.

    Attributes
    ----------
    tracking_level : int
    model : Model
    gamma : float
    eps : float
    expand_append : bool
    run_ts : datetime
        The time at which the SolverHistory object was instantiated which is assumed to be the start of the solving run.
    expansion_times : list[float]
        A list of recorded times of the expand function.
    backup_times : list[float]
        A list of recorded times of the backup function.
    alpha_vector_counts : list[int]
        A list of recorded alpha vector count making up the value function over the solving process.
    beliefs_counts : list[int]
        A list of recorded belief count making up the belief set over the solving process.
    value_function_changes : list[float]
        A list of recorded value function changes (the maximum changed value between 2 value functions).
    value_functions : list[ValueFunction]
        A list of recorded value functions.
    belief_sets : list[BeliefSet]
        A list of recorded belief sets.
    solution : ValueFunction
    explored_beliefs : BeliefSet
    '''
    def __init__(self,
                 tracking_level: int,
                 model: POMDP,
                 gamma: float,
                 eps: float,
                 expand_append: bool,
                 initial_value_function: ValueFunction,
                 initial_belief_set: BeliefSet
                 ):

        self.tracking_level = tracking_level
        self.model = model
        self.gamma = gamma
        self.eps = eps
        self.run_ts = datetime.now()

        self.expand_append = expand_append

        # Time tracking
        self.expansion_times = []
        self.backup_times = []
        self.pruning_times = []

        # Value function and belief set sizes tracking
        self.alpha_vector_counts = []
        self.beliefs_counts = []
        self.prune_counts = []

        if self.tracking_level >= 1:
            self.alpha_vector_counts.append(len(initial_value_function))
            self.beliefs_counts.append(len(initial_belief_set))

        # Value function and belief set tracking
        self.belief_sets = []
        self.value_functions = []
        self.value_function_changes = []

        if self.tracking_level >= 2:
            self.belief_sets.append(initial_belief_set)
            self.value_functions.append(initial_value_function)


    @property
    def solution(self) -> ValueFunction:
        '''
        The last value function of the solving process.
        '''
        assert self.tracking_level >= 2, "Tracking level is set too low, increase it to 2 if you want to have value function tracking as well."
        return self.value_functions[-1]


    @property
    def explored_beliefs(self) -> BeliefSet:
        '''
        The final set of beliefs explored during the solving.
        '''
        assert self.tracking_level >= 2, "Tracking level is set too low, increase it to 2 if you want to have belief sets tracking as well."
        return self.belief_sets[-1]


    def add_expand_step(self,
                        expansion_time: float,
                        belief_set: BeliefSet
                        ) -> None:
        '''
        Function to add an expansion step in the simulation history by the explored belief set the expand function generated.

        Parameters
        ----------
        expansion_time : float
            The time it took to run a step of expansion of the belief set. (Also known as the exploration step.)
        belief_set : BeliefSet
            The belief set used for the Update step of the solving process.
        '''
        if self.tracking_level >= 1:
            self.expansion_times.append(float(expansion_time))
            self.beliefs_counts.append(len(belief_set))

        if self.tracking_level >= 2:
            self.belief_sets.append(belief_set.on_cpu)


    def add_backup_step(self,
                        backup_time: float,
                        value_function_change: float,
                        value_function: ValueFunction
                        ) -> None:
        '''
        Function to add a backup step in the simulation history by recording the value function the backup function generated.

        Parameters
        ----------
        backup_time : float
            The time it took to run a step of backup of the value function. (Also known as the value function update.)
        value_function_change : float
            The change between the value function of this iteration and of the previous iteration.
        value_function : ValueFunction
            The value function resulting after a step of the solving process.
        '''
        if self.tracking_level >= 1:
            self.backup_times.append(float(backup_time))
            self.alpha_vector_counts.append(len(value_function))
            self.value_function_changes.append(float(value_function_change))

        if self.tracking_level >= 2:
            self.value_functions.append(value_function.on_cpu)


    def add_prune_step(self,
                       prune_time: float,
                       alpha_vectors_pruned: int
                       ) -> None:
        '''
        Function to add a prune step in the simulation history by recording the amount of alpha vectors that were pruned by the pruning function and how long it took.

        Parameters
        ----------
        prune_time : float
            The time it took to run the pruning step.
        alpha_vectors_pruned : int
            How many alpha vectors were pruned.
        '''
        if self.tracking_level >= 1:
            self.pruning_times.append(prune_time)
            self.prune_counts.append(alpha_vectors_pruned)


    @property
    def summary(self) -> str:
        '''
        A summary as a string of the information recorded.
        '''
        summary_str =  f'Summary of Point Based Value Iteration run'
        summary_str += f'\n  - Model: {self.model.state_count} state, {self.model.action_count} action, {self.model.observation_count} observations'
        summary_str += f'\n  - Converged or stopped after {len(self.expansion_times)} expansion steps and {len(self.backup_times)} backup steps.'

        if self.tracking_level >= 1:
            summary_str += f'\n  - Resulting value function has {self.alpha_vector_counts[-1]} alpha vectors.'
            summary_str += f'\n  - Converged in {(sum(self.expansion_times) + sum(self.backup_times)):.4f}s'
            summary_str += f'\n'

            summary_str += f'\n  - Expand function took on average {sum(self.expansion_times) / len(self.expansion_times):.4f}s '
            if self.expand_append:
                summary_str += f'and yielded on average {sum(np.diff(self.beliefs_counts)) / len(self.beliefs_counts[1:]):.2f} beliefs per iteration.'
            else:
                summary_str += f'and yielded on average {sum(self.beliefs_counts[1:]) / len(self.beliefs_counts[1:]):.2f} beliefs per iteration.'
            summary_str += f' ({np.sum(np.divide(self.expansion_times, self.beliefs_counts[1:])) / len(self.expansion_times):.4f}s/it/belief)'

            summary_str += f'\n  - Backup function took on average {sum(self.backup_times) /len(self.backup_times):.4f}s '
            summary_str += f'and yielded on average {np.average(np.diff(self.alpha_vector_counts)):.2f} alpha vectors per iteration.'
            summary_str += f' ({np.sum(np.divide(self.backup_times, self.alpha_vector_counts[1:])) / len(self.backup_times):.4f}s/it/alpha)'

            summary_str += f'\n  - Pruning function took on average {sum(self.pruning_times) /len(self.pruning_times):.4f}s '
            summary_str += f'and yielded on average prunings of {sum(self.prune_counts) / len(self.prune_counts):.2f} alpha vectors per iteration.'

        return summary_str


class PBVI_Agent(Agent):
    '''
    A generic Point-Based Value Iteration based agent. It relies on Model-Based reinforcement learning as described in: Pineau J. et al, Point-based value iteration: An anytime algorithm for POMDPs
    The training consist in two steps:

    - Expand: Where belief points are explored based on the some strategy (to be defined by subclasses).

    - Backup: Using the generated belief points, the value function is updated.

    The belief points are probability distributions over the state space and are therefore vectors of |S| elements.

    Actions are chosen based on a value function. A value function is a set of alpha vectors of dimentionality |S|.
    Each alpha vector is associated to a single action but multiple alpha vectors can be associated to the same action.
    To choose an action at a given belief point, a dot product is taken between each alpha vector and the belief point and the action associated with the highest result is chosen.

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
        Whether the agent is aware of it's own position in space.
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
    seed : int, default = 12131415
        For reproducible randomness.
    model : POMDP, optional
        A POMDP model to use to represent the olfactory environment.
        If not provided, the environment_converter parameter will be used.
    use_reachability : bool, default = False
            Whether or not to use the reachable states as a shortcut to find the posterior state(s).
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
    model : POMDP
        The environment converted to a POMDP model using the "from_environment" constructor of the POMDP class.
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
    on_cpu : PBVI_Agent
        An instance of the agent on the CPU. If it already is, it returns itself.
    on_gpu : PBVI_Agent
        An instance of the agent on the GPU. If it already is, it returns itself.
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
    def __init__(self,
                 environment: Environment,
                 thresholds: float | list[float] | dict[str, float] | dict[str, list[float]] = 3e-6,
                 space_aware: bool = False,
                 spacial_subdivisions: np.ndarray = None,
                 actions: dict[str, np.ndarray] | np.ndarray = None,
                 name: str = None,
                 seed: int = 12131415,
                 model: POMDP = None,
                 use_reachability: bool = False,
                 environment_converter: Callable = None,
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
        self.model:POMDP = loaded_model

        self.use_reachability = use_reachability

        # Trainable variables
        self.trained_at = None
        self.value_function: ValueFunction = None

        # Status variables
        self.belief: BeliefSet = None
        self.action_played = None
        self.succeeded_update = None


    @property
    def on_gpu(self) -> Self:
        '''
        A version of the Agent on the GPU.
        If the agent is already on the GPU it returns itself, otherwise a new one is generated.
        '''
        # Check whether the agent is already on the gpu or not
        if self.is_on_gpu:
            return self

        assert gpu_support, "GPU support is not enabled, Cupy might need to be installed..."

        # Check if an alternate version doesnt exists create a new one
        if self._alternate_version is None:
            # Generating a new instance
            cls = self.__class__
            gpu_agent = cls.__new__(cls)

            # Copying arguments to gpu
            for arg, val in self.__dict__.items():
                if isinstance(val, np.ndarray):
                    setattr(gpu_agent, arg, cp.array(val))
                elif arg == 'rnd_state':
                    setattr(gpu_agent, arg, cp.random.RandomState(self.seed))
                elif isinstance(val, POMDP):
                    setattr(gpu_agent, arg, val.on_gpu)
                elif isinstance(val, ValueFunction):
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
    def on_cpu(self) -> Self:
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
                elif arg == 'rnd_state':
                    setattr(cpu_agent, arg, np.random.RandomState(self.seed))
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


    def save(self,
             folder: str = None,
             force: bool = False,
             save_environment: bool = False
             ) -> None:
        '''
        The save function for PBVI Agents consists in recording the value function after the training.
        It saves the agent in a folder with the name of the agent (class name + training timestamp).
        In this folder, there will be the metadata of the agent (all the attributes) in a json format and the value function.

        Optionally, the environment can be saved too to be able to load it alongside the agent for future reuse.
        If the agent has already been saved, the saving will not happen unless the force parameter is toggled.

        Parameters
        ----------
        folder : str, optional
            The folder under which to save the agent (a subfolder will be created under this folder).
            The agent will therefore be saved at <folder>/Agent-<agent_name> .
            By default the current folder is used.
        force : bool, default = False
            Whether to overwrite an already saved agent with the same name at the same path.
        save_environment : bool, default = False
            Whether to save the environment data along with the agent.
        '''
        assert self.trained_at is not None, "The agent is not trained, there is nothing to save."

        # GPU support
        if self.is_on_gpu:
            self.on_cpu.save(folder=folder, force=force, save_environment=save_environment)
            return

        # Adding env name to folder path
        if folder is None:
            folder = f'./Agent-{self.name}'
        else:
            folder += '/Agent-' + self.name

        # Checking the folder exists or creates it
        if not os.path.exists(folder):
            os.mkdir(folder)
        elif len(os.listdir(folder)):
            if force:
                shutil.rmtree(folder)
                os.mkdir(folder)
            else:
                raise Exception(f'{folder} is not empty. If you want to overwrite the saved model, enable "force".')

        # If requested save environment
        if save_environment:
            self.environment.save(folder=folder)

        # TODO: Add MODEL to save function
        # Generating the metadata arguments dictionary
        arguments = {}
        arguments['name'] = self.name
        arguments['class'] = self.class_name
        if len(self.thresholds.shape) == 2:
            arguments['thresholds'] = {layer_lab: layer_thresholds for layer_lab, layer_thresholds in zip(self.environment.layer_labels, self.thresholds.tolist())}
        else:
            arguments['thresholds'] = self.thresholds.tolist()
        arguments['environment_name'] = self.environment.name
        arguments['environment_saved_at'] = self.environment.saved_at
        arguments['space_aware'] = self.space_aware
        arguments['spacial_subdivisions'] = self.spacial_subdivisions.tolist()
        arguments['action_labels'] = self.action_labels
        arguments['action_set'] = self.action_set.tolist()
        arguments['trained_at'] = self.trained_at
        arguments['seed'] = self.seed

        # Output the arguments to a METADATA file
        with open(folder + '/METADATA.json', 'w') as json_file:
            json.dump(arguments, json_file, indent=4)

        # Save value function
        self.value_function.save(folder=folder, file_name='Value_Function.npy')

        # Finalization
        self.saved_at = os.path.abspath(folder).replace('\\', '/')
        print(f'Agent saved to: {folder}')


    @classmethod
    def load(cls,
             folder: str
             ) -> 'PBVI_Agent':
        '''
        Function to load a PBVI agent from a given folder it has been saved to.
        It will load the environment the agent has been trained on along with it.

        If it is a subclass of the PBVI_Agent, an instance of that specific subclass will be returned.

        Parameters
        ----------
        folder : str
            The agent folder.

        Returns
        -------
        instance : PBVI_Agent
            The loaded instance of the PBVI Agent.
        '''
        # Load arguments
        arguments = None
        with open(folder + '/METADATA.json', 'r') as json_file:
            arguments = json.load(json_file)

        # Load environment
        environment = Environment.load(arguments['environment_saved_at'])

        # Load specific class
        if arguments['class'] != 'PBVI_Agent':
            from olfactory_navigation import agents
            cls = {name:obj for name, obj in inspect.getmembers(agents)}[arguments['class']]

        # Build instance
        instance = cls(
            environment = environment,
            thresholds = arguments['thresholds'],
            space_aware = arguments['space_aware'],
            spacial_subdivisions = np.array(arguments['spacial_subdivisions']),
            actions = {a_label: a_vector for a_label, a_vector in zip(arguments['action_labels'], arguments['action_set'])},
            name = arguments['name'],
            seed = arguments['seed']
        )

        # Load and set the value function on the instance
        instance.value_function = ValueFunction.load(
            file=folder + '/Value_Function.npy',
            model=instance.model
        )
        instance.trained_at = arguments['trained_at']
        instance.saved_at = folder

        return instance


    def train(self,
              expansions: int,
              full_backup: bool = True,
              update_passes: int = 1,
              max_belief_growth: int = 10,
              initial_belief: BeliefSet | Belief = None,
              initial_value_function: ValueFunction = None,
              prune_level: int = 1,
              prune_interval: int = 10,
              limit_value_function_size: int = -1,
              gamma: float = 0.99,
              eps: float = 1e-6,
              convergence_stop: bool = False,
              use_gpu: bool = False,
              history_tracking_level: int = 1,
              overwrite_training: bool = False,
              print_progress: bool = True,
              print_stats: bool = True,
              **expand_arguments
              ) -> TrainingHistory:
        '''
        Main loop of the Point-Based Value Iteration algorithm.
        It consists in 2 steps, Backup and Expand.
        1. Expand: Expands the belief set base with a expansion strategy given by the parameter expand_function
        2. Backup: Updates the alpha vectors based on the current belief set

        Parameters
        ----------
        expansions : int
            How many times the algorithm has to expand the belief set. (the size will be doubled every time, eg: for 5, the belief set will be of size 32)
        full_backup : bool, default = True
            Whether to force the backup function has to be run on the full set beliefs uncovered since the beginning or only on the new points.
        update_passes : int, default = 1
            How many times the backup function has to be run every time the belief set is expanded.
        max_belief_growth : int, default = 10
            How many beliefs can be added at every expansion step to the belief set.
        initial_belief : BeliefSet or Belief, optional
            An initial list of beliefs to start with.
        initial_value_function : ValueFunction, optional
            An initial value function to start the solving process with.
        prune_level : int, default = 1
            Parameter to prune the value function further before the expand function.
        prune_interval : int, default = 10
            How often to prune the value function. It is counted in number of backup iterations.
        limit_value_function_size : int, default = -1
            When the value function size crosses this threshold, a random selection of 'max_belief_growth' alpha vectors will be removed from the value function
            If set to -1, the value function can grow without bounds.
        use_gpu : bool, default = False
            Whether to use the GPU with cupy array to accelerate solving.
        gamma : float, default = 0.99
            The discount factor to value immediate rewards more than long term rewards.
            The learning rate is 1/gamma.
        eps : float, default = 1e-6
            The smallest allowed changed for the value function.
            Bellow the amound of change, the value function is considered converged and the value iteration process will end early.
            convergence_stop : bool, default = False
        convergence_stop : bool, default = False
            Whether to compute to compute the change in the value function and stop early if this change is smaller than eps.
        history_tracking_level : int, default = 1
            How thorough the tracking of the solving process should be. (0: Nothing; 1: Times and sizes of belief sets and value function; 2: The actual value functions and beliefs sets)
        overwrite_training : bool, default = False
            Whether to force the overwriting of the training if a value function already exists for this agent.
        print_progress : bool, default = True
            Whether or not to print out the progress of the value iteration process.
        print_stats : bool, default = True
            Whether or not to print out statistics at the end of the training run.
        expand_arguments : kwargs
            An arbitrary amount of parameters that will be passed on to the expand function.

        Returns
        -------
        solver_history : SolverHistory
            The history of the solving process with some plotting options.
        '''
        raise NotImplementedError('The train function is not implemented, make a PBVI agent subclass to implement the method')


    def modify_environment(self,
                           new_environment: Environment
                           ) -> 'Agent':
        '''
        Function to modify the environment of the agent.
        If the agent is already trained, the trained element should also be adapted to fit this new environment.

        Parameters
        ----------
        new_environment : Environment
            A modified environment.

        Returns
        -------
        modified_agent : PBVI_Agent
            A new pbvi agent with a modified environment
        '''
        # TODO: Fix this to account for other init parameters
        # GPU support
        if self.is_on_gpu:
            return self.on_cpu.modify_environment(new_environment=new_environment)

        # Creating a new agent instance
        modified_agent = self.__class__(environment = new_environment,
                                        thresholds = self.thresholds,
                                        name = self.name)

        # Modifying the value function
        if self.value_function is not None:
            reshaped_vf_array = np.array([cv2.resize(av, np.array(modified_agent.model.state_grid.shape)[::-1]).ravel()
                                          for av in self.value_function.alpha_vector_array.reshape(len(self.value_function), *self.model.state_grid.shape)])
            modified_vf = ValueFunction(modified_agent.model, alpha_vectors=reshaped_vf_array, action_list=self.value_function.actions)
            modified_agent.value_function = modified_vf

        return modified_agent


    def initialize_state(self,
                         n: int = 1,
                         belief: BeliefSet = None
                         ) -> None:
        '''
        To use an agent within a simulation, the agent's state needs to be initialized.
        The initialization consists of setting the agent's initial belief.
        Multiple agents can be used at once for simulations, for this reason, the belief parameter is a BeliefSet by default.

        Parameters
        ----------
        n : int, default = 1
            How many agents are to be used during the simulation.
        belief : BeliefSet, optional
            An optional set of beliefs to initialize the simulations with.
        '''
        assert self.value_function is not None, "Agent was not trained, run the training function first..."

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

        Returns
        -------
        movement_vector : np.ndarray
            A single or a list of actions chosen by the agent(s) based on their belief.
        '''
        assert self.belief is not None, "Agent was not initialized yet, run the initialize_state function first"

        # Evaluated value function
        _, action = self.value_function.evaluate_at(self.belief)

        # Recording the action played
        self.action_played = action

        # Converting action indexes to movement vectors
        movemement_vector = self.action_set[action,:]

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
                                                     use_reachability = True,
                                                     return_provenance = True)

        # Check for failed updates
        update_successful = xp.isin(xp.arange(len(self.belief)), provenance[:,0])
        self.succeeded_update = update_successful

        return update_successful


    def kill(self,
             simulations_to_kill: np.ndarray
             ) -> None:
        '''
        Function to kill any simulations that have not reached the source but can't continue further.

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


    def generate_beliefs_from_trajectory(self,
                                         history: SimulationHistory,
                                         trajectory_i: int = 0,
                                         initial_belief: Belief = None
                                         ) -> BeliefSet:
        '''
        Function to generate a sequence of belief points from the trajectory from SimulationHistory instance.

        Parameters
        ----------
        history : SimulationHistory
            The simulation history from which the agent's trajectory is extracted.
        trajectory_i : int, default = 0
            The id of the trajectory from which to build the belief sequence.
        initial_belief : Belief, optional
            The initial belief point from which to start the sequence.

        Returns
        -------
        belief_sequence : BeliefSet
            The sequence of beliefs the agent going through in the the trajectory of the simulation.
        '''
        # If the initial belief is not provided, generate one
        if initial_belief is None:
            initial_belief = Belief(self.model)

        # Retrieve the trjactory's simulation dataframe
        df = history.simulation_dfs[trajectory_i]

        # Set the belief that will be iterate on
        belief = initial_belief

        # Belief sequence to be returned at the end
        belief_sequence = [initial_belief]

        for row_id, row in enumerate(df.iterrows()):
            row = row[1]

            # Skip initial position
            if row_id == 0:
                continue

            # Check the ID of the action
            a = np.argwhere(np.all((self.action_set == [row['dy'],row['dx']]), axis=1))[0,0]

            # Retrieve observations
            o = [row['o']]
            if self.space_aware:
                o += [row['y'],row['x']]

            # Discretize observations
            discrete_o = self.discretize_observations(observation=np.array([o]), action=np.array([a]), source_reached=np.array([False]))[0]

            try:
                # Update belief
                belief = belief.update(a=a, o=discrete_o, use_reachability=self.use_reachability)
                belief_sequence.append(belief)
            except:
                print(f'[Warning] Update of belief failed at step {row_id}...')

        return BeliefSet(self.model, belief_sequence)