import cv2
import inspect
import json
import os
import shutil

from datetime import datetime
from tqdm.auto import trange
from typing import Callable

from olfactory_navigation.environment import Environment
from olfactory_navigation.agent import Agent
from olfactory_navigation.agents.model_based_util.pomdp import Model
from olfactory_navigation.agents.model_based_util.value_function import ValueFunction
from olfactory_navigation.agents.model_based_util.belief import Belief, BeliefSet
from olfactory_navigation.agents.model_based_util.environment_converter import exact_converter

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
                 model: Model,
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
            self.belief_sets.append(belief_set if not belief_set.is_on_gpu else belief_set.to_cpu())


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
            self.value_functions.append(value_function if not value_function.is_on_gpu else value_function.to_cpu())


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
    threshold : float, optional, default=3e-6
        The olfactory sensitivity of the agent. Odor cues under this threshold will not be detected by the agent.
    actions : dict or np.ndarray, optional
        The set of action available to the agent. It should match the type of environment (ie: if the environment has layers, it should contain a layer component to the action vector, and similarly for a third dimension).
        Else, a dict of strings and action vectors where the strings represent the action labels.
        If none is provided, by default, all unit movement vectors are included and shuch for all layers (if the environment has layers.)
    name : str, optional
        A custom name to give the agent. If not provided is will be a combination of the class-name and the threshold.
    environment_converter : Callable, default=exact_converter
        A function to convert the olfactory environment instance to a POMDP Model instance.
        By default, we use an exact convertion that keeps the shape of the environment to make the amount of states of the POMDP Model.
    converter_parameters : dict, optional
        A set of additional parameters to be passed down to the environment converter.

    Attributes
    ---------
    environment : Environment
    threshold : float
    name : name
    action_set : np.ndarray
        The actions allowed of the agent. Formulated as movement vectors as [(layer,) (dz,) dy, dx].
    action_labels : list[str]
        The labels associated to the action vectors present in the action set.
    model : Model
        The environment converted to a POMDP model using the "from_environment" constructor of the Model class.
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
    def __init__(self,
                 environment: Environment,
                 threshold: float | None = 3e-6,
                 actions: dict[str, np.ndarray] | np.ndarray | None = None,
                 name: str | None = None,
                 environment_converter: Callable | None = None,
                 **converter_parameters
                 ) -> None:
        super().__init__(
            environment = environment,
            threshold = threshold,
            actions = actions,
            name = name
        )

        # Converting the olfactory environment to a POMDP Model
        if callable(environment_converter):
            loaded_model = environment_converter(agent=self, **converter_parameters)
        else:
            # Using the exact converter
            loaded_model = exact_converter(agent=self)
        self.model:Model = loaded_model

        # Trainable variables
        self.trained_at = None
        self.value_function = None

        # Status variables
        self.belief = None
        self.action_played = None


    def to_gpu(self) -> Agent:
        '''
        Function to send the numpy arrays of the agent to the gpu.
        It returns a new instance of the Agent class with the arrays on the gpu

        Returns
        -------
        gpu_agent : Agent
            A copy of the agent with the arrays on the GPU.
        '''
        assert gpu_support, "GPU support is not enabled, Cupy might need to be installed..."

        # Generating a new instance
        cls = self.__class__
        gpu_agent = cls.__new__(cls)

        # Copying arguments to gpu
        for arg, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                setattr(gpu_agent, arg, cp.array(val))
            elif isinstance(val, Model):
                gpu_agent.model = self.model.gpu_model
            elif isinstance(val, ValueFunction):
                gpu_agent.value_function =self.value_function.to_gpu()
            elif isinstance(val, BeliefSet):
                gpu_agent.belief = self.belief.to_gpu()
            else:
                setattr(gpu_agent, arg, val)

        # Self reference instances
        self._alternate_version = gpu_agent
        gpu_agent._alternate_version = self

        gpu_agent.on_gpu = True
        return gpu_agent


    def save(self,
             folder: str | None = None,
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
        force : bool, default=False
            Whether to overwrite an already saved agent with the same name at the same path.
        save_environment : bool, default=False
            Whether to save the environment data along with the agent.
        '''
        assert self.trained_at is not None, "The agent is not trained, there is nothing to save."

        # GPU support
        if self.on_gpu:
            self.to_cpu().save(folder=folder, force=force, save_environment=save_environment)
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

        # Generating the metadata arguments dictionary
        arguments = {}
        arguments['name'] = self.name
        arguments['class'] = self.class_name
        arguments['threshold'] = self.threshold
        arguments['environment_name'] = self.environment.name
        arguments['environment_saved_at'] = self.environment.saved_at
        arguments['trained_at'] = self.trained_at

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
            environment=environment,
            threshold=arguments['threshold'],
            name=arguments['name']
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
              initial_belief: BeliefSet | Belief | None = None,
              initial_value_function: ValueFunction | None = None,
              prune_level: int = 1,
              prune_interval: int = 10,
              limit_value_function_size: int = -1,
              gamma: float = 0.99,
              eps: float = 1e-6,
              use_gpu: bool = False,
              history_tracking_level: int = 1,
              force: bool = False,
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
        full_backup : bool, default=True
            Whether to force the backup function has to be run on the full set beliefs uncovered since the beginning or only on the new points.
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
        expand_arguments : kwargs
            An arbitrary amount of parameters that will be passed on to the expand function.

        Returns
        -------
        solver_history : SolverHistory
            The history of the solving process with some plotting options.
        '''
        # GPU support
        if use_gpu and not self.on_gpu:
            gpu_agent = self.to_gpu()
            solver_history = super(self.__class__, gpu_agent).train(
                expansions=expansions,
                full_backup=full_backup,
                update_passes=update_passes,
                max_belief_growth=max_belief_growth,
                initial_belief=initial_belief,
                initial_value_function=initial_value_function,
                prune_level=prune_level,
                prune_interval=prune_interval,
                limit_value_function_size=limit_value_function_size,
                gamma=gamma,
                eps=eps,
                use_gpu=use_gpu,
                history_tracking_level=history_tracking_level,
                force=force,
                print_progress=print_progress,
                print_stats=print_stats,
                **expand_arguments
            )
            self.value_function = gpu_agent.value_function.to_cpu()
            return solver_history

        xp = np if not self.on_gpu else cp

        # Getting model
        model = self.model

        # Initial belief
        if initial_belief is None:
            belief_set = BeliefSet(model, [Belief(model)])
        elif isinstance(initial_belief, BeliefSet):
            belief_set = initial_belief.to_gpu() if self.on_gpu else initial_belief 
        else:
            initial_belief = Belief(model, xp.array(initial_belief.values))
            belief_set = BeliefSet(model, [initial_belief])
        
        # Handeling the case where the agent is already trained
        if (self.value_function is not None):
            if not force:
                raise Exception('Agent has already been trained. The force parameter needs to be set to "True" if training should still happen')
            else:
                self.trained_at = None
                self.name = '-'.join(self.name.split('-')[:-1])
                self.value_function = None

        # Initial value function
        if initial_value_function is None:
            value_function = ValueFunction(model, model.expected_rewards_table.T, model.actions)
        else:
            value_function = initial_value_function.to_gpu() if self.on_gpu else initial_value_function

        # Convergence check boundary
        max_allowed_change = eps * (gamma / (1-gamma))

        # History tracking
        training_history = TrainingHistory(tracking_level=history_tracking_level,
                                           model=model,
                                           gamma=gamma,
                                           eps=eps,
                                           expand_append=full_backup,
                                           initial_value_function=value_function,
                                           initial_belief_set=belief_set)
        
        # Loop
        iteration = 0
        expand_value_function = value_function
        old_value_function = value_function

        try:
            iterator = trange(expansions, desc='Expansions') if print_progress else range(expansions)
            iterator_postfix = {}
            for expansion_i in iterator:

                # 1: Expand belief set
                start_ts = datetime.now()

                new_belief_set = self.expand(belief_set=belief_set,
                                             value_function=value_function,
                                             max_generation=max_belief_growth,
                                             **expand_arguments)

                # Add new beliefs points to the total belief_set
                belief_set = belief_set.union(new_belief_set)

                expand_time = (datetime.now() - start_ts).total_seconds()
                training_history.add_expand_step(expansion_time=expand_time, belief_set=belief_set)

                # 2: Backup, update value function (alpha vector set)
                for _ in range(update_passes) if (not print_progress or update_passes <= 1) else trange(update_passes, desc=f'Backups {expansion_i}'):
                    start_ts = datetime.now()

                    # Backup step
                    value_function = self.backup(belief_set if full_backup else new_belief_set,
                                                 value_function,
                                                 gamma=gamma,
                                                 append=(not full_backup),
                                                 belief_dominance_prune=False)
                    backup_time = (datetime.now() - start_ts).total_seconds()

                    # Additional pruning
                    if (iteration % prune_interval) == 0 and iteration > 0:
                        start_ts = datetime.now()
                        vf_len = len(value_function)

                        value_function.prune(prune_level)

                        prune_time = (datetime.now() - start_ts).total_seconds()
                        alpha_vectors_pruned = len(value_function) - vf_len
                        training_history.add_prune_step(prune_time, alpha_vectors_pruned)

                    # Check if value function size is above threshold
                    if limit_value_function_size >= 0 and len(value_function) > limit_value_function_size:
                        # Compute matrix multiplications between avs and beliefs
                        alpha_value_per_belief = xp.matmul(value_function.alpha_vector_array, belief_set.belief_array.T)

                        # Select the useful alpha vectors
                        best_alpha_vector_per_belief = xp.argmax(alpha_value_per_belief, axis=0)
                        useful_alpha_vectors = xp.unique(best_alpha_vector_per_belief)

                        # Select a random selection of vectors to delete
                        unuseful_alpha_vectors = xp.delete(xp.arange(len(value_function)), useful_alpha_vectors)
                        random_vectors_to_delete = xp.random.choice(unuseful_alpha_vectors,
                                                                    size=max_belief_growth,
                                                                    p=(xp.arange(len(unuseful_alpha_vectors))[::-1] / xp.sum(xp.arange(len(unuseful_alpha_vectors)))))
                                                                    # replace=False,
                                                                    # p=1/len(unuseful_alpha_vectors))

                        value_function = ValueFunction(model=model,
                                                       alpha_vectors=xp.delete(value_function.alpha_vector_array, random_vectors_to_delete, axis=0),
                                                       action_list=xp.delete(value_function.actions, random_vectors_to_delete))
                        
                        iterator_postfix['|useful|'] = useful_alpha_vectors.shape[0]
                    
                    # Compute the change between value functions
                    max_change = self.compute_change(value_function, old_value_function, belief_set)

                    # History tracking
                    training_history.add_backup_step(backup_time, max_change, value_function)

                    # Convergence check
                    if max_change < max_allowed_change:
                        break

                    old_value_function = value_function

                    # Update iteration counter
                    iteration += 1

                # Compute change with old expansion value function
                expand_max_change = self.compute_change(expand_value_function, value_function, belief_set)

                if expand_max_change < max_allowed_change:
                    print('Converged!')
                    break

                expand_value_function = value_function

                iterator_postfix['|V|'] = len(value_function)
                iterator_postfix['|B|'] = len(belief_set)

                if print_progress:
                    iterator.set_postfix(iterator_postfix)

        except MemoryError as e:
            print(f'Memory full: {e}')
            print('Returning value function and history as is...\n')

        # Final pruning
        start_ts = datetime.now()
        vf_len = len(value_function)

        value_function.prune(prune_level)

        # History tracking
        prune_time = (datetime.now() - start_ts).total_seconds()
        alpha_vectors_pruned = len(value_function) - vf_len
        training_history.add_prune_step(prune_time, alpha_vectors_pruned)

        # Record when it was trained
        self.trained_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.name += f'-trained_{self.trained_at}'

        # Saving value function
        self.value_function = value_function

        # Print stats if requested
        if print_stats:
            print(training_history.summary)

        return training_history


    def compute_change(self,
                       value_function: ValueFunction,
                       new_value_function: ValueFunction,
                       belief_set: BeliefSet
                       ) -> float:
        '''
        Function to compute whether the change between two value functions can be considered as having converged based on the eps parameter of the Solver.
        It check for each belief, the maximum value and take the max change between believe's value functions.
        If this max change is lower than eps * (gamma / (1 - gamma)).

        Parameters
        ----------
        value_function : ValueFunction
            The first value function to compare.
        new_value_function : ValueFunction
            The second value function to compare.
        belief_set : BeliefSet
            The set of believes to check the values on to compute the max change on.

        Returns
        -------
        max_change : float
            The maximum change between value functions at belief points.
        '''
        # Get numpy corresponding to the arrays
        xp = np if not gpu_support else cp.get_array_module(value_function.alpha_vector_array)

        # Computing Delta for each beliefs
        max_val_per_belief = xp.max(xp.matmul(belief_set.belief_array, value_function.alpha_vector_array.T), axis=1)
        new_max_val_per_belief = xp.max(xp.matmul(belief_set.belief_array, new_value_function.alpha_vector_array.T), axis=1)
        max_change = xp.max(xp.abs(new_max_val_per_belief - max_val_per_belief))

        return max_change


    def expand(self,
               belief_set: BeliefSet,
               value_function: ValueFunction,
               max_generation: int,
               **kwargs
               ) -> BeliefSet:
        '''
        Abstract function!
        This function should be implemented in subclasses.
        The expand function consists in the exploration of the belief set.
        It takes as input a belief set and generates at most 'max_generation' beliefs from it.

        The current value function is also passed as an argument as it is used in some PBVI techniques to guide the belief exploration.

        Parameters
        ----------
        belief_set : BeliefSet
            The belief or set of beliefs to be used as a starting point for the exploration.
        value_function : ValueFunction
            The current value function. To be used to guide the exploration process.
        max_generation : int
            How many beliefs to be generated at most.
        kwargs
            Special parameters for the particular flavors of the PBVI Agent.

        Returns
        -------
        new_belief_set : BeliefSet
            A new (or expanded) set of beliefs.
        '''
        raise NotImplementedError('PBVI class is abstract so expand function is not implemented, make an PBVI_agent subclass to implement the method')


    def backup(self,
               belief_set: BeliefSet,
               value_function: ValueFunction,
               gamma: float = 0.99,
               append: bool = False,
               belief_dominance_prune: bool = True
               ) -> ValueFunction:
        '''
        This function has purpose to update the set of alpha vectors. It does so in 3 steps:
        1. It creates projections from each alpha vector for each possible action and each possible observation
        2. It collapses this set of generated alpha vectors by taking the weighted sum of the alpha vectors weighted by the observation probability and this for each action and for each belief.
        3. Then it further collapses the set to take the best alpha vector and action per belief
        In the end we have a set of alpha vectors as large as the amount of beliefs.

        The alpha vectors are also pruned to avoid duplicates and remove dominated ones.

        Parameters
        ----------
        belief_set : BeliefSet
            The belief set to use to generate the new alpha vectors with.
        value_function : ValueFunction
            The alpha vectors to generate the new set from.
        gamma : float, default=0.99
            The discount factor to value immediate rewards more than long term rewards.
            The learning rate is 1/gamma.
        append : bool, default=False
            Whether to append the new alpha vectors generated to the old alpha vectors before pruning.
        belief_dominance_prune : bool, default=True
            Whether, before returning the new value function, checks what alpha vectors have a supperior value, if so it adds it.
            
        Returns
        -------
        new_alpha_set : ValueFunction
            A list of updated alpha vectors.
        '''
        xp = np if not self.on_gpu else cp
        model = self.model

        # Step 1
        vector_array = value_function.alpha_vector_array
        vectors_array_reachable_states = vector_array[xp.arange(vector_array.shape[0])[:,None,None,None], model.reachable_states[None,:,:,:]]
        
        gamma_a_o_t = gamma * xp.einsum('saor,vsar->aovs', model.reachable_transitional_observation_table, vectors_array_reachable_states)

        # Step 2
        belief_array = belief_set.belief_array # bs
        best_alpha_ind = xp.argmax(xp.tensordot(belief_array, gamma_a_o_t, (1,3)), axis=3) # argmax(bs,aovs->baov) -> bao

        best_alphas_per_o = gamma_a_o_t[model.actions[None,:,None,None], model.observations[None,None,:,None], best_alpha_ind[:,:,:,None], model.states[None,None,None,:]] # baos

        alpha_a = model.expected_rewards_table.T + xp.sum(best_alphas_per_o, axis=2) # as + bas

        # Step 3
        best_actions = xp.argmax(xp.einsum('bas,bs->ba', alpha_a, belief_array), axis=1)
        alpha_vectors = xp.take_along_axis(alpha_a, best_actions[:,None,None],axis=1)[:,0,:]

        # Belief domination
        if belief_dominance_prune:
            best_value_per_belief = xp.sum((belief_array * alpha_vectors), axis=1)
            old_best_value_per_belief = xp.max(xp.matmul(belief_array, vector_array.T), axis=1)
            dominating_vectors = best_value_per_belief > old_best_value_per_belief

            best_actions = best_actions[dominating_vectors]
            alpha_vectors = alpha_vectors[dominating_vectors]

        # Creation of value function
        new_value_function = ValueFunction(model, alpha_vectors, best_actions)

        # Union with previous value function
        if append:
            new_value_function.extend(value_function)
        
        return new_value_function


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
        # GPU support
        if self.on_gpu:
            return self.to_cpu().modify_environment(new_environment=new_environment)

        # Creating a new agent instance
        modified_agent = self.__class__(environment=new_environment,
                                        threshold=self.threshold,
                                        name=self.name)
        
        # Modifying the value function
        if self.value_function is not None:
            reshaped_vf_array = np.array([cv2.resize(av, np.array(modified_agent.model.state_grid.shape)[::-1]).ravel()
                                          for av in self.value_function.alpha_vector_array.reshape(len(self.value_function), *self.model.state_grid.shape)])
            modified_vf = ValueFunction(modified_agent.model, alpha_vectors=reshaped_vf_array, action_list=self.value_function.actions)
            modified_agent.value_function = modified_vf

        return modified_agent


    def initialize_state(self,
                         n: int = 1
                         ) -> None:
        '''
        To use an agent within a simulation, the agent's state needs to be initialized.
        The initialization consists of setting the agent's initial belief.
        Multiple agents can be used at once for simulations, for this reason, the belief parameter is a BeliefSet by default.
        
        Parameters
        ----------
        n : int, default=1
            How many agents are to be used during the simulation.
        '''
        assert self.value_function is not None, "Agent was not trained, run the training function first..."

        self.belief = BeliefSet(self.model, [Belief(self.model) for _ in range(n)])


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
                     observation: np.ndarray,
                     source_reached: np.ndarray
                     ) -> None:
        '''
        Function to update the internal state(s) of the agent(s) based on the previous action(s) taken and the observation(s) received.

        Parameters
        ----------
        observation : np.ndarray
            The observation(s) the agent(s) made.
        source_reached : np.ndarray
            A boolean array of whether the agent(s) have reached the source or not.
        '''
        assert self.belief is not None, "Agent was not initialized yet, run the initialize_state function first"

        # Binarize observations
        observation_ids = np.where(observation > self.threshold, 1, 0).astype(int)
        observation_ids[source_reached] = 2 # Observe source

        # Update the set of beliefs
        self.belief = self.belief.update(actions=self.action_played, observations=observation_ids, throw_error=False)

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