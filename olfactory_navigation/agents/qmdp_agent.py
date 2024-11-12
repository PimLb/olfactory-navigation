from datetime import datetime
from olfactory_navigation.agents.pbvi_agent import PBVI_Agent, TrainingHistory
from olfactory_navigation.agents.model_based_util.value_function import ValueFunction
from olfactory_navigation.agents.model_based_util import vi_solver


class QMDP_Agent(PBVI_Agent):
    '''
    An agent that relies on Model-Based Reinforcement Learning. It is a simplified version of the PBVI_Agent.
    It runs the a Value Iteration solver, assuming full observability. The value function that comes out from this is therefore used to make choices.

    As stated, during simulations, the agent will choose actions based on an argmax of what action has the highest matrix product of the value function with the belief vector.


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
    def train(self,
              expansions: int,
              initial_value_function: ValueFunction | None = None,
              gamma: float = 0.99,
              eps: float = 1e-6,
              use_gpu: bool = False,
              history_tracking_level: int = 1,
              overwrite_training: bool = False,
              print_progress: bool = True,
              print_stats: bool = True
              ) -> TrainingHistory:
        '''
        Simplified version of the training. It consists in running the Value Iteration process.

        Parameters
        ----------
        expansions : int
            How many iterations to run the Value Iteration process for.
        initial_value_function : ValueFunction, optional
            An initial value function to start the solving process with.
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
        overwrite_training : bool, default=False
            Whether to force the overwriting of the training if a value function already exists for this agent.
        print_progress : bool, default=True
            Whether or not to print out the progress of the value iteration process.
        print_stats : bool, default=True
            Whether or not to print out statistics at the end of the training run.

        Returns
        -------
        solver_history : SolverHistory
            The history of the solving process with some plotting options.
        '''
        # Handeling the case where the agent is already trained
        if (self.value_function is not None):
            if overwrite_training:
                self.trained_at = None
                self.name = '-'.join(self.name.split('-')[:-1])
                self.value_function = None
            else:
                initial_value_function = self.value_function

        model = self.model if not use_gpu else self.model.gpu_model

        # Value Iteration solving
        value_function, hist = vi_solver.solve(model = model,
                                               horizon = expansions,
                                               initial_value_function = initial_value_function,
                                               gamma = gamma,
                                               eps = eps,
                                               use_gpu = use_gpu,
                                               history_tracking_level = history_tracking_level,
                                               print_progress = print_progress)

        # Record when it was trained
        self.trained_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.name += f'-trained_{self.trained_at}'

        self.value_function = value_function.to_cpu() if not self.on_gpu else value_function.to_gpu()

        # Print stats if requested
        if print_stats:
            print(hist.summary)

        return hist