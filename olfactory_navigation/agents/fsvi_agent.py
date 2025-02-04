from olfactory_navigation.agents.pbvi_agent import PBVI_Agent, TrainingHistory
from olfactory_navigation.agents.model_based_util.mdp import log
from olfactory_navigation.agents.model_based_util.value_function import ValueFunction
from olfactory_navigation.agents.model_based_util.belief import Belief, BeliefSet
from olfactory_navigation.agents.model_based_util import vi_solver

import numpy as np
gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')


class FSVI_Agent(PBVI_Agent):
    '''
    A particular flavor of the Point-Based Value Iteration based agent.
    The general concept relies on Model-Based reinforcement learning as described in: Pineau, J., Gordon, G., & Thrun, S. (2003, August). Point-based value iteration: An anytime algorithm for POMDPs
    The Forward Search Value Iteration algorithm is described in: Shani, G., Brafman, R. I., & Shimony, S. E. (2007, January). Forward Search Value Iteration for POMDPs

    The training consist in two steps:

    - Expand: Where belief points are explored based on the some strategy (to be defined by subclasses).

    - Backup: Using the generated belief points, the value function is updated.

    The belief points are probability distributions over the state space and are therefore vectors of |S| elements.

    Actions are chosen based on a value function. A value function is a set of alpha vectors of dimentionality |S|.
    Each alpha vector is associated to a single action but multiple alpha vectors can be associated to the same action.
    To choose an action at a given belief point, a dot product is taken between each alpha vector and the belief point and the action associated with the highest result is chosen.

    Forward Search exploration concept:
    It relies of the solution of the Fully-Observable (MDP) problem to guide the exploration of belief points.
    It makes an agent start randomly in the environment and makes him take steps following the MDP solution while generating belief points along the way.
    Each time the expand function is called it starts generated a new set of belief points and the update function uses only the latest generated belief points to make update the value function.

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
    mdp_policy : ValueFunction
        The solution to the fully version of the problem.
    '''
    # FSVI special attribute
    mdp_policy = None

    def expand(self,
               belief_set: BeliefSet,
               value_function: ValueFunction,
               max_generation: int,
               mdp_policy: ValueFunction
               ) -> BeliefSet:
        '''
        Function implementing the exploration process using the MDP policy in order to generate a sequence of Beliefs following the the Forward Search Value Iteration principles.
        It is a loop is started by a initial state 's' and using the MDP policy, chooses the best action to take.
        Following this, a random next state 's_p' is being sampled from the transition probabilities and a random observation 'o' based on the observation probabilities.
        Then the given belief is updated using the chosen action and the observation received and the updated belief is added to the sequence.
        Once the state is a goal state, the loop is done and the belief sequence is returned.

        Parameters
        ----------
        belief_set : BeliefSet
            A belief set containing a single belief to start the sequence with.
            A random state will be chosen based on the probability distribution of the belief.
        value_function : ValueFunction
            The current value function. (NOT USED)
        max_generation : int
            How many beliefs to be generated at most.
        mdp_policy : ValueFunction
            The mdp policy used to choose the action from with the given state 's'.

        Returns
        -------
        belief_set : BeliefSet
            A new sequence of beliefs.
        '''
        # GPU support
        xp = np if not self.on_gpu else cp
        model = self.model

        # Getting initial belief
        b0 = belief_set.belief_list[0]
        belief_list = [b0]

        # Choose a random starting state
        s = b0.random_state()

        # Setting the working belief
        b = b0

        for _ in range(max_generation - 1): #-1 due to a one belief already being present in the set
            # Choose action based on mdp value function
            a_star = xp.argmax(mdp_policy.alpha_vector_array[:,s])

            # Pick a random next state (weighted by transition probabilities)
            s_p = model.transition(s, a_star)

            # Pick a random observation weighted by observation probabilities in state s_p and after having done action a_star
            o = model.observe(s_p, a_star)

            # Generate a new belief based on a_star and o
            b_p = b.update(a_star, o)

            # Record new belief
            belief_list.append(b_p)

            # Updating s and b
            s = s_p
            b = b_p

            # Reset and belief if end state is reached
            if s in model.end_states:
                s = b0.random_state()
                b = b0

        return BeliefSet(model, belief_list)


    def train(self,
              expansions: int,
              update_passes: int = 1,
              max_belief_growth: int = 10,
              initial_belief: BeliefSet | Belief | None = None,
              initial_value_function: ValueFunction | None = None,
              mdp_policy: ValueFunction | None = None,
              prune_level: int = 1,
              prune_interval: int = 10,
              limit_value_function_size: int = -1,
              gamma: float = 0.99,
              eps: float = 1e-6,
              use_gpu: bool = False,
              history_tracking_level: int = 1,
              overwrite_training: bool = False,
              print_progress: bool = True,
              print_stats: bool = True
              ) -> TrainingHistory:
        '''
        Main loop of the Point-Based Value Iteration algorithm.
        It consists in 2 steps, Backup and Expand.
        1. Expand: Expands the belief set base with a expansion strategy given by the parameter expand_function
        2. Backup: Updates the alpha vectors based on the current belief set

        Foward Search Value Iteration:
        - By default it performs the backup only on set of beliefs generated by the expand function. (so it full_backup=False)

        Parameters
        ----------
        expansions : int
            How many times the algorithm has to expand the belief set. (the size will be doubled every time, eg: for 5, the belief set will be of size 32)
        update_passes : int, default=1
            How many times the backup function has to be run every time the belief set is expanded.
        max_belief_growth : int, default=10
            How many beliefs can be added at every expansion step to the belief set.
        initial_belief : BeliefSet or Belief, optional
            An initial list of beliefs to start with.
        initial_value_function : ValueFunction, optional
            An initial value function to start the solving process with.
        mdp_policy : ValueFunction, optional
            The MDP solution to guide the expand process.
            If it is not provided, the Value Iteration for the MDP version of the problem will be run. (using the same gamma and eps as set here; horizon=1000)
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
        if mdp_policy is not None:
            self.mdp_policy = mdp_policy
        elif (self.mdp_policy is None) or overwrite_training:
            log('MDP_policy, not provided. Solving MDP with Value Iteration...')
            self.mdp_policy, hist = vi_solver.solve(model = self.model,
                                                    horizon = 1000,
                                                    initial_value_function = initial_value_function,
                                                    gamma = gamma,
                                                    eps = eps,
                                                    use_gpu = use_gpu,
                                                    history_tracking_level = 1,
                                                    print_progress = print_progress)

            if print_stats:
                print(hist.summary)

        return super().train(expansions = expansions,
                             full_backup = False,
                             update_passes = update_passes,
                             max_belief_growth = max_belief_growth,
                             initial_belief = initial_belief,
                             initial_value_function = initial_value_function,
                             prune_level = prune_level,
                             prune_interval = prune_interval,
                             limit_value_function_size = limit_value_function_size,
                             gamma = gamma,
                             eps = eps,
                             use_gpu = use_gpu,
                             history_tracking_level = history_tracking_level,
                             overwrite_training = overwrite_training,
                             print_progress = print_progress,
                             print_stats = print_stats,
                             mdp_policy = self.mdp_policy)