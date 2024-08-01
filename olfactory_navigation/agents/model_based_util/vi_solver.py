from datetime import datetime
from matplotlib import pyplot as plt
from tqdm.auto import trange

from olfactory_navigation.agents.model_based_util.mdp import Model
from olfactory_navigation.agents.model_based_util.value_function import ValueFunction

import numpy as np
gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')


class SolverHistory:
    '''
    Class to represent the solving history of a solver.
    The purpose of this class is to allow plotting of the solution and plotting the evolution of the value function over the training process.
    This class is not meant to be instanciated manually, it meant to be used when returned by the solve() method of a Solver object.


    Parameters
    ----------
    tracking_level : int
        The tracking level of the solver.
    model : mdp.Model
        The model that has been solved by the Solver.
    gamma : float
        The gamma parameter used by the solver (learning rate).
    eps : float
        The epsilon parameter used by the solver (covergence bound).
    initial_value_function : ValueFunction, optional
        The initial value function the solver will use to start the solving process.
    
    Attributes
    ----------
    tracking_level : int
    model : mdp.Model
    gamma : float
    eps : float
    run_ts : datetime
        The time at which the SolverHistory object was instantiated which is assumed to be the start of the solving run.
    iteration_times : list[float]
        A list of recorded iteration times.
    value_function_changes : list[float]
        A list of recorded value function changes (the maximum changed value between 2 value functions).
    value_functions : list[ValueFunction]
        A list of recorded value functions.
    solution : ValueFunction
    summary : str
    '''
    def __init__(self,
                 tracking_level: int,
                 model: Model,
                 gamma: float,
                 eps: float,
                 initial_value_function: ValueFunction | None = None
                 ) -> None:
        self.tracking_level = tracking_level
        self.model = model
        self.gamma = gamma
        self.eps = eps
        self.run_ts = datetime.now()

        # Tracking metrics
        self.iteration_times = []
        self.value_function_changes = []

        self.value_functions = []
        if self.tracking_level >= 2:
            self.value_functions.append(initial_value_function)


    @property
    def solution(self) -> ValueFunction:
        '''
        The last value function of the solving process.
        '''
        assert self.tracking_level >= 2, "Tracking level is set too low, increase it to 2 if you want to have value function tracking as well."
        return self.value_functions[-1]
    

    def add(self,
            iteration_time: float,
            value_function_change: float,
            value_function: ValueFunction
            ) -> None:
        '''
        Function to add a step in the simulation history.

        Parameters
        ----------
        iteration_time : float
            The time it took to run the iteration.
        value_function_change : float
            The change between the value function of this iteration and of the previous iteration.
        value_function : ValueFunction
            The value function resulting after a step of the solving process.
        '''
        if self.tracking_level >= 1:
            self.iteration_times.append(float(iteration_time))
            self.value_function_changes.append(float(value_function_change))

        if self.tracking_level >= 2:
            self.value_functions.append(value_function if not value_function.is_on_gpu else value_function.to_cpu())
    

    @property
    def summary(self) -> str:
        '''
        A summary as a string of the information recorded.
        '''
        summary_str =  f'Summary of Value Iteration run'
        summary_str += f'\n  - Model: {self.model.state_count}-state, {self.model.action_count}-action'
        summary_str += f'\n  - Converged in {len(self.iteration_times)} iterations and {sum(self.iteration_times):.4f} seconds'
        
        if self.tracking_level >= 1:
            summary_str += f'\n  - Took on average {sum(self.iteration_times) / len(self.iteration_times):.4f}s per iteration'
        
        return summary_str
    

    def plot_changes(self) -> None:
        '''
        Function to plot the value function changes over the solving process.
        '''
        assert self.tracking_level >= 1, "To plot the change of the value function over time, use tracking level 1 or higher."

        plt.title('Value function change over time')
        plt.plot(np.arange(len(self.value_function_changes)), self.value_function_changes)
        plt.xlabel('Iteration')
        plt.ylabel('Value function change')
        plt.show()


def solve(model: Model,
          horizon: int = 100,
          initial_value_function: ValueFunction | None = None,
          gamma: float = 0.99,
          eps: float = 1e-6,
          use_gpu: bool = False,
          history_tracking_level: int = 1,
          print_progress: bool = True
          ) -> tuple[ValueFunction, SolverHistory]:
    '''
    Function to solve an MDP model using Value Iteration.
    If an initial value function is not provided, the value function will be initiated with the expected rewards.

    Parameters
    ----------
    model : mdp.Model
        The model on which to run value iteration.
    horizon : int, default=100
        How many iterations to run the value iteration solver for.
    initial_value_function : ValueFunction, optional
        An optional initial value function to kick-start the value iteration process.
    gamma : float, default=0.99
        The discount factor to value immediate rewards more than long term rewards.
        The learning rate is 1/gamma.
    eps : float, default=1e-6
        The smallest allowed changed for the value function.
        Bellow the amound of change, the value function is considered converged and the value iteration process will end early.
    use_gpu : bool, default=False
        Whether to use the GPU with cupy array to accelerate solving.
    history_tracking_level : int, default=1
        How thorough the tracking of the solving process should be. (0: Nothing; 1: Times and sizes of belief sets and value function; 2: The actual value functions and beliefs sets)
    print_progress : bool, default=True
        Whether or not to print out the progress of the value iteration process.

    Returns
    -------
    value_function: ValueFunction
        The resulting value function solution to the model.
    history : SolverHistory
        The tracking of the solution over time.
    '''
    # GPU support
    if use_gpu:
        assert gpu_support, "GPU support is not enabled, Cupy might need to be installed..."

    xp = np if not use_gpu else cp
    model = model if not use_gpu else model.gpu_model

    # Value function initialization
    if initial_value_function is None:
        V = ValueFunction(model, model.expected_rewards_table.T, model.actions)
    else:
        V = initial_value_function.to_gpu() if use_gpu else initial_value_function
    V_opt = xp.max(V.alpha_vector_array, axis=0)

    # History tracking setup
    solve_history = SolverHistory(tracking_level=history_tracking_level,
                                  model=model,
                                  gamma=gamma,
                                  eps=eps,
                                  initial_value_function=V)

    # Computing max allowed change from epsilon and gamma parameters
    max_allowed_change = eps * (gamma / (1-gamma))

    iterator = trange(horizon) if print_progress else range(horizon)
    for _ in iterator:
        old_V_opt = V_opt
        
        start = datetime.now()

        # Computing the new alpha vectors
        alpha_vectors = model.expected_rewards_table.T + (gamma * xp.einsum('sar,sar->as', model.reachable_probabilities, V_opt[model.reachable_states]))
        V = ValueFunction(model, alpha_vectors, model.actions)

        V_opt = xp.max(V.alpha_vector_array, axis=0)
        
        # Change computation
        max_change = xp.max(xp.abs(V_opt - old_V_opt))

        # Tracking the history
        iteration_time = (datetime.now() - start).total_seconds()
        solve_history.add(iteration_time=iteration_time,
                            value_function_change=max_change,
                            value_function=V)

        # Convergence check
        if max_change < max_allowed_change:
            if print_progress:
                iterator.close()
            break

    return V, solve_history