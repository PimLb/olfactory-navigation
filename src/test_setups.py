import numpy as np

from src.agent import Agent
from src.environment import Environment
from src.simulation import run_test, SimulationHistory


def run_all_starts_test(
             agent:Agent,
             environment:Environment|None=None,
             time_shift:int|np.ndarray=0,
             time_loop:bool=True,
             horizon:int=1000,
             reward_discount:float=0.99,
             print_progress:bool=True,
             print_stats:bool=True,
             use_gpu:bool=False
             ) -> SimulationHistory:
    '''
    Function to run a test with all the available starting positions based on the environment provided (or the environmnent of the agent).

    Parameters
    ----------
    agent : Agent
        The agent to be tested
    environment : Environment, optional
        The environment to run the simulations in.
        By default, the environment linked to the agent will used.
        This parameter is intended if the environment needs to be modified compared to environment the agent was trained on.
    time_shift : int or np.ndarray, default=0
        The time at which to start the olfactory simulation array.
        It can be either a single value, or n values.
    time_loop : bool, default=True
        Whether to loop the time if reaching the end. (starts back at 0)
    horizon : int, default=1000
        The amount of steps to run the simulation for before killing the remaining simulations.
    reward_discount : float, default=0.99
        How much a given reward is discounted based on how long it took to get it.
        It is purely used to compute the Average Discount Reward (ADR) after the simulation.
    print_progress : bool, default=True
        Wheter to show a progress bar of what step the simulations are at.
    print_stats : bool, default=True
        Wheter to print the stats at the end of the run.
    use_gpu : bool, default=False
        Whether to run the simulations on the GPU or not.
    
    Returns
    -------
    hist : SimulationHistory
        A SimulationHistory object that tracked all the positions, actions and observations.
    '''
    # Handle the case an specific environment is given
    environment_provided = environment is not None
    if environment_provided:
        assert environment.shape == agent.environment.shape, "The provided environment's shape doesn't match the environment has been trained on..."
    else:
        environment = agent.environment
    
    # Gathering starting points
    start_points = np.argwhere(environment.start_probabilities > 0)
    n = len(start_points)

    return run_test(
        agent=agent,
        n=n,
        start_points=start_points,
        environment=environment if environment_provided else None,
        time_shift=time_shift,
        time_loop=time_loop,
        horizon=horizon,
        reward_discount=reward_discount,
        print_progress=print_progress,
        print_stats=print_stats,
        use_gpu=use_gpu
    )


def run_n_by_cell_test(
             agent:Agent,
             cell_width:int=10,
             n_by_cell:int=10,
             environment:Environment|None=None,
             time_shift:int|np.ndarray=0,
             time_loop:bool=True,
             horizon:int=1000,
             reward_discount:float=0.99,
             print_progress:bool=True,
             print_stats:bool=True,
             use_gpu:bool=False
             ) -> SimulationHistory:
    '''
    Function to run a test with simulations starting in different cells across the available starting zones.
    A number n_by_cell determines how many simulations should start within each cell (the same position can be chosen multiple times).

    Parameters
    ----------
    agent : Agent
        The agent to be tested
    cell_width : int, default=10
        The size of the sides of each cells to be considered.
    n_by_cell : int, default=10
        How many simulations should start within each cell.
    environment : Environment, optional
        The environment to run the simulations in.
        By default, the environment linked to the agent will used.
        This parameter is intended if the environment needs to be modified compared to environment the agent was trained on.
    time_shift : int or np.ndarray, default=0
        The time at which to start the olfactory simulation array.
        It can be either a single value, or n values.
    time_loop : bool, default=True
        Whether to loop the time if reaching the end. (starts back at 0)
    horizon : int, default=1000
        The amount of steps to run the simulation for before killing the remaining simulations.
    reward_discount : float, default=0.99
        How much a given reward is discounted based on how long it took to get it.
        It is purely used to compute the Average Discount Reward (ADR) after the simulation.
    print_progress : bool, default=True
        Wheter to show a progress bar of what step the simulations are at.
    print_stats : bool, default=True
        Wheter to print the stats at the end of the run.
    use_gpu : bool, default=False
        Whether to run the simulations on the GPU or not.
    
    Returns
    -------
    hist : SimulationHistory
        A SimulationHistory object that tracked all the positions, actions and observations.
    '''
    # Handle the case an specific environment is given
    environment_provided = environment is not None
    if environment_provided:
        assert environment.shape == agent.environment.shape, "The provided environment's shape doesn't match the environment has been trained on..."
    else:
        environment = agent.environment
    
    # Gathering starting points
    cells_x = int(environment.shape[0] / cell_width)
    cells_y = int(environment.shape[1] / cell_width)

    indices = np.arange(np.prod(environment.shape), dtype=int)
    indices_grid = indices.reshape(environment.shape)
    all_chosen_indices = []

    for i in range(cells_x):
        for j in range(cells_y):
            cell_probs = environment.start_probabilities[(i*cell_width):(i*cell_width)+cell_width, (j*cell_width):(j*cell_width)+cell_width]
            if np.any(cell_probs > 0):
                cell_indices = indices_grid[(i*cell_width):(i*cell_width)+cell_width, (j*cell_width):(j*cell_width)+cell_width]
                cell_probs /= np.sum(cell_probs)

                chosen_indices = np.random.choice(cell_indices.ravel(), size=n_by_cell, replace=True, p=cell_probs.ravel()).tolist()
                all_chosen_indices += chosen_indices

    n = len(all_chosen_indices)
    start_points = np.array(np.unravel_index(all_chosen_indices, environment.shape)).T

    return run_test(
        agent=agent,
        n=n,
        start_points=start_points,
        environment=environment if environment_provided else None,
        time_shift=time_shift,
        time_loop=time_loop,
        horizon=horizon,
        reward_discount=reward_discount,
        print_progress=print_progress,
        print_stats=print_stats,
        use_gpu=use_gpu
    )