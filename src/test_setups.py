import numpy as np

from src.agent import Agent
from src.environment import Environment
from src.simulation import run_test, SimulationHistory


def run_all_starts_test(
             agent:Agent,
             environment:Environment|None=None,
             time_shift:int|np.ndarray=0,
             horizon:int=1000,
             reward_discount:float=0.99,
             print_progress:bool=True,
             print_stats:bool=True,
             use_gpu:bool=False
             ) -> SimulationHistory:
    '''
    # TODO
    '''
        # Handle the case an specific environment is given
    if environment is not None:
        assert environment.shape == agent.environment.shape, "The provided environment's shape doesn't match the environment has been trained on..."
        print('Using the provided environment, not the agent environment.')
    else:
        environment = agent.environment
    
    # Gathering starting points
    start_points = np.argwhere(environment.start_probabilities > 0)
    n = len(start_points)

    return run_test(
        agent=agent,
        n=n,
        start_points=start_points,
        environment=environment,
        time_shift=time_shift,
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
             horizon:int=1000,
             reward_discount:float=0.99,
             print_progress:bool=True,
             print_stats:bool=True,
             use_gpu:bool=False
             ) -> SimulationHistory:
    '''
    # TODO
    '''
        # Handle the case an specific environment is given
    if environment is not None:
        assert environment.shape == agent.environment.shape, "The provided environment's shape doesn't match the environment has been trained on..."
        print('Using the provided environment, not the agent environment.')
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
        environment=environment,
        time_shift=time_shift,
        horizon=horizon,
        reward_discount=reward_discount,
        print_progress=print_progress,
        print_stats=print_stats,
        use_gpu=use_gpu
    )