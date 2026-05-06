import numpy as np
import os
import pandas as pd
import psutil
import time

from datetime import datetime
from tqdm.auto import tqdm
from typing import Literal

from olfactory_navigation.agent import Agent
from olfactory_navigation.environment import Environment
from olfactory_navigation.simulation import run_test, SimulationHistory


def run_all_starts_test(agent: Agent,
                        environment: Environment = None,
                        time_shift: int | np.ndarray = 0,
                        time_loop: bool = True,
                        horizon: int = 1000,
                        initialization_values: dict = {},
                        reward_discount: float = 0.99,
                        distance_metric: Literal['l1', 'l2'] = 'l1',
                        print_progress: bool = True,
                        print_stats: bool = True,
                        print_warning: bool = True,
                        use_gpu: bool = False,
                        parallel_agent_simulation: bool = True,
                        batches: int = -1
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
    time_shift : int or np.ndarray, default = 0
        The time at which to start the olfactory simulation array.
        It can be either a single value, or n values.
    time_loop : bool, default = True
        Whether to loop the time if reaching the end. (starts back at 0)
    horizon : int, default = 1000
        The amount of steps to run the simulation for before killing the remaining simulations.
    initialization_values : dict, default = {}
        In the case the agent is to be initialized with custom values,
        the paramaters to be passed on the initialize_state function can be set here.
    reward_discount : float, default = 0.99
        How much a given reward is discounted based on how long it took to get it.
        It is purely used to compute the Average Discount Reward (ADR) after the simulation.
    distance_metric : "l1" or "l2", default = "l1"
        The distance metric used to compute for example the distance between the starting points and the goal after the simulation.
        This is done in order to compute the extra steps and t_min over t metrics for example.
    print_progress : bool, default = True
        Wheter to show a progress bar of what step the simulations are at.
    print_stats : bool, default = True
        Wheter to print the stats at the end of the run.
    print_warning : bool, default = True
        Whether to print warnings when they occur or not.
    use_gpu : bool, default = False
        Whether to run the simulations on the GPU or not.
    parallel_agent_simulation : bool, default = True
        Whether to run the agent simulations in parallel or sequentially (ie in batches of 1).
    batches : int, default = -1
        In how many batches the simulations should be run.
        This is useful in the case there are too many simulations and the memory can fill up.
        The value of batches=-1 will make it that different batches amount are tried in increasing order if a MemoryError is encountered.

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
        agent = agent,
        n = n,
        start_points = start_points,
        environment = environment if environment_provided else None,
        time_shift = time_shift,
        time_loop = time_loop,
        horizon = horizon,
        initialization_values = initialization_values,
        reward_discount = reward_discount,
        distance_metric = distance_metric,
        print_progress = print_progress,
        print_stats = print_stats,
        print_warning = print_warning,
        use_gpu = use_gpu,
        parallel_agent_simulation = parallel_agent_simulation,
        batches = batches
    )


def run_n_by_cell_test(agent: Agent,
                       cell_width: int = 10,
                       n_by_cell: int = 10,
                       environment: Environment = None,
                       time_shift: int | np.ndarray = 0,
                       time_loop: bool = True,
                       horizon: int = 1000,
                       initialization_values: dict = {},
                       reward_discount: float = 0.99,
                       distance_metric: Literal['l1', 'l2'] = 'l1',
                       print_progress: bool = True,
                       print_stats: bool = True,
                       print_warning: bool = True,
                       use_gpu: bool = False,
                       parallel_agent_simulation: bool = True,
                       batches: int = -1
                       ) -> SimulationHistory:
    '''
    Function to run a test with simulations starting in different cells across the available starting zones.
    A number n_by_cell determines how many simulations should start within each cell (the same position can be chosen multiple times).

    Parameters
    ----------
    agent : Agent
        The agent to be tested
    cell_width : int, default = 10
        The size of the sides of each cells to be considered.
    n_by_cell : int, default = 10
        How many simulations should start within each cell.
    environment : Environment, optional
        The environment to run the simulations in.
        By default, the environment linked to the agent will used.
        This parameter is intended if the environment needs to be modified compared to environment the agent was trained on.
    time_shift : int or np.ndarray, default = 0
        The time at which to start the olfactory simulation array.
        It can be either a single value, or n values.
    time_loop : bool, default = True
        Whether to loop the time if reaching the end. (starts back at 0)
    horizon : int, default = 1000
        The amount of steps to run the simulation for before killing the remaining simulations.
    initialization_values : dict, default = {}
        In the case the agent is to be initialized with custom values,
        the paramaters to be passed on the initialize_state function can be set here.
    reward_discount : float, default = 0.99
        How much a given reward is discounted based on how long it took to get it.
        It is purely used to compute the Average Discount Reward (ADR) after the simulation.
    distance_metric : "l1" or "l2", default = "l1"
        The distance metric used to compute for example the distance between the starting points and the goal after the simulation.
        This is done in order to compute the extra steps and t_min over t metrics for example.
    print_progress : bool, default = True
        Wheter to show a progress bar of what step the simulations are at.
    print_stats : bool, default = True
        Wheter to print the stats at the end of the run.
    print_warning : bool, default = True
        Whether to print warnings when they occur or not.
    use_gpu : bool, default = False
        Whether to run the simulations on the GPU or not.
    parallel_agent_simulation : bool, default = True
        Whether to run the agent simulations in parallel or sequentially (ie in batches of 1).
    batches : int, default = -1
        In how many batches the simulations should be run.
        This is useful in the case there are too many simulations and the memory can fill up.
        The value of batches=-1 will make it that different batches amount are tried in increasing order if a MemoryError is encountered.

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
        agent = agent,
        n = n,
        start_points = start_points,
        environment = environment if environment_provided else None,
        time_shift = time_shift,
        time_loop = time_loop,
        horizon = horizon,
        initialization_values = initialization_values,
        reward_discount = reward_discount,
        distance_metric = distance_metric,
        print_progress = print_progress,
        print_stats = print_stats,
        print_warning = print_warning,
        use_gpu = use_gpu,
        parallel_agent_simulation = parallel_agent_simulation,
        batches = batches
    )


def analyse_shape_robustness(all_histories: list[SimulationHistory],
                             multipliers: np.ndarray
                             ) -> pd.DataFrame:
    '''
    Function to generate an analysis of a set of simulation tests with different multipliers applied in the environment.
    It returns a pandas dataframe summarizing the results for each multiplier pairs.
    The results analyzed are the following:

    - convergence
    - steps taken
    - discounted rewards
    - extra steps taken (compared to a minimum path)
    - t min over t (a ratio of how optimal the path taken was)

    For each result, the mean, standard deviation along with the mean and standard deviation of the successful trajectories are recorded.

    Parameters
    ----------
    all_histories : list[SimulationHistory]
        A list of all the simulation histories to summarize
    multipliers : np.ndarray
        An array of the multiplier pairs used (for the y multiplier then the x multiplier)

    Returns
    -------
    df : pd.DataFrame
        The analysis dataframe.
    '''
    rows = []
    # For each simulation history and multiplier, the analysis dataframe is extracted
    for hist, multiplier_pair in zip(all_histories, multipliers):
        df = hist.general_analysis_df

        # Then the summarized metrics are collapsed on a single row
        col_metric_dict = {'y_multiplier': multiplier_pair[0].astype(int), 'x_multiplier': multiplier_pair[1].astype(int)}
        for col in ['converged', 'reached_horizon', 'steps_taken', 'discounted_rewards', 'extra_steps', 't_min_over_t']:
            for metric in ['mean', 'standard_deviation', 'success_mean', 'success_standard_deviation']:
                col_metric_dict[f'{col}_{metric}'] = df.loc[metric, col]

        rows.append(col_metric_dict)

    # Creating the dataframe from all the rows
    df = pd.DataFrame(rows)

    # Removal of 4 unnecessary columns
    df = df.drop(columns=['converged_success_mean',
                          'converged_success_standard_deviation',
                          'reached_horizon_success_mean',
                          'reached_horizon_success_standard_deviation'])

    return df


def test_shape_robustness(agent: Agent,
                          environment: Environment = None,
                          time_shift: int | np.ndarray = 0,
                          time_loop: bool = True,
                          horizon: int = 1000,
                          initialization_values: dict = {},
                          reward_discount: float = 0.99,
                          distance_metric: Literal['l1', 'l2'] = 'l1',
                          step_percentage: int = 20,
                          min_percentage:int = 20,
                          max_percentage:int = 200,
                          multipliers: list[int] = None,
                          print_progress: bool = True,
                          print_stats: bool = True,
                          print_warning: bool = True,
                          use_gpu: bool = False,
                          parallel_agent_simulation: bool = True,
                          batches: int = -1,
                          save: bool = True,
                          save_folder: str = None,
                          save_analysis: bool = True
                          ) -> list[SimulationHistory]:
    '''
    Function to test the robustness of an agent in a environment where the odor plume's shape is altered by some percentage.

    A list of multipliers will be constructed from the min_percentage to 100% and up to max_percentage values with between each percentage step_percentage values.
    These percentage multipliers will be applied both in the x and y direction but cropped to the largest allowed multiplier along each axis.

    For each multiplier pair, a completed test will be run. This complete test consists in running from all possible start positions of the original environment.

    Parameters
    ----------
    agent : Agent
        The agent to run the shape robustness test on.
    environment : Environment, optional
        The environment to run the test in.
        By default, the environment linked to the agent will used.
        This parameter is intended if the environment needs to be modified compared to environment the agent was trained on.
    time_shift : int or np.ndarray, default = 0
        The time at which to start the olfactory simulation array.
        It can be either a single value, or n values.
    time_loop : bool, default = True
        Whether to loop the time if reaching the end. (starts back at 0)
    horizon : int, default = 1000
        The amount of steps to run the simulation for before killing the remaining simulations.
    initialization_values : dict, default = {}
        In the case the agent is to be initialized with custom values,
        the paramaters to be passed on the initialize_state function can be set here.
    reward_discount : float, default = 0.99
        How much a given reward is discounted based on how long it took to get it.
        It is purely used to compute the Average Discount Reward (ADR) after the simulation.
    distance_metric : "l1" or "l2", default = "l1"
        The distance metric used to compute for example the distance between the starting points and the goal after the simulation.
        This is done in order to compute the extra steps and t_min over t metrics for example.
    step_percentage : int, default = 20
        Starting at 100%, how much of a percentage step to do to reach the min and max percentages.
    min_percentage : int, default = 20
        The minimum percentage of deformation to apply on the environment's odor plume.
    max_percentage : int, default = 200
        The maximum percentage of deformation to apply on the environment's odor plume.
        If this value is larger than the maximum shape allowed by the margins, the largest allowed percentage will be used.
    multipliers : list[int], optional
        If provided, the step_percentage, min_percentage and max_percentage parameters will be ignored.
        A list of percentages of deformations to use to deforme the environment's odor plume.
    print_progress : bool, default = True
        Whether to display a progress bar of how many test have been performed so far.
    print_stats : bool, default = True
        Whether to display statistics at the end of each test.
    print_warning : bool, default = True
        Whether to print warnings when they occur or not.
    use_gpu : bool, default = False
        Whether to use the GPU to speed up the tests.
    parallel_agent_simulation : bool, default = True
        Whether to run the agent simulations in parallel or sequentially (ie in batches of 1).
    batches : int, default = -1
        In how many batches the simulations should be run.
        This is useful in the case there are too many simulations and the memory can fill up.
        The value of batches=-1 will make it that different batches amount are tried in increasing order if a MemoryError is encountered.
    save : bool, default = True
        Whether to save the results of each test to a save_folder.
        Each test's result will be under the name 'test_env_y-<y_multiplier>_x-<x_multiplier>.csv'
    save_folder : str, optional
        The path to which the test results are saved.
        If not provided, it will automatically create a new folder './results/<timestamp>_shape_robustness_test_<environment_name>/'
    save_analysis : bool, default = True
        Whether to save the analysis of the histories.
        It will be saved under a file named '_analysis.csv' in the save_folder.

    Returns
    -------
    all_histories : list[SimulationHistory]
        A list of SimulationHistory instances.
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

    # Generating multipliers
    if multipliers is None:
        with np.errstate(divide='ignore'):
            low_max_mult = ((environment.margins[:,0] / environment.data_source_position) + 1)
            high_max_mult = (1 + (environment.margins[:,1] / (environment.data_shape - environment.data_source_position)))
            max_mult = np.min(np.vstack([low_max_mult, high_max_mult]), axis=0)

        multipliers = [(100 - perc_mult) for perc_mult in range(0, (100-min_percentage)+step_percentage, step_percentage)[1:]] + [perc_mult for perc_mult in range(100, min(max_percentage, int(max(max_mult)*100)), step_percentage)]
    multipliers.sort()

    # Generating all combinations of multipliers
    mult_combinations = np.array(np.meshgrid(multipliers, multipliers, indexing='xy')).T.reshape(-1,2).astype(float)
    mult_combinations /= 100
    mult_combinations = mult_combinations[np.all(mult_combinations < max_mult, axis=1), :]

    # Save Folder name and creation
    if save or save_analysis:
        if save_folder is None:
            save_folder = f'./results/{datetime.now().strftime("%Y%m%d_%H%M%S")}_shape_robustness_test_' + environment.name

        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        print(f'The results will be saved to: {save_folder}\n')

    all_histories = []
    for mults in (tqdm(mult_combinations) if print_progress else mult_combinations):
        print(f'Testing on environment with height {int(mults[0]*100)}% and width {int(mults[1] * 100)}%')

        # Modifying environment
        modified_environment = environment.modify(multiplier=mults)

        # Running test
        hist = run_test(
            agent = agent,
            n = n,
            start_points = start_points,
            environment = modified_environment,
            time_shift = time_shift,
            time_loop = time_loop,
            horizon = horizon,
            initialization_values = initialization_values,
            reward_discount = reward_discount,
            distance_metric = distance_metric,
            print_progress = False,
            print_stats = print_stats,
            print_warning = print_warning,
            use_gpu = use_gpu,
            parallel_agent_simulation = parallel_agent_simulation,
            batches = batches
        )

        all_histories.append(hist)

        # Saving history
        if save:
            file_name = f'test_env_y-{int(mults[0]*100)}_x-{int(mults[1]*100)}'
            hist.save(file=file_name,
                      folder=save_folder,
                      save_analysis=False)

        print()

    # Analysis saving
    if save and save_analysis:
        analysis_df = analyse_shape_robustness(all_histories=all_histories, multipliers=(mult_combinations*100))
        analysis_file_name = '_analysis.csv'
        analysis_df.to_csv(save_folder + '/' + analysis_file_name, index=False)
        print(f'Shape robustness analysis saved to: {save_folder}/{analysis_file_name}')

    return all_histories


def analyse_scale_robustness(all_histories: list[SimulationHistory],
                             multipliers: np.ndarray
                             ) -> pd.DataFrame:
    '''
    Function to generate an analysis of a set of simulation tests with different multipliers applied in the environment.
    It returns a pandas dataframe summarizing the results for each multiplier pairs.
    The results analyzed are the following:

    - convergence
    - steps taken
    - discounted rewards
    - extra steps taken (compared to a minimum path)
    - t min over t (a ratio of how optimal the path taken was)

    For each result, the mean, standard deviation along with the mean and standard deviation of the successful trajectories are recorded.

    Parameters
    ----------
    all_histories : list[SimulationHistory]
        A list of all the simulation histories to summarize
    multipliers : np.ndarray
        An array of the multiplier pairs used (for the y multiplier then the x multiplier)

    Returns
    -------
    df : pd.DataFrame
        The analysis dataframe.
    '''
    rows = []
    # For each simulation history and multiplier, the analysis dataframe is extracted
    for hist, multiplier in zip(all_histories, multipliers):
        df = hist.general_analysis_df

        # Then the summarized metrics are collapsed on a single row
        col_metric_dict = {'multiplier': int(multiplier)}
        for col in ['converged', 'reached_horizon', 'steps_taken', 'discounted_rewards', 'extra_steps', 't_min_over_t']:
            for metric in ['mean', 'standard_deviation', 'success_mean', 'success_standard_deviation']:
                col_metric_dict[f'{col}_{metric}'] = df.loc[metric, col]

        rows.append(col_metric_dict)

    # Creating the dataframe from all the rows
    df = pd.DataFrame(rows)

    # Removal of 4 unnecessary columns
    df = df.drop(columns=['converged_success_mean',
                          'converged_success_standard_deviation',
                          'reached_horizon_success_mean',
                          'reached_horizon_success_standard_deviation'])

    return df


def test_scale_robustness(agent: Agent,
                          environment: Environment = None,
                          time_shift: int | np.ndarray = 0,
                          time_loop: bool = True,
                          horizon: int = 1000,
                          initialization_values: dict = {},
                          reward_discount: float = 0.99,
                          distance_metric: Literal['l1', 'l2'] = 'l1',
                          step_percentage: int = 20,
                          min_percentage:int = 20,
                          max_percentage:int = 200,
                          multipliers: list[int] = None,
                          print_progress: bool = True,
                          print_stats: bool = True,
                          print_warning: bool = True,
                          use_gpu: bool = False,
                          parallel_agent_simulation: bool = True,
                          batches: int = -1,
                          save: bool = True,
                          save_folder: str = None,
                          save_analysis: bool = True
                          ) -> list[SimulationHistory]:
    '''
    Function to test the robustness of an agent in a environment where the scale of the environment's shape is altered by some percentage.

    A list of multipliers will be constructed from the min_percentage to 100% and up to max_percentage values with between each percentage step_percentage values.
    These percentage multipliers will be applied both in the x and y direction but cropped to the largest allowed multiplier along each axis.

    This complete test consists in running from all possible start positions of the original environment.

    Parameters
    ----------
    agent : Agent
        The agent to run the shape robustness test on.
    environment : Environment, optional
        The environment to run the test in.
        By default, the environment linked to the agent will used.
        This parameter is intended if the environment needs to be modified compared to environment the agent was trained on.
    time_shift : int or np.ndarray, default = 0
        The time at which to start the olfactory simulation array.
        It can be either a single value, or n values.
    time_loop : bool, default = True
        Whether to loop the time if reaching the end. (starts back at 0)
    horizon : int, default = 1000
        The amount of steps to run the simulation for before killing the remaining simulations.
    initialization_values : dict, default = {}
        In the case the agent is to be initialized with custom values,
        the paramaters to be passed on the initialize_state function can be set here.
    reward_discount : float, default = 0.99
        How much a given reward is discounted based on how long it took to get it.
        It is purely used to compute the Average Discount Reward (ADR) after the simulation.
    distance_metric : "l1" or "l2", default = "l1"
        The distance metric used to compute for example the distance between the starting points and the goal after the simulation.
        This is done in order to compute the extra steps and t_min over t metrics for example.
    step_percentage : int, default = 20
        Starting at 100%, how much of a percentage step to do to reach the min and max percentages.
    min_percentage : int, default = 20
        The minimum percentage of deformation to apply on the environment's odor plume.
    max_percentage : int, default = 200
        The maximum percentage of deformation to apply on the environment's odor plume.
        If this value is larger than the maximum shape allowed by the margins, the largest allowed percentage will be used.
    multipliers : list[int], optional
        If provided, the step_percentage, min_percentage and max_percentage parameters will be ignored.
        A list of percentages of deformations to use to deforme the environment's odor plume.
    print_progress : bool, default = True
        Whether to display a progress bar of how many test have been performed so far.
    print_stats : bool, default = True
        Whether to display statistics at the end of each test.
    print_warning : bool, default = True
        Whether to print warnings when they occur or not.
    use_gpu : bool, default = False
        Whether to use the GPU to speed up the tests.
    parallel_agent_simulation : bool, default = True
        Whether to run the agent simulations in parallel or sequentially (ie in batches of 1).
    batches : int, default = -1
        In how many batches the simulations should be run.
        This is useful in the case there are too many simulations and the memory can fill up.
        The value of batches=-1 will make it that different batches amount are tried in increasing order if a MemoryError is encountered.
    save : bool, default = True
        Whether to save the results of each test to a save_folder.
        Each test's result will be under the name 'test_env_mult-<multiplier>.csv'
    save_folder : str, optional
        The path to which the test results are saved.
        If not provided, it will automatically create a new folder './results/<timestamp>_scale_robustness_test_<environment_name>/'
    save_analysis : bool, default = True
        Whether to save the analysis of the histories.
        It will be saved under a file named '_analysis.csv' in the save_folder.

    Returns
    -------
    all_histories : list[SimulationHistory]
        A list of SimulationHistory instances.
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

    # Generating multipliers
    if multipliers is None:
        with np.errstate(divide='ignore'):
            low_max_mult = ((environment.margins[:,0] / environment.data_source_position) + 1)
            high_max_mult = (1 + (environment.margins[:,1] / (environment.data_shape - environment.data_source_position)))
            max_mult = np.min(np.vstack([low_max_mult, high_max_mult]), axis=0)

        multipliers = [(100 - perc_mult) for perc_mult in range(0, (100-min_percentage)+step_percentage, step_percentage)[1:]] + [perc_mult for perc_mult in range(100, min(max_percentage, int(max(max_mult)*100)), step_percentage)]
    multipliers.sort()

    # Save Folder name and creation
    if save or save_analysis:
        if save_folder is None:
            save_folder = f'./results/{datetime.now().strftime("%Y%m%d_%H%M%S")}_scale_robustness_test_' + environment.name

        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        print(f'The results will be saved to: {save_folder}\n')

    all_histories = []
    for mult in (tqdm(multipliers) if print_progress else multipliers):
        print(f'Testing on environment with scale modifier {mult}%')

        # Modifying environment
        modified_environment = environment.modify_scale(scale_factor=mult/100)

        # Running test
        hist = run_test(
            agent = agent,
            n = n,
            start_points = start_points,
            environment = modified_environment,
            time_shift = time_shift,
            time_loop = time_loop,
            horizon = horizon,
            initialization_values = initialization_values,
            reward_discount = reward_discount,
            distance_metric = distance_metric,
            print_progress = False,
            print_stats = print_stats,
            print_warning = print_warning,
            use_gpu = use_gpu,
            parallel_agent_simulation = parallel_agent_simulation,
            batches = batches
        )

        all_histories.append(hist)

        # Saving history
        if save:
            file_name = f'test_env_mult-{mult}'
            hist.save(file=file_name,
                      folder=save_folder,
                      save_analysis=False)

        print()

    # Analysis saving
    if save and save_analysis:
        analysis_df = analyse_scale_robustness(all_histories=all_histories, multipliers=multipliers)
        analysis_file_name = '_analysis.csv'
        analysis_df.to_csv(save_folder + '/' + analysis_file_name, index=False)
        print(f'Scale robustness analysis saved to: {save_folder}/{analysis_file_name}')

    return all_histories


def test_agents(*agents: Agent,
                environments: list[Environment],
                time_shift: int | np.ndarray = 0,
                time_loop: bool = True,
                horizon: int = 1000,
                initialization_values: list[dict] = None,
                reward_discount: float = 0.99,
                distance_metric: Literal['l1', 'l2'] = 'l1',
                print_progress: bool = False,
                print_stats: bool = True,
                print_warning: bool = True,
                use_gpu: bool = False,
                parallel_agent_simulation: bool = True,
                batches: int = -1,
                save_histories_path: str = None,
                save_result_table: bool = True
                ) -> pd.DataFrame:
    '''
    A function to test multiple (trained) agents in multiple given environments.

    A summary table will be generated to compare the performance of the various agents within each environment.

    Parameters
    ----------
    agents : Agent
        The agents to test. They must be already trained.
    environments : list[Environment]
        The environment to test the agents in.
    time_shift : int | np.ndarray, default = 0
        By how many steps to shift the t0 of the environment.
        It can be fixed or for each starting point of the simulation (in such case the amount of starting points must be same in each environments).
    time_loop : bool, default = True
        Whether the simulation t should loop back to 0 when reaching the max t of the given environment.
    horizon : int, default = 1000
        For how many steps the simulation should run for.
    initialization_values : dict, optional
        In the case the agents are to be initialized with custom values,
        the paramaters to be passed on the initialize_state function can be set here.
        If provided, one dict must be provided per agent.
    reward_discount : float, default = 0.99
        The reward discount that is used to compare the cummulative discounted reward.
    distance_metric : "l1" or "l2", default = "l1"
        The distance metric used to compute for example the distance between the starting points and the goal after the simulation.
        This is done in order to compute the extra steps and t_min over t metrics for example.
    print_progress : bool, default = False
        Whether to show a progress bar for the simulations.
    print_stats : bool, default = True
        Whether to print the stats (results) after each simulation.
    print_warning : bool, default = True
        Whether to print warnings when they occur or not.
    use_gpu : bool, default = False
        Whether to use the gpu to speedup testing.
    parallel_agent_simulation : bool, default = True
        Whether to run the agent simulations in parallel or sequentially (ie in batches of 1).
    batches : int, default = -1
        In how many batches the simulations should be run.
        This is useful in the case there are too many simulations and the memory can fill up.
        The value of batches=-1 will make it that different batches amount are tried in increasing order if a MemoryError is encountered.
    save_histories_path : str, optional
        If the details of the simulation histories are to be saved, a path can be provided here.
    save_result_table : bool, default = True
        Whether the returned table should also be saved, it will be saved at the save_histories_path if it is set.

    Returns
    -------
    simulations_comparison_df : pd.DataFrame
        A table with as row indices (agent, environment) pairs and columns the same columns as the output of SimulationHistory.compare_all.
    '''
    simulation_histories = []

    # Processing the initialization_values parameter
    if initialization_values is None:
        initialization_values = [{}] * len(agents)

    # Loop over the provided agents
    for i_agent, (agent, agent_initialization_values) in enumerate(zip(agents, initialization_values)):
        print(f'Testing Agent {i_agent}:')

        if not agent.trained:
            print(f'[Warning] Skipping agent {i_agent} due to it not being marked as trained...')
            continue

        agent_histories = []
        for i_environment, environment in enumerate(environments):

            print(f'- Environment {i_environment}')
            hist = run_all_starts_test(
                agent = agent,
                environment = environment,
                time_shift = time_shift,
                time_loop = time_loop,
                horizon = horizon,
                initialization_values = agent_initialization_values,
                reward_discount = reward_discount,
                distance_metric = distance_metric,
                print_progress = print_progress,
                print_warning = print_warning,
                print_stats = print_stats,
                use_gpu = use_gpu,
                parallel_agent_simulation = parallel_agent_simulation,
                batches = batches
            )

            agent_histories.append(hist)
            print('')

            # Save simulation history if requested
            if save_histories_path is not None:
                hist.save(file=f'Simualtions-agent_{i_agent}-environment_{i_environment}', folder=save_histories_path, save_analysis=False)

        simulation_histories.append(agent_histories)
        print('--------------------------------------')

    # Generating comparison result table
    all_agent_comparison_dfs = []
    for agent_simulation_histories in simulation_histories:
        agent_comparison_df = SimulationHistory.compare_all(agent_simulation_histories)
        agent_comparison_df['environment'] = [f'environment_{i}' for i in range(len(environments))]
        agent_comparison_df.set_index('environment')

        all_agent_comparison_dfs.append(agent_comparison_df)

    simulations_comparison_df: pd.DataFrame = pd.concat(all_agent_comparison_dfs, keys=[f'agent_{i}' for i in range(len(agents))], names='agent')

    # Save comparison table if needed
    if save_result_table:
        folder = './' if save_histories_path is None else save_histories_path
        file = 'Simulation_comparison-' + datetime.now().strftime('%Y%m%d_%H%M%S%f')
        simulations_comparison_df.to_csv(folder+file)

    return simulations_comparison_df


def train_and_test_agents(*agent_classes: type[Agent],
                          environments: list[Environment],
                          agent_thresholds: float | list[float] = 3e-6,
                          agent_space_aware: bool = False,
                          agent_spacial_subdivisions: np.ndarray = None,
                          agent_actions: dict[str, np.ndarray] | np.ndarray = None,
                          agent_additional_parameters: list[dict] = None, # Has to be as long the agents
                          training_environment: Environment = None,
                          training_parameters: list[dict] = None, # Has to be as long the agents
                          time_shift: int | np.ndarray = 0,
                          time_loop: bool = True,
                          horizon: int = 1000,
                          initialization_values: list[dict] = None, # Has to be as long the agents
                          reward_discount: float = 0.99,
                          distance_metric: Literal['l1', 'l2'] = 'l1',
                          print_progress: bool = False,
                          print_stats: bool = True,
                          print_warning: bool = True,
                          use_gpu: bool = False,
                          parallel_agent_simulation: bool = True,
                          batches: int = -1,
                          save_histories_path: str = None,
                          save_result_table: bool = True
                          ) -> pd.DataFrame:
    '''
    A function to train (with a given training_environment) and test multiple agents in multiple given environments.

    A summary table will be generated to compare the performance of the various agents within each environment.

    Parameters
    ----------
    agent_classes: type[Agent]
        The classes of the agents to create, train and test.
    environments : list[Environment]
        The environment to test the agents in.
    agent_thresholds : float | list[float], default = 3e-6
        The olfactory thresholds. If an odor cue above this threshold is detected, the agent detects it, else it does not.
        If a list of thresholds is provided, the agent should be able to detect |thresholds|+1 levels of odor.
    agent_space_aware : bool, default = False
        Whether the agent is aware of it's own position in space.
    agent_spacial_subdivisions : np.ndarray, optional
        How many spacial compartments the agent has to internally represent the space it lives in.
        By default, it will be as many as there are grid points in the environment.
    agent_actions : dict[str, np.ndarray] | np.ndarray, optional
        The set of action available to the agent. It should match the type of environment (ie: if the environment has layers, it should contain a layer component to the action vector, and similarly for a third dimension).
        Else, a dict of strings and action vectors where the strings represent the action labels.
        If none is provided, by default, all unit steps in all cardinal directions are included and such for all layers (if the environment has layers.)
    agent_additional_parameters : list[dict], optional
        Any additional parameters to pass over to the agent constructor.
        The list needs to be as long as the amount of agents provided.
    training_parameters : list[dict], optional
        Any additional parameters to pass over to the agent's training process.
        The list needs to be as long as the amount of agents provided.
    time_shift : int | np.ndarray, default = 0
        By how many steps to shift the t0 of the environment.
        It can be fixed or for each starting point of the simulation (in such case the amount of starting points must be same in each environments).
    time_loop : bool, default = True
        Whether the simulation t should loop back to 0 when reaching the max t of the given environment.
    horizon : int, default = 1000
        For how many steps the simulation should run for.
    initialization_values : dict, optional
        In the case the agents are to be initialized with custom values,
        the paramaters to be passed on the initialize_state function can be set here.
        If provided, one dict must be provided per agent.
    reward_discount : float, default = 0.99
        The reward discount that is used to compare the cummulative discounted reward.
    distance_metric : "l1" or "l2", default = "l1"
        The distance metric used to compute for example the distance between the starting points and the goal after the simulation.
        This is done in order to compute the extra steps and t_min over t metrics for example.
    print_progress : bool, default = False
        Whether to show a progress bar for the simulations.
    print_stats : bool, default = True
        Whether to print the stats (results) after each simulation.
    print_warning : bool, default = True
        Whether to print warnings when they occur or not.
    use_gpu : bool, default = False
        Whether to use the gpu to speedup testing.
    parallel_agent_simulation : bool, default = True
        Whether to run the agent simulations in parallel or sequentially (ie in batches of 1).
    batches : int, default = -1
        In how many batches the simulations should be run.
        This is useful in the case there are too many simulations and the memory can fill up.
        The value of batches=-1 will make it that different batches amount are tried in increasing order if a MemoryError is encountered.
    save_histories_path : str, optional
        If the details of the simulation histories are to be saved, a path can be provided here.
    save_result_table : bool, default = True
        Whether the returned table should also be saved, it will be saved at the save_histories_path if it is set.

    Returns
    -------
    simulations_comparison_df : pd.DataFrame
        A table with as row indices (agent, environment) pairs and columns the same columns as the output of SimulationHistory.compare_all.
    '''
    training_stats = []
    simulation_histories = []

    # Processing the initialization_values parameter
    if initialization_values is None:
        initialization_values = [{}] * len(agent_classes)

    # Print warning in case no training environment is set because the agent will be retrained for each environment
    if training_environment is None and print_warning:
        print('[Warning] The training environment has not been provided so the agent will be re-trained for each environment provided...')

    # Loop through all agents
    for i_agent, (agent_class, agent_additional_params, training_params, agent_initialization_values) in enumerate(zip(agent_classes, agent_additional_parameters, training_parameters, initialization_values)):
        print(f'Agent {i_agent} ({agent_class.__name__}):')

        if training_environment is not None:
            print(f'- Training...')

            agent: Agent = agent_class(
                environment = training_environment,
                thresholds = agent_thresholds,
                space_aware = agent_space_aware,
                spatial_subdivisions = agent_spacial_subdivisions,
                actions = agent_actions,
                **agent_additional_params)

            # Tracking
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss
            start_time = time.perf_counter()

            if not agent.trained:
                agent.train(**training_params)

            # End tracking
            end_time = time.perf_counter()
            memory_after = process.memory_info().rss

            # Saving tracking
            training_stats.append({'memory_used': memory_after - memory_before,
                                   'time_taken': end_time - start_time})

            # Testing trained agent on all environments
            agent_simulation_histories = []
            for i_environment, environment in enumerate(environments):
                print(f'- Testing environment {i_environment}')

                hist = run_all_starts_test(
                    agent = agent,
                    environment = environment,
                    time_shift = time_shift,
                    time_loop = time_loop,
                    horizon = horizon,
                    initialization_values = agent_initialization_values,
                    reward_discount = reward_discount,
                    distance_metric = distance_metric,
                    print_progress = print_progress,
                    print_stats = print_stats,
                    print_warning = print_warning,
                    use_gpu = use_gpu,
                    parallel_agent_simulation = parallel_agent_simulation,
                    batches = batches
                )

                agent_simulation_histories.append(hist)
                print('')

                # Save simulation history if requested
                if save_histories_path is not None:
                    hist.save(file=f'Simualtions-agent_{i_agent}-environment_{i_environment}', folder=save_histories_path, save_analysis=False)

            simulation_histories.append(agent_simulation_histories)
            print('--------------------------------------')

        # No training environment
        else:
            agent_histories = []
            agent_training_stats = []

            # Loop through the environments
            for i_environment, environment in enumerate(environments):
                print(f'- Environment {i_environment}')

                agent: Agent = agent_class(
                    environment = environment,
                    thresholds = agent_thresholds,
                    space_aware = agent_space_aware,
                    spatial_subdivisions = agent_spacial_subdivisions,
                    actions = agent_actions,
                    **agent_additional_params
                    )

                # Tracking
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss
                start_time = time.perf_counter()

                agent.train(**training_params)

                # End tracking
                end_time = time.perf_counter()
                memory_after = process.memory_info().rss

                # Saving tracking
                agent_training_stats.append({'memory_used': memory_after - memory_before,
                                             'time_taken': end_time - start_time})

                hist = run_all_starts_test( # No environment provided because it will use the same environment is trained on
                    agent = agent,
                    time_shift = time_shift,
                    time_loop = time_loop,
                    horizon = horizon,
                    initialization_values = agent_initialization_values,
                    reward_discount = reward_discount,
                    distance_metric = distance_metric,
                    print_progress = print_progress,
                    print_stats = print_stats,
                    print_warning = print_warning,
                    use_gpu = use_gpu,
                    parallel_agent_simulation = parallel_agent_simulation,
                    batches = batches
                )

                agent_histories.append(hist)
                print('')

                # Save simulation history if requested
                if save_histories_path is not None:
                    hist.save(file=f'Simualtions-agent_{i_agent}-environment_{i_environment}', folder=save_histories_path, save_analysis=False)

            training_stats.append(agent_training_stats)
            simulation_histories.append(agent_histories)

        print('--------------------------------------')

    # Generating comparison result table
    all_agent_comparison_dfs = []
    n_environment = len(environments)
    for agent_simulation_histories, agent_training_stats in zip(simulation_histories, training_stats):
        agent_comparison_df = SimulationHistory.compare_all(agent_simulation_histories)
        agent_comparison_df['environment'] = [f'environment_{i}' for i in range(len(environments))]
        agent_comparison_df['training_memory_usage'] = [agent_training_stats['memory_used']] * n_environment if isinstance(agent_training_stats, dict) else [env_training_stats['memory_used'] for env_training_stats in agent_training_stats]
        agent_comparison_df['training_time_taken'] = [agent_training_stats['time_taken']] * n_environment if isinstance(agent_training_stats, dict) else [env_training_stats['time_taken'] for env_training_stats in agent_training_stats]
        agent_comparison_df.set_index('environment')

        all_agent_comparison_dfs.append(agent_comparison_df)

    simulations_comparison_df: pd.DataFrame = pd.concat(all_agent_comparison_dfs, keys=[f'agent_{i}' for i in range(len(agent_classes))], names='agent')

    # Save comparison table if needed
    if save_result_table:
        folder = './' if save_histories_path is None else save_histories_path
        file = 'Simulation_comparison-' + datetime.now().strftime('%Y%m%d_%H%M%S%f')
        simulations_comparison_df.to_csv(folder+file)

    return simulations_comparison_df


def test_agent_memory_scaling(agent: Agent,
                              initialization_values: dict = {},
                              use_gpu: bool = False) -> int:
    '''
    Function to test the limits of up to how many agents can be simulated at once.
    For this, a single iteration of run_test will be run.

    The amounts tested will be powers of 2.

    Parameters
    ----------
    agent : Agent
        The agent to be evaluated
    initialization_values : dict, default = {}
        In the case the agents are to be initialized with custom values,
        the paramaters to be passed on the initialize_state function can be set here.
        If provided, one dict must be provided per agent.
    use_gpu : bool, default = False
        Whether to use the gpu to speedup testing.

    Returns
    -------
    failure_amount : int
        The power of 2 of agents that led to a memory full error.
    '''
    n_exp = 0
    while True:
        try:
            print(f'Attempt with {2**n_exp} agents')
            run_test(
                agent = agent,
                n = 2**n_exp,
                horizon = 1, # Single iteration
                initialization_values = initialization_values,
                print_progress = False,
                print_stats = False,
                print_warning = False,
                use_gpu = use_gpu,
                batches = 1
            )

            # If successfull grow
            n_exp += 1

        except MemoryError as e:
            print(f'Reached full memory with {2**n_exp} agents')
            print(f'Memory full: {e}')
            return 2**n_exp