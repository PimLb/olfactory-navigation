import math
import os
import inspect
import pandas as pd
import sys

from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.patches import Circle
from tqdm.auto import trange, tqdm

from olfactory_navigation import Agent
from olfactory_navigation.environment import Environment
from olfactory_navigation.agent import *

import numpy as np
gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')


__all__ = (
    'run_test',
    'SimulationHistory'
)


class SimulationHistory:
    '''
    Class to record the steps that happened during a simulation with the following information being saved:

    - the positions the agents pass by
    - the actions the agents take
    - the observations the agents receive ('observations')
    - the time in the simulation process


    Parameters
    ----------
    start_points : np.ndarray
        The initial points of the agents in the simulation.
    environment : Environment
        The environment on which the simulation is run (can be different from the one associated with the agent).
    agent : Agent
        The agent used in the simulation.
    time_shift : np.ndarray
        An array of time shifts in the simulation data.
    horizon : int
        The horizon of the simulation. i.e. how many steps can be taken by the agent during the simulation before he is considered lost.
    reward_discount : float, default=0.99
        A discount to be applied to the rewards received by the agent. (eg: reward of 1 received at time n would be: 1 * reward_discount^n)

    Attributes
    ----------
    start_points : np.ndarray
    environment : Environment
    agent : Agent
    time_shift : np.ndarray
    horizon : int
    reward_discount : float
    environment_dimensions : int
        The amount of dimensions of the environment.
    environment_shape : tuple[int]
        The shape of the environment.
    environment_source_position : np.ndarray
        The position of the odor source in the environment.
    environment_source_radius : float
        The radius of the odor source in the environment.
    environment_layer_labels : list[str] or None
        A list of the layer labels if the environment has layers.
    agent_thresholds : np.ndarray
        An array of the olfaction thresholds of the agent.
    n : int
        The amount of simulations.
    start_time : datetime
        The datetime the simulations start.
    actions : list[np.ndarray]
        A list of numpy arrays. At each step of the simulation, an array of shape n by 2 is appended to this list representing the n actions as dy,dx vectors.
    positions : list[np.ndarray]
        A list of numpy arrays. At each step of the simulation, an array of shape n by 2 is appended to this list representing the n positions as y,x vectors.
    observations : list[np.ndarray]
        A list of numpy arrays. At each step of the simulation, an array of shape n is appended to this list representing the n observations received by the agents.
    reached_source : np.ndarray
        A numpy array of booleans saying whether the simulations reached the source or not.
    done_at_step : np.ndarray
        A numpy array containing n elements that records when a given simulation reaches the source (-1 is not reached).
    '''
    def __init__(self,
                 start_points: np.ndarray,
                 environment: Environment,
                 agent: Agent,
                 time_shift: np.ndarray,
                 horizon: int,
                 reward_discount: float = 0.99
                 ) -> None:
        # If only on state is provided, we make it a 1x2 vector
        if len(start_points.shape) == 1:
            start_points = start_points[None,:]

        # Fixed parameters
        self.n = len(start_points)
        self.environment = environment.cpu_version
        self.agent = agent.cpu_version
        self.time_shift = time_shift if gpu_support and cp.get_array_module(time_shift) == np else cp.asnumpy(time_shift)
        self.horizon = horizon
        self.reward_discount = reward_discount
        self.start_time = datetime.now()

        # Simulation Tracking
        self.start_points = start_points if gpu_support and cp.get_array_module(start_points) == np else cp.asnumpy(start_points)
        self.actions = []
        self.positions = []
        self.observations = []
        self.timestamps: list[datetime] = []

        self._running_sims = np.arange(self.n)
        self.reached_source = np.zeros(self.n, dtype=bool)
        self.done_at_step = np.full(self.n, fill_value=-1)

        # Environment and agent attributes
        self.environment_dimensions = self.environment.dimensions
        self.environment_shape = self.environment.shape
        self.environment_source_position = self.environment.source_position
        self.environment_source_radius = self.environment.source_radius
        self.environment_layer_labels = self.environment.layer_labels
        self.agent_thresholds = self.agent.thresholds

        # Other parameters
        self._simulation_dfs = None


    def add_step(self,
                 actions: np.ndarray,
                 next_positions: np.ndarray,
                 observations: np.ndarray,
                 reached_source: np.ndarray,
                 interupt: np.ndarray
                 ) -> None:
        '''
        Function to add a step in the simulation history.

        Parameters
        ----------
        actions : np.ndarray
            The actions that were taken by the agents.
        next_positions : np.ndarray
            The positions that were reached by the agents after having taken actions.
        observations : np.ndarray
            The observations the agents receive after having taken actions.
        reached_source : np.ndarray
            A boolean array of whether each agent has reached the source or not.
        interupt : np.ndarray
            A boolean array of whether each agent has to be terminated even if it hasnt reached the source yet.
        '''
        self._simulation_dfs = None

        # Time tracking
        self.timestamps.append(datetime.now())

        # Check if environment if layered and/or 3D
        layered = 0 if self.environment_layer_labels is None else 1

        # Handle case cupy arrays are provided
        if gpu_support:
            actions = actions if cp.get_array_module(actions) == np else cp.asnumpy(actions)
            next_positions = next_positions if cp.get_array_module(next_positions) == np else cp.asnumpy(next_positions)
            observations = observations if cp.get_array_module(observations) == np else cp.asnumpy(observations)
            reached_source = reached_source if cp.get_array_module(reached_source) == np else cp.asnumpy(reached_source)
            interupt = interupt if cp.get_array_module(interupt) == np else cp.asnumpy(interupt)

        # Actions tracking
        action_all_sims = np.full((self.n, (layered + self.environment_dimensions)), fill_value=-1)
        action_all_sims[self._running_sims] = actions
        self.actions.append(action_all_sims)

        # Next states tracking
        next_position_all_sims = np.full((self.n, self.environment_dimensions), fill_value=-1)
        next_position_all_sims[self._running_sims] = next_positions
        self.positions.append(next_position_all_sims)

        # Observation tracking
        observation_all_sims = np.full((self.n,), fill_value=-1, dtype=float)
        observation_all_sims[self._running_sims] = observations
        self.observations.append(observation_all_sims)

        # Recording at which step the simulation is done if it is done and whether it reached the source
        self.done_at_step[self._running_sims[reached_source | interupt]] = len(self.positions)
        self.reached_source[self._running_sims[reached_source]] = True

        # Updating the list of running sims
        self._running_sims = self._running_sims[~reached_source & ~interupt]


    def compute_distance_to_source(self) -> np.ndarray:
        '''
        Function to compute the optimal distance to the source of each starting point according to the optimal_distance_metric attribute.

        Returns
        -------
        distance : np.ndarray
            The optimal distances to the source point.
        '''
        point = self.start_points

        # Handling the case we have a single point
        is_single_point = (len(point.shape) == 1)
        if is_single_point:
            point = point[None,:]

        # Computing dist
        dist = None
        # if self.optimal_distance_metric == 'manhattan': # TODO Allow for other metrics to be used
        dist = np.sum(np.abs(self.environment_source_position[None,:] - point), axis=-1) - self.environment_source_radius

        if dist is None: # Meaning it was not computed
            raise NotImplementedError('This distance metric has not yet been implemented')

        return float(dist[0]) if is_single_point else dist


    @property
    def runs_analysis_df(self) -> pd.DataFrame:
        '''
        A Pandas DataFrame analyzing the results of the simulations.
        It aggregates the simulations in single rows, recording:

         - <axis>:              The starting positions at the given axis
         - optimal_steps_count: The minimal amount of steps to reach the source
         - converged:           Whether or not the simulation reached the source
         - reached_horizon:     Whether the failed simulation reached to horizon
         - steps_taken:         The amount of steps the agent took to reach the source, (horizon if the simulation did not reach the source)
         - discounted_rewards:  The discounted reward received by the agent over the course of the simulation
         - extra_steps:         The amount of extra steps compared to the optimal trajectory
         - t_min_over_t:        Normalized version of the extra steps measure, where it tends to 1 the least amount of time the agent took to reach the source compared to an optimal trajectory.
        '''
        # Get axes labels
        axes_labels = None
        if self.environment_dimensions <= 3:
            axes_labels = ['z', 'y', 'x'][-self.environment_dimensions:]
        else:
            axes_labels = [f'x{i}' for i in range(self.environment_dimensions)]

        # Dataframe creation
        df = pd.DataFrame(self.start_points, columns=axes_labels)
        df['optimal_steps_count'] = self.compute_distance_to_source()
        df['converged'] = self.reached_source
        df['reached_horizon'] = np.all(self.positions[-1] != -1, axis=1) & ~self.reached_source & (len(self.positions) == self.horizon)
        df['steps_taken'] = np.where(self.done_at_step >= 0, self.done_at_step, len(self.positions))
        df['discounted_rewards'] = self.reward_discount ** df['steps_taken']
        df['extra_steps'] = df['steps_taken'] - df['optimal_steps_count']
        df['t_min_over_t'] = df['optimal_steps_count'] / df['steps_taken']

        # Reindex
        runs_list = [f'run_{i}' for i in range(self.n)]
        df.index = runs_list

        return df


    @property
    def general_analysis_df(self) -> pd.DataFrame:
        '''
        A Pandas DataFrame analyzing the results of the simulations.
        Summarizing the performance of all the simulations with the following metrics:

         - converged:           Whether or not the simulation reached the source
         - reached_horizon:     Whether the failed simulation reached to horizon
         - steps_taken:         The amount of steps the agent took to reach the source, (horizon if the simulation did not reach the source)
         - discounted_rewards:  The discounted reward received by the agent over the course of the simulation
         - extra_steps:         The amount of extra steps compared to the optimal trajectory
         - t_min_over_t:        Normalized version of the extra steps measure, where it tends to 1 the least amount of time the agent took to reach the source compared to an optimal trajectory.

        For the measures (converged, steps_taken, discounted_rewards, extra_steps, t_min_over_t), the average and standard deviations are computed in rows at the top.
        '''
        df = self.runs_analysis_df

        # Analysis aggregations
        columns_to_analyze = ['converged', 'reached_horizon', 'steps_taken', 'discounted_rewards', 'extra_steps', 't_min_over_t']
        row_names = [['mean', 'standard_deviation', 'success_mean', 'success_standard_deviation']]
        general_analysis_data = [
            df[columns_to_analyze].mean(),
            df[columns_to_analyze].std(),
            df.loc[df['converged'], columns_to_analyze].mean(),
            df.loc[df['converged'], columns_to_analyze].std()
        ]

        return pd.DataFrame(data=general_analysis_data, index=row_names, columns=columns_to_analyze)


    @property
    def done_count(self) -> int:
        '''
        Returns how many simulations are terminated (whether they reached the source or not).
        '''
        return self.n - len(self._running_sims)


    @property
    def successful_simulation(self) -> np.ndarray:
        return self.reached_source


    @property
    def success_count(self) -> int:
        '''
        Returns how many simulations reached the source.
        '''
        return int(np.sum(self.successful_simulation))


    @property
    def simulations_at_horizon(self) -> np.ndarray:
        '''
        Returns a boolean array of which simulations reached the horizon.
        '''
        last_position_exists = np.all(self.positions[-1] != -1, axis=1)
        simulation_reached_horizon = (len(self.positions) == self.horizon)
        return last_position_exists & ~self.reached_source & simulation_reached_horizon


    @property
    def summary(self) -> str:
        '''
        A string summarizing the performances of all the simulations.
        The metrics used are averages of:

         - Step count
         - Extra steps
         - Discounted rewards
         - Tmin / T

        Along with the respective the standard deviations and equally for only for the successful simulations.
        '''
        success_sim_count = self.success_count
        failed_count = self.n - success_sim_count
        reached_horizon_count = int(np.sum(self.simulations_at_horizon))
        summary_str = f'Simulations reached goal: {success_sim_count}/{self.n} ({failed_count} failures (reached horizon: {reached_horizon_count})) ({(success_sim_count*100)/self.n:.2f}% success)'

        if success_sim_count == 0:
            return summary_str

        # Metrics
        df = self.general_analysis_df

        summary_str += f"\n - {'Average step count:':<35} {df.loc['mean','steps_taken'].item():.3f} +- {df.loc['standard_deviation','steps_taken'].item():.2f} "
        summary_str += f"(Successful only: {df.loc['success_mean','steps_taken'].item():.3f} +- {df.loc['success_standard_deviation','steps_taken'].item():.2f})"

        summary_str += f"\n - {'Extra steps:':<35} {df.loc['mean','extra_steps'].item():.3f} +- {df.loc['standard_deviation','extra_steps'].item():.2f} "
        summary_str += f"(Successful only: {df.loc['success_mean','extra_steps'].item():.3f} +- {df.loc['success_standard_deviation','extra_steps'].item():.2f})"

        summary_str += f"\n - {'Average discounted rewards (ADR):':<35} {df.loc['mean','discounted_rewards'].item():.3f} +- {df.loc['standard_deviation','discounted_rewards'].item():.2f} "
        summary_str += f"(Successful only: {df.loc['success_mean','discounted_rewards'].item():.3f} +- {df.loc['success_standard_deviation','discounted_rewards'].item():.2f})"

        summary_str += f"\n - {'Tmin/T:':<35} {df.loc['mean','t_min_over_t'].item():.3f} +- {df.loc['standard_deviation','t_min_over_t'].item():.2f} "
        summary_str += f"(Successful only: {df.loc['success_mean','t_min_over_t'].item():.3f} +- {df.loc['success_standard_deviation','t_min_over_t'].item():.2f})"

        return summary_str


    @property
    def simulation_dfs(self) -> list[pd.DataFrame]:
        '''
        A list of the pandas DataFrame where each dataframe is a single simulation history.
        Each row is a different time instant of simulation process with each column being:

         - time (of the simulation data)
         - [position] (z,) y, x  OR  x0, x1, ... xn
         - (layer)
         - [movement] (dz,) dy, dx  OR  dx0, dx1, ... dxn
         - o (pure, not thresholded)
         - reached_source (boolean)
        '''
        if self._simulation_dfs is None:
            self._simulation_dfs = []

            # Converting state, actions and observation to numpy arrays
            states_array = np.array(self.positions)
            action_array = np.array(self.actions)
            observation_array = np.array(self.observations)

            # Get axes labels
            axes_labels = None
            if self.environment_dimensions <= 3:
                axes_labels = ['z', 'y', 'x'][-self.environment_dimensions:]
            else:
                axes_labels = [f'x{i}' for i in range(self.environment_dimensions)]

            # Loop through the n simulations
            for i in range(self.n):
                length = self.done_at_step[i] if self.done_at_step[i] >= 0 else len(states_array)

                # Creation of the dataframe
                df = {}
                df['time'] = np.arange(length+1) + self.time_shift[i]

                # - Position variables
                for axis_i, axis in enumerate(axes_labels):
                    df[axis] = np.hstack([self.start_points[i, axis_i], states_array[:length, i, axis_i]])

                # - Action variables
                if self.environment_layer_labels is not None:
                    df['layer'] = np.hstack([[None], action_array[:length, i, 0]])

                for axis_i, axis in enumerate(axes_labels):
                    axis_i += (0 if self.environment_layer_labels is None else 1)
                    df['d' + axis]   = np.hstack([[None], action_array[:length, i, axis_i]])

                # - Other variables
                df['o'] = np.hstack([[None], observation_array[:length, i]])
                df['reached_source'] = np.hstack([[None], np.where(np.arange(1,length+1) == self.done_at_step[i], (1 if self.reached_source[i] else 0), 0)])

                # Append
                self._simulation_dfs.append(pd.DataFrame(df))

        return self._simulation_dfs


    def __add__(self, other_hist: 'SimulationHistory'):
        # Asserting the SimulationHistory objects are compatible
        assert self.horizon == other_hist.horizon, "The 'horizon' parameters must match between the two SimulationHistory objects..."
        assert self.reward_discount == other_hist.reward_discount, "The 'reward_discount' parameters must match between the two SimulationHistory objects..."
        assert self.environment_dimensions == other_hist.environment_dimensions, "The 'environment_dimensions' parameters must match between the two SimulationHistory objects..."
        assert self.environment_shape == other_hist.environment_shape, "The 'environment_shape' parameters must match between the two SimulationHistory objects..."
        assert self.environment_layer_labels == other_hist.environment_layer_labels, "The 'environment_layer_labels' parameters must match between the two SimulationHistory objects..."
        assert all(self.environment_source_position == other_hist.environment_source_position), "The 'environment_source_position' parameters must match between the two SimulationHistory objects..."
        assert self.environment_source_radius == other_hist.environment_source_radius, "The 'environment_source_radius' parameters must match between the two SimulationHistory objects..."
        assert all(self.agent_thresholds == other_hist.agent_thresholds), "The 'agent_thresholds' parameters must match between the two SimulationHistory objects..."

        # Combining arrays
        combined_start_points = np.vstack([self.start_points,
                                           other_hist.start_points])
        combined_time_shifts = np.hstack([self.time_shift,
                                          other_hist.time_shift])
        combined_reached_source = np.hstack([self.reached_source,
                                             other_hist.reached_source])
        combined_done_at_step = np.hstack([self.done_at_step,
                                           other_hist.done_at_step])

        combined_actions = []
        combined_positions = []
        combined_observations = []
        for step_i in range(max([len(self.actions), len(other_hist.actions)])):
            self_in_range = (step_i < len(self.actions))
            other_in_range = (step_i < len(other_hist.actions))
            combined_actions.append(np.vstack([self.actions[step_i] if self_in_range else np.full_like(self.actions[0], fill_value=-1),
                                               other_hist.actions[step_i] if other_in_range else np.full_like(other_hist.actions[0], fill_value=-1)]))
            combined_positions.append(np.vstack([self.positions[step_i] if self_in_range else np.full_like(self.positions[0], fill_value=-1),
                                                 other_hist.positions[step_i] if other_in_range else np.full_like(other_hist.positions[0], fill_value=-1)]))
            combined_observations.append(np.hstack([self.observations[step_i] if self_in_range else np.full_like(self.observations[0], fill_value=-1),
                                                    other_hist.observations[step_i] if other_in_range else np.full_like(other_hist.observations[0], fill_value=-1)]))

        # Creating the combined simulation history object
        combined_hist = SimulationHistory(start_points = combined_start_points,
                                          environment = self.environment,
                                          agent = self.agent,
                                          time_shift = combined_time_shifts,
                                          horizon = self.horizon,
                                          reward_discount = self.reward_discount)

        combined_hist.start_time = self.start_time
        combined_hist.actions = combined_actions
        combined_hist.positions = combined_positions
        combined_hist.observations = combined_observations
        combined_hist.reached_source = combined_reached_source
        combined_hist.done_at_step = combined_done_at_step

        return combined_hist


    def save(self,
             file: str | None = None,
             folder: str | None = None,
             save_analysis: bool = True,
             save_components: bool = False
             ) -> None:
        '''
        Function to save the simulation history to a csv file in a given folder.
        Additionally, an analysis of the runs can be saved if the save_analysis is enabled.
        The environment and agent used can be saved in the saved folder by enabling the 'save_component' parameter.

        Parameters
        ----------
        file : str, optional
            The name of the file the simulation histories will be saved to.
            If it is not provided, it will be by default "Simulations-<env_name>-n_<sim_count>-<sim_start_timestamp>-horizon_<max_sim_length>.csv"
        folder : str, optional
            Folder to save the simulation histories to.
            If the folder name is not provided the current folder will be used.
        save_analysis : bool, default=True
            Whether to save an additional csv file with an analysis of the runs of the simulation.
            It will contain the amount of steps taken, the amount of extra steps compared to optimality, the discounted rewards and the ratio between optimal trajectory and the steps taken.
            The means and standard deviations of all the runs are also computed.
            The file will have the same name as the simulation history file with an additional '-analysis' tag at the end.
        save_components : bool, default=False
            Whether or not to save the environment and agent along with the simulation histories in the given folder.
        '''
        assert (self.environment is not None) and (self.agent is not None), "Function not available, the agent and/or the environment is not set."

        # Handle file name
        if file is None:
            env_name = f's_' + '_'.join([str(axis_shape) for axis_shape in self.environment_shape])
            file = f'Simulations-{env_name}-n_{self.n}-{self.start_time.strftime("%Y%m%d_%H%M%S")}-horizon_{len(self.positions)}.csv'

        if not file.endswith('.csv'):
            file += '.csv'

        # Handle folder
        if folder is None:
            folder = './'

        if '/' not in folder:
            folder = './' + folder

        if not os.path.exists(folder):
            os.mkdir(folder)

        if not folder.endswith('/'):
            folder += '/'

        # Save components if requested
        if save_components:
            if (self.environment.saved_at is None) or (folder not in self.environment.saved_at):
                self.environment.save(folder=folder)

            if (self.agent.saved_at is None) or (folder not in self.agent.saved_at):
                self.agent.save(folder=folder)

        # Create csv file
        combined_df = pd.concat(self.simulation_dfs)

        # Adding other useful info
        padding = [None] * len(combined_df)
        combined_df['timestamps'] = [self.start_time.strftime('%Y%m%d_%H%M%S%f')] + [ts.strftime('%H%M%S%f') for ts in self.timestamps] + padding[:-(len(self.timestamps)+1)]
        combined_df['horizon'] = [self.horizon] + padding[:-1]
        combined_df['reward_discount'] = [self.reward_discount] + padding[:-1]

        environment_info = [
            self.environment.name,
            self.environment.saved_at,
            str(self.environment_dimensions), # int
            '_'.join(str(axis_size) for axis_size in self.environment_shape),
            '_'.join(str(axis_position) for axis_position in self.environment_source_position),
            str(self.environment_source_radius), # float
            '' if (self.environment_layer_labels is None) else '&'.join(self.environment_layer_labels) # Using '&' as splitter as '_' could be used in the labels themselves
        ]
        combined_df['environment'] = (environment_info + padding[:-len(environment_info)])

        # Converting the thresholds array to a string to be saved
        thresholds_string = ''
        if len(self.agent_thresholds.shape) == 2:
            thresholds_string = '&'.join(['_'.join([str(item) for item in row]) for row_i, row in enumerate(self.agent_thresholds[:,1:-1])]) # Using '&' as layer splitter as '-' can be used for negative thresholds
        else:
            thresholds_string = '_'.join([str(item) for item in self.agent_thresholds])

        agent_info = [
            self.agent.name,
            self.agent.class_name,
            self.agent.saved_at,
            thresholds_string
        ]
        combined_df['agent'] = (agent_info + padding[:-len(agent_info)])

        # Saving csv
        combined_df.to_csv(folder + file, index=False)

        print(f'Simulations saved to: {folder + file}')

        if save_analysis:
            runs_analysis_file_name = file.replace('.csv', '-runs_analysis.csv')
            self.runs_analysis_df.to_csv(folder + runs_analysis_file_name)
            print(f"Simulation's runs analysis saved to: {folder + runs_analysis_file_name}")

            general_analysis_file_name = file.replace('.csv', '-general_analysis.csv')
            self.general_analysis_df.to_csv(folder + general_analysis_file_name)
            print(f"Simulation's general analysis saved to: {folder + general_analysis_file_name}")


    @classmethod
    def load_from_file(cls,
                       file: str,
                       environment: bool | Environment = False,
                       agent: bool | Agent = False
                       ) -> 'SimulationHistory':
        return cls.load(file, environment, agent)


    @classmethod
    def load(cls,
             file: str,
             environment: bool | Environment = False,
             agent: bool | Agent = False
             ) -> 'SimulationHistory':
        '''
        Function to load the simulation history from a file.
        This can be useful to use the plot functions on the simulations saved in such file.

        The environment and agent can provided as a backup in the case they cannot be loaded from the file.

        Parameters
        ----------
        file : str
            A file (with the path) of the simulation histories csv. (the analysis file cannot be used for this)
        environment : bool or Environment, default=False
            If set to True, it will try to load the environment that was used for the simulation (if the save path is available).
            Or, an environment instance to be linked with the simulation history object.
        agent : bool or Agent, default=False
            If set to True, it will try to load the agent that was used for the simulation (if the save path is available).
            An agent instance to be linked with the simulation history object.

        Returns
        -------
        hist : SimulationHistory
            The loaded instance of a simulation history object.
        '''
        # Retrieving columns
        with open(file, 'r') as f:
            header = f.readline()
        columns = header.replace('\n','').split(',')

        # Setting the datatypes of columns
        column_dtypes = {col: float for col in columns}
        column_dtypes['time'] = int
        if 'layer' in columns:
            column_dtypes['layer'] = int
        column_dtypes['timestamps'] = str
        column_dtypes['environment'] = str
        column_dtypes['agent'] = str

        # Retrieving the combined dataframe
        combined_df = pd.read_csv(file, dtype=column_dtypes)

        # Retrieving horizon and reward discount
        horizon = int(combined_df['horizon'][0])
        reward_discount = combined_df['reward_discount'][0]

        # Retrieving environment
        if (not isinstance(environment, Environment)) and (environment == True):
            environment_name = combined_df['environment'][0]
            environment_path = combined_df['environment'][1]

            environment_path_check = (environment_path is not None) and (not np.isnan(environment_path))
            assert environment_path_check, "Environment was not saved at the time of the saving of the simulation history. Input an environment to the environment parameter or toggle the parameter to False."

            try:
                environment = Environment.load(environment_path)
            except:
                print(f'Failed to retrieve "{environment_name}" environment from memory')

        # Retrieving agent
        if (not isinstance(agent, Agent)) and (agent == True):
            agent_name = combined_df['environment'][0]
            agent_class = combined_df['environment'][1]
            agent_path = combined_df['environment'][2]

            agent_path_check = (agent_path is not None) and (not np.isnan(agent_path))
            assert agent_path_check, "Agent was not saved at the time of the saving of the simulation history. Input an agent to the agent parameter or toggle the parameter to False."

            try:
                class_instance = None
                for (class_name, class_obj) in inspect.getmembers(sys.modules[__name__], inspect.isclass):
                    if class_name == agent_class:
                        class_instance = class_obj
                        break
                agent = class_instance.load(combined_df['agent'][2])
            except:
                print(f'Failed to retrieve "{agent_name}" agent from memory')

        # Other attributes
        environment_dimensions = int(combined_df['environment'][2])
        environment_shape = tuple([int(axis_shape) for axis_shape in combined_df['environment'][3].split('_')])
        environment_source_position = np.array([float(pos_axis) for pos_axis in combined_df['environment'][4].split('_')])
        environment_source_radius = float(combined_df['environment'][5])
        layer_entery = combined_df['environment'][6]
        environment_layer_labels = (None if ((not isinstance(layer_entery, str)) or (len(layer_entery) == 0)) else layer_entery.split('&'))

        # Processing the threshold string
        thresholds_string = str(combined_df['agent'][3])
        if '&' in thresholds_string:
            rows_thresholds_string = thresholds_string.split('&')
            layer_thresholds = []
            for row in rows_thresholds_string:
                layer_thresholds.append(np.array(row.split('_')).astype(float))
            agent_thresholds = np.array(layer_thresholds)

        else:
            agent_thresholds = np.array(np.array(thresholds_string.split('_')).astype(float))

        # Columns to retrieve
        columns = [col for col in columns if col not in ['reward_discount', 'environment', 'agent']]

        # Checking how many dimensions there are
        has_layers = (((len(columns) - 5) % 2) == 1)
        dimensions = int((len(columns) - 5) / 2)

        # Recreation of list of simulations
        sim_start_rows = np.argwhere(combined_df[['reached_source']].isnull())[1:,0].tolist()

        simulation_arrays = np.split(combined_df[columns].to_numpy(), sim_start_rows)
        simulation_dfs = [pd.DataFrame(sim_array, columns=columns) for sim_array in simulation_arrays]

        # Making a combined numpy array with all the simulations
        sizes = np.array([len(sim_array) for sim_array in simulation_arrays])
        max_length = sizes.max()
        paddings = max_length - sizes

        padded_simulation_arrays = [np.pad(sim_arr, ((0,pad),(0,0)), constant_values=-1) for sim_arr, pad in zip(simulation_arrays, paddings)]
        all_simulation_arrays = np.array(padded_simulation_arrays).transpose((1,0,2))

        # Timeshift
        time_shift = all_simulation_arrays[0,:,0].astype(int)

        # Gathering start states
        start_points = all_simulation_arrays[0,:,1:(1+dimensions)].astype(int)

        # Recreating action, state and observations
        positions = all_simulation_arrays[1:, :, 1:(1+dimensions)]
        actions = all_simulation_arrays[1:, :, (1+dimensions):((1+dimensions) + (1 if has_layers else 0) + dimensions)]
        observations = all_simulation_arrays[1:, :, ((1+dimensions) + (1 if has_layers else 0) + dimensions)]
        reached_source = np.array([(df['reached_source'][len(df)-1] == 1) for df in simulation_dfs])
        done_at_step = np.where((sizes-1 < horizon), sizes-1, -1)

        # Building SimulationHistory instance
        hist = cls.__new__(cls)

        hist.n = len(start_points)
        hist.environment = environment.cpu_version if isinstance(environment, Environment) else None
        hist.agent = agent.cpu_version if isinstance(agent, Agent) else None
        hist.time_shift = time_shift
        hist.horizon = horizon
        hist.reward_discount = reward_discount
        hist.start_time = datetime.strptime(combined_df['timestamps'][0], '%Y%m%d_%H%M%S%f')

        hist.start_points = start_points
        hist._running_sims = None

        hist.positions = [*positions]
        hist.actions = [*actions]
        hist.observations = [*observations]
        hist.reached_source = reached_source
        hist.done_at_step = done_at_step
        hist.timestamps = [datetime.strptime(ts, '%H%M%S%f') for ts in combined_df['timestamps'][1:max_length]]

        # Other attributes
        hist.environment_dimensions = environment_dimensions
        hist.environment_shape = environment_shape
        hist.environment_source_position = environment_source_position
        hist.environment_source_radius = environment_source_radius
        hist.environment_layer_labels = environment_layer_labels
        hist.agent_thresholds = agent_thresholds

        # Saving simulation dfs back
        hist._simulation_dfs = simulation_dfs

        return hist


    def plot(self,
             sim_id: int = 0,
             ax: plt.Axes | None = None
             ) -> None:
        '''
        Function to plot a the trajectory of a given simulation.
        An ax can be use to plot it on.

        Parameters
        ----------
        sim_id : int, default=0
            The id of the simulation to plot.
        ax : plt.Axes, optional
            The ax on which to plot the path. (If not provided, a new axis will be created)
        '''
        # TODO: Setup 3D plotting
        assert self.environment_dimensions == 2, "Plotting function only available for 2D environments for now..."

        # Generate ax is not provided
        if ax is None:
            _, ax = plt.subplots(figsize=(18,3))

        # Retrieving sim
        sim = self.simulation_dfs[sim_id]

        # Plot setup
        env_shape = self.environment_shape
        ax.imshow(np.zeros(self.environment_shape), cmap='Greys', zorder=-100)
        ax.set_xlim(0, env_shape[1])
        ax.set_ylim(env_shape[0], 0)

        # Start
        start_coord = sim[['x', 'y']].to_numpy()[0]
        ax.scatter(start_coord[0], start_coord[1], c='green', label='Start')

        # Source circle
        goal_circle = Circle(self.environment_source_position[::-1], self.environment_source_radius, color='r', fill=False, label='Source')
        ax.add_patch(goal_circle)

        # Until step
        seq = sim[['x','y']].to_numpy()

        # Path
        ax.plot(seq[:,0], seq[:,1], zorder=-1, c='black', label='Path')

        # Layer observations
        if self.environment_layer_labels is not None:
            obs_layer = sim[['layer']][1:].to_numpy()
            layer_colors = np.array(list(colors.TABLEAU_COLORS.values()))

            for layer_i, layer_label in enumerate(self.environment_layer_labels[1:]):
                layer_i += 1
                layer_mask = (obs_layer == layer_i)[:,0] # Reshaping to a single vector and not an n by 1 array
                ax.scatter(seq[1:][layer_mask,0], seq[1:][layer_mask,1], # X, Y
                           marker='x',
                           color=layer_colors[(layer_i-1) % len(layer_colors)], # Looping over the colors in case there are more layers than colors
                           zorder=2,
                           label=layer_label)

        # Process odor cues
        odor_cues = sim['o'][1:].to_numpy()
        observation_ids = None
        if (self.environment_layer_labels is not None) and len(self.agent_thresholds.shape) == 2:
            layer_ids = sim[['layer']][1:].to_numpy()
            action_layer_thresholds = self.agent_thresholds[layer_ids]
            observation_ids = np.argwhere((odor_cues[:,None] >= action_layer_thresholds[:,:-1]) & (odor_cues[:,None] < action_layer_thresholds[:,1:]))[:,1]
        else:
            # Setting observation ids
            observation_ids = np.argwhere((odor_cues[:,None] >= self.agent_thresholds[:-1][None,:]) & (odor_cues[:,None] < self.agent_thresholds[1:][None,:]))[:,1]

        # Check whether the odor detection is binary or by level
        odor_bins = self.agent_thresholds.shape[-1] - 1
        if odor_bins > 2:
            odor_levels = np.arange(odor_bins - 1) + 1
            for level in odor_levels:
                cues_at_level = (observation_ids == level)
                ax.scatter(seq[1:][cues_at_level,0], seq[1:][cues_at_level,1],
                           zorder=1,
                           alpha=(level / odor_bins),
                           label=f'Sensed level {level}')
        else:
            something_sensed = (observation_ids == 1)
            ax.scatter(seq[1:][something_sensed,0], seq[1:][something_sensed,1],
                       zorder=1,
                       label='Something observed')

        # Generate legend
        ax.legend()


    def plot_runtimes(self,
                      ax: plt.Axes | None = None
                      ) -> None:
        '''
        Function to plot the runtimes over the iterations.

        Parameters
        ----------
        ax : plt.Axes, optional
            The ax on which to plot the path. (If not provided, a new axis will be created)
        '''
        # Generate ax is not provided
        if ax is None:
            _, ax = plt.subplots(figsize=(18,3))

        # Computing differences
        timestamp_differences_ms = np.diff(np.array([int(ts.strftime('%H%M%S%f')) for ts in self.timestamps])) / 1000

        # Actual plot
        ax.plot(timestamp_differences_ms)

        # Axes
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Runtime (ms)')


    def plot_successes(self,
                       ax: plt.Axes | None = None
                       ) -> None:
        '''
        Function to plot a 2D map of whether a given starting point was successfull or not (and whether it died early).

        Parameters
        ----------
        ax : plt.Axes, optional
            The ax on which to plot the path. (If not provided, a new axis will be created)
        '''
        assert self.environment_dimensions == 2, "Only implemented for 2D environments..."

        # Generate ax is not provided
        if ax is None:
            _, ax = plt.subplots(figsize=(18,3))

        # Setting up an empty grid of the starting points
        start_points_grid = np.zeros(self.environment_shape)

        # Compute the successful, failed and the ones that reached the horizon
        success_points = self.start_points[self.successful_simulation]
        failed_points = self.start_points[~self.successful_simulation]
        failed_not_at_horizon_points = self.start_points[~self.successful_simulation & ~self.simulations_at_horizon]

        start_points_grid[failed_points[:,0], failed_points[:,1]] = -1
        start_points_grid[success_points[:,0], success_points[:,1]] = 1

        ax.imshow(start_points_grid, cmap='RdBu')

        # The crosses where the points did not reach the horizon
        ax.scatter(failed_not_at_horizon_points[:,1], failed_not_at_horizon_points[:,0], marker='x', color='black', s=10, label='Died early')
        ax.legend()


def run_test(agent: Agent,
             n: int | None = None,
             start_points: np.ndarray | None = None,
             environment: Environment | None = None,
             time_shift: int | np.ndarray = 0,
             time_loop: bool = True,
             horizon: int = 1000,
             initialization_values: dict = {},
             reward_discount: float = 0.99,
             print_progress: bool = True,
             print_stats: bool = True,
             print_warning: bool = True,
             use_gpu: bool = False,
             batches: int = -1
             ) -> SimulationHistory:
    '''
    Function to run n simulations for a given agent in its environment (or a given modified environment).
    The simulations start either from random start points or provided trough the start_points parameter.
    The simulation can have shifted initial times (in the olfactory simulation).

    The simulation will run for at most 'horizon' steps, after which the simulations will be considered failed.

    Some statistics can be printed at end of the simulation with the 'print_stats' parameter.
    It will print some performance statisitcs about the simulations such as the average discounter reward.
    The reward discount can be set by the 'reward_discount' parameter.

    To speedup the simulations, it can be run on the gpu by toggling the 'use_gpu' parameter.
    This will have the consequence to send the various arrays to the gpu memory.
    This will only work if the agent has the support for to work with cupy arrays.

    This method returns a SimulationHistory object that saves all the positions the agent went through,
    the actions the agent took, and the observation the agent received.
    It also provides the possibility the save the results to a csv file and plot the various trajectories.

    Parameters
    ----------
    agent : Agent
        The agent to be tested
    n : int, optional
        How many simulation to run in parallel.
        n is optional but it needs to match with what is provided in start_points.
    start_points : np.ndarray, optional
        The starting points of the simulation in 2d space.
        If not provided, n random points will be generated based on the start probabilities of the environment.
        Else, the amount of start_points need to match to n, if it is provided.
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
    initialization_values : dict, default={}
        In the case the agent is to be initialized with custom values,
        the paramaters to be passed on the initialize_state function can be set here.
    reward_discount : float, default=0.99
        How much a given reward is discounted based on how long it took to get it.
        It is purely used to compute the Average Discount Reward (ADR) after the simulation.
    print_progress : bool, default=True
        Whether to show a progress bar of what step the simulations are at.
    print_stats : bool, default=True
        Whether to print the stats at the end of the run.
    print_warning : bool, default=True
        Whether to print warnings when they occur or not.
    use_gpu : bool, default=False
        Whether to run the simulations on the GPU or not.
    batches : int, default=-1
        In how many batches the simulations should be run.
        This is useful in the case there are too many simulations and the memory can fill up.
        The value of batches=-1 will make it that different batches amount are tried in increasing order if a MemoryError is encountered.

    Returns
    -------
    hist : SimulationHistory
        A SimulationHistory object that tracked all the positions, actions and observations.
    '''
    # Gathering n
    if n is None:
        if (start_points is None) or (len(start_points.shape) == 1):
            n = 1
        else:
            n = len(start_points)

    # Handle the case an specific environment is given
    if (environment is not None) and print_warning:
        if environment.shape != agent.environment.shape:
            print("[Warning] The provided environment's shape doesn't match the environment has been trained on...")
        print('Using the provided environment, not the agent environment.')
    else:
        environment = agent.environment

    # Timeshift
    if isinstance(time_shift, int):
        time_shift = np.ones(n) * time_shift
    else:
        time_shift = np.array(time_shift)
        assert time_shift.shape == (n,), f"time_shift array has a wrong shape (Given: {time_shift.shape}, expected ({n},))"
    time_shift = time_shift.astype(int)

    # Auto batches selector where the amount of batches increases if a memory error is detected
    if batches < 0:
        all_try_batches = (2**np.arange(np.log2(11000), dtype=int))
        for try_batches in all_try_batches:
            try:
                hist = run_test(agent = agent,
                                n = n,
                                start_points = start_points,
                                environment = environment,
                                time_shift = time_shift,
                                time_loop = time_loop,
                                horizon = horizon,
                                initialization_values = initialization_values,
                                reward_discount = reward_discount,
                                print_progress = print_progress,
                                print_stats = print_stats,
                                print_warning = False, # If there was any, it would have been printed already
                                use_gpu = use_gpu,
                                batches = try_batches)
                return hist
            except MemoryError as e:
                print(f'Memory full: {e}')
                print('Increasing the amount of batches...')

    # If more than one batch is selected, split the starting point arrays by the amounts of simulations in each batch
    elif batches > 1:
        # Computing the amount of simulations to be in each batch
        n_batches = np.array([n / batches] * batches).astype(int)
        n_batches[:(n%batches)] += 1
        n_start = 0

        # Full SimulationHistory object
        combined_hist = None

        # Time tracking
        all_sim_start_ts = datetime.now()

        # Batches loop
        batch_iterator = tqdm(n_batches, desc='Batches') if print_progress else n_batches
        for b_n in batch_iterator:
            b_hist = run_test(agent = agent,
                              n = b_n,
                              start_points = start_points[n_start:n_start+b_n],
                              environment = environment,
                              time_shift = time_shift[n_start:n_start+b_n],
                              time_loop = time_loop,
                              horizon = horizon,
                              initialization_values = initialization_values,
                              reward_discount = reward_discount,
                              print_progress = print_progress,
                              print_stats = False, # Forced false to not print too many things
                              print_warning = False, # If there was any, it would have been printed already
                              use_gpu = use_gpu,
                              batches = 1)
            n_start += b_n

            # Combining SimulationHistory objects
            if combined_hist is None:
                combined_hist = b_hist
            else:
                combined_hist += b_hist

        # Print stats of the complete history is asked
        if print_stats:
            all_sim_end_ts = datetime.now()
            print(f'Simulations done in {(all_sim_end_ts - all_sim_start_ts).total_seconds():.3f}s:')
            print(combined_hist.summary)

        return combined_hist

    # Move things to GPU if needed
    xp = np
    if use_gpu:
        assert gpu_support, f"GPU support is not enabled, the use_gpu option is not available."
        xp = cp

        # Move instances to GPU
        agent = agent.gpu_version
        environment = environment.gpu_version
        time_shift = cp.array(time_shift)

        if start_points is not None:
            start_points = cp.array(start_points)

    # Set start positions
    agent_position = None
    if start_points is not None:
        assert start_points.shape == (n, environment.dimensions), f'The provided start_points are of the wrong shape (expected {environment.dimensions}; received {start_points.shape[1]})'
        agent_position = start_points
    else:
        # Generating random starts
        agent_position = environment.random_start_points(n)

    # Initialize agent's state
    agent.initialize_state(n, **initialization_values)

    # Create simulation history tracker
    hist = SimulationHistory(
        start_points=agent_position,
        environment=environment,
        agent=agent,
        time_shift=time_shift,
        horizon=horizon,
        reward_discount=reward_discount
    )

    # Track begin of simulation ts
    sim_start_ts = datetime.now()

    # Simulation loop
    iterator = trange(horizon) if print_progress else range(horizon)
    for i in iterator:
        # Letting agent choose the action to take based on it's curent state
        action = agent.choose_action()

        # Updating the agent's actual position (hidden to him)
        agent_position = environment.move(pos=agent_position,
                                          movement=(action if not environment.has_layers else action[:,1:])) # Getting only the physical component of the action vector if environment has layers.

        # Get an observation based on the new position of the agent
        observation = environment.get_observation(pos=agent_position,
                                                  time=(time_shift + i),
                                                  layer=(0 if not environment.has_layers else action[:,0])) # Getting the layer information column of the action matrix.

        # Check if the source is reached
        source_reached = environment.source_reached(agent_position)

        # Add the position to the observation if the agent is space aware
        if agent.space_aware:
            observation = xp.hstack((observation[:,None], agent_position))

        # Return the observation to the agent
        update_succeeded = agent.update_state(action=action,
                                              observation=observation,
                                              source_reached=source_reached)
        if update_succeeded is None:
            update_succeeded = xp.ones(len(source_reached), dtype=bool)

        # Handling the case where simulations have reached the end
        sims_at_end = ((time_shift + i + 1) >= (math.inf if time_loop else environment.timesteps))

        # Agents to terminate
        to_terminate = source_reached | sims_at_end | ~update_succeeded

        # Send the values to the tracker
        hist.add_step(
            actions=action,
            next_positions=agent_position,
            observations=observation[:,0] if agent.space_aware else observation,
            reached_source=source_reached,
            interupt=to_terminate
        )

        # Interupt agents that reached the end
        agent_position = agent_position[~to_terminate]
        time_shift = time_shift[~to_terminate]
        agent.kill(simulations_to_kill=to_terminate)

        # Early stopping if all agents done
        if len(agent_position) == 0:
            break

        # Update progress bar
        if print_progress:
            done_count = hist.done_count
            success_count = hist.success_count
            success_percentage = (success_count/done_count)*100 if done_count > 0 else 100
            dead_percentage = ((done_count-success_count)/done_count)*100 if done_count > 0 else 0
            iterator.set_postfix({
                'done ': f' {done_count}/{n} ({(done_count/n)*100:.1f}%)',
                'success ': f' {success_count}/{done_count} ({success_percentage:.1f}%)',
                'dead ': f' {done_count-success_count}/{done_count} ({dead_percentage:.1f}%)'
            })

    # If requested print the simulation start
    if print_stats:
        sim_end_ts = datetime.now()
        print(f'Simulations done in {(sim_end_ts - sim_start_ts).total_seconds():.3f}s:')
        print(hist.summary)

    return hist
