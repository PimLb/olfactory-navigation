import math
import os
import inspect
import pandas as pd
import sys

from datetime import datetime
from matplotlib import pyplot as plt 
from tqdm.auto import trange

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
    Class to represent a list of the steps that happened during a simulation with:

        - the positions the agents pass by
        - the actions the agents take
        - the observations the agents receive ('observations')


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
    n : int
        The amount of simulations.
    start_time : datetime
        The datetime the simulations start.
    actions : list[np.ndarray]
        A list of numpy arrays. At each step of the simulation, an array of shape n by 2 is appended to this list representing the n actions as dy,dx vectors.
    positions : list[np.ndarray]
        A list of numpy arrays. At each step of the simulation, an array of shape n by 2 is appended to this list representing the n positions as y,x vectors.
    observations : list[np.ndarray]
        A list of numpy arrays. At each step of the simulation, an array of shape n by 2 is appended to this list representing the n observations received by the agents.
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
        self.environment = environment.to_cpu()
        self.agent = agent.to_cpu()
        self.time_shift = time_shift if gpu_support and cp.get_array_module(time_shift) == np else cp.asnumpy(time_shift)
        self.horizon = horizon
        self.reward_discount = reward_discount
        self.start_time = datetime.now()

        # Simulation Tracking
        self.start_points = start_points if gpu_support and cp.get_array_module(start_points) == np else cp.asnumpy(start_points)
        self.actions = []
        self.positions = []
        self.observations = []
        self.timestamps = []

        self._running_sims = np.arange(self.n)
        self.done_at_step = np.full(self.n, fill_value=-1)

        # Other parameters
        self._simulation_dfs = None


    def add_step(self,
                 actions: np.ndarray,
                 next_positions: np.ndarray,
                 observations: np.ndarray,
                 is_done: np.ndarray,
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
        is_done : np.ndarray
            A boolean array of whether each agent has reached the source or not.
        interupt : np.ndarray
            A boolean array of whether each agent has to be terminated even if it hasnt reached the source yet.
        '''
        self._simulation_dfs = None

        # Time tracking
        self.timestamps.append(datetime.now())

        # Handle case cupy arrays are provided
        if gpu_support:
            actions = actions if cp.get_array_module(actions) == np else cp.asnumpy(actions)
            next_positions = next_positions if cp.get_array_module(next_positions) == np else cp.asnumpy(next_positions)
            observations = observations if cp.get_array_module(observations) == np else cp.asnumpy(observations)
            is_done = is_done if cp.get_array_module(is_done) == np else cp.asnumpy(is_done)
            interupt = interupt if cp.get_array_module(interupt) == np else cp.asnumpy(interupt)

        # Actions tracking
        action_all_sims = np.full((self.n,2), fill_value=-1)
        action_all_sims[self._running_sims] = actions
        self.actions.append(action_all_sims)

        # Next states tracking
        next_position_all_sims = np.full((self.n, 2), fill_value=-1)
        next_position_all_sims[self._running_sims] = next_positions
        self.positions.append(next_position_all_sims)

        # Observation tracking
        observation_all_sims = np.full((self.n,), fill_value=-1, dtype=float)
        observation_all_sims[self._running_sims] = observations
        self.observations.append(observation_all_sims)

        # Recording at which step the simulation is done if it is done
        self.done_at_step[self._running_sims[is_done]] = len(self.positions)

        # Updating the list of running sims
        self._running_sims = self._running_sims[~is_done & ~interupt]


    @property
    def analysis_df(self) -> pd.DataFrame:
        '''
        A Pandas DataFrame analyzing the results of the simulations.
        It aggregates the simulations in single rows, recording:
         - y_start and x_start: The x, y starting positions
         - optimal_steps_count: The minimal amount of steps to reach the source
         - converged:           Whether or not the simulation reached the source
         - reached_horizon:     Whether the failed simulation reached to horizon
         - steps_taken:         The amount of steps the agent took to reach the source, (horizon if the simulation did not reach the source)
         - discounted_rewards:  The discounted reward received by the agent over the course of the simulation
         - extra_steps:         The amount of extra steps compared to the optimal trajectory
         - t_min_over_t:        Normalized version of the extra steps measure, where it tends to 1 the least amount of time the agent took to reach the source compared to an optimal trajectory.

        For the measures (converged, steps_taken, discounted_rewards, extra_steps, t_min_over_t), the average and standard deviations are computed in rows at the top.
        '''
        # Dataframe creation
        df = pd.DataFrame(self.start_points, columns=['y_start', 'x_start'])
        df['optimal_steps_count'] = self.environment.distance_to_source(self.start_points)
        df['converged'] = self.done_at_step >= 0
        df['reached_horizon'] = np.all(self.positions[-1] != -1, axis=1) & (self.done_at_step == -1) & (len(self.positions) == self.horizon)
        df['steps_taken'] = np.where(df['converged'], self.done_at_step, len(self.positions))
        df['discounted_rewards'] = self.reward_discount ** df['steps_taken']
        df['extra_steps'] = df['steps_taken'] - df['optimal_steps_count']
        df['t_min_over_t'] = df['optimal_steps_count'] / df['steps_taken']

        # Reindex
        runs_list = [f'run_{i}' for i in range(self.n)]
        df.index = runs_list

        # Analysis aggregations
        columns_to_analyze = ['converged', 'reached_horizon', 'steps_taken', 'discounted_rewards', 'extra_steps', 't_min_over_t']
        success_averages = df.loc[df['converged'], columns_to_analyze].mean()
        succes_std = df.loc[df['converged'], columns_to_analyze].std()

        df.loc['mean', columns_to_analyze] = df[columns_to_analyze].mean()
        df.loc['standard_deviation', columns_to_analyze] = df[columns_to_analyze].std()

        df.loc['success_mean', columns_to_analyze] = success_averages
        df.loc['success_standard_deviation', columns_to_analyze] = succes_std

        # Bringing analysis rows to top
        df = df.reindex([
            'mean',
            'standard_deviation',
            'success_mean',
            'success_standard_deviation',
            *runs_list])

        return df


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
        done_sim_count = np.sum(self.done_at_step >= 0)
        failed_count = self.n - done_sim_count
        reached_horizon_count = int(np.sum(np.all(self.positions[-1] != -1, axis=1) & (self.done_at_step == -1) & (len(self.positions) == self.horizon)))
        summary_str = f'Simulations reached goal: {done_sim_count}/{self.n} ({failed_count} failures (reached horizon: {reached_horizon_count})) ({(done_sim_count*100)/self.n:.2f}% success)'

        if done_sim_count == 0:
            return summary_str
        
        # Metrics
        df = self.analysis_df

        summary_str += f"\n - {'Average step count:':<35} {df.loc['mean','steps_taken']:.3f} +- {df.loc['standard_deviation','steps_taken']:.2f} "
        summary_str += f"(Successfull only: {df.loc['success_mean','steps_taken']:.3f} +- {df.loc['success_standard_deviation','steps_taken']:.2f})"

        summary_str += f"\n - {'Extra steps:':<35} {df.loc['mean','extra_steps']:.3f} +- {df.loc['standard_deviation','extra_steps']:.2f} "
        summary_str += f"(Successful only: {df.loc['success_mean','extra_steps']:.3f} +- {df.loc['success_standard_deviation','extra_steps']:.2f})"

        summary_str += f"\n - {'Average discounted rewards (ADR):':<35} {df.loc['mean','discounted_rewards']:.3f} +- {df.loc['standard_deviation','discounted_rewards']:.2f} "
        summary_str += f"(Successfull only: {df.loc['success_mean','discounted_rewards']:.3f} +- {df.loc['success_standard_deviation','discounted_rewards']:.2f})"

        summary_str += f"\n - {'Tmin/T:':<35} {df.loc['mean','t_min_over_t']:.3f} +- {df.loc['standard_deviation','t_min_over_t']:.2f} "
        summary_str += f"(Successful only: {df.loc['success_mean','t_min_over_t']:.3f} +- {df.loc['success_standard_deviation','t_min_over_t']:.2f})"

        return summary_str


    @property
    def simulation_dfs(self) -> list[pd.DataFrame]:
        '''
        A list of the pandas DataFrame where each dataframe is a single simulation history.
        Each row is a different time instant of simulation process with each column being:
         - time (of the simulation data)
         - x
         - y
         - dx
         - dy
         - o (pure, not thresholded)
         - done (boolean)
        '''
        if self._simulation_dfs is None:
            self._simulation_dfs = []

            # Converting state, actions and observation to numpy arrays
            states_array = np.array(self.positions)
            action_array = np.array(self.actions)
            observation_array = np.array(self.observations)

            for i in range(self.n):
                length = self.done_at_step[i] if self.done_at_step[i] >= 0 else len(states_array)

                df = {
                    'time':  np.arange(length+1) + self.time_shift[i],
                    'y':     np.hstack([self.start_points[i,0], states_array[:length, i, 0]]),
                    'x':     np.hstack([self.start_points[i,1], states_array[:length, i, 1]]),
                    'dy':    np.hstack([[None], action_array[:length, i, 0]]),
                    'dx':    np.hstack([[None], action_array[:length, i, 1]]),
                    'o':     np.hstack([[None], observation_array[:length, i]]),
                    'done':  np.hstack([[None], np.where(np.arange(1,length+1) == self.done_at_step[i], 1, 0)])
                }

                # Append
                self._simulation_dfs.append(pd.DataFrame(df))

        return self._simulation_dfs


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
            env_name = f's_{self.environment.shape[0]}_{self.environment.shape[1]}'
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

        # Adding Environment and Agent info
        padding = [None] * len(combined_df)
        combined_df['timestamps'] = [self.start_time.strftime('%Y%m%d_%H%M%S%f')] + [ts.strftime('%H%M%S%f') for ts in self.timestamps] + padding[:-(len(self.timestamps)+1)]
        combined_df['horizon'] = [self.horizon] + padding[:-1]
        combined_df['reward_discount'] = [self.reward_discount] + padding[:-1]
        combined_df['environment'] = [self.environment.name, self.environment.saved_at] + padding[:-2]
        combined_df['agent'] = [self.agent.name, self.agent.class_name, self.agent.saved_at] + padding[:-3]

        # Saving csv
        combined_df.to_csv(folder + file, index=False)

        print(f'Simulations saved to: {folder + file}')

        if save_analysis:
            analysis_file = file.replace('.csv', '-analysis.csv')
            self.analysis_df.to_csv(folder + analysis_file)
            
            print(f"Simulation's analysis saved to: {folder + analysis_file}")


    @classmethod
    def load_from_file(cls,
                       file: str,
                       environment: bool | Environment = False,
                       agent: bool | Agent = False
                       ) -> 'SimulationHistory':
        '''
        Function to load the simulation history from a file.
        This can be useful to use the plot functions on the simulations saved in succh file.

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
        combined_df = pd.read_csv(file, dtype={
            'time':             int,
            'y':                float,
            'x':                float,
            'dy':               float,
            'dx':               float,
            'o':                float,
            'done':             float,
            'timestamps':       str,
            'horizon':          float,
            'reward_discount':  float,
            'environment':      str,
            'agent':            str
        })

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

        # Columns to retrieve
        columns = [
            'time',
            'y',
            'x',
            'dy',
            'dx',
            'o',
            'done'
        ]

        # Recreation of list of simulations
        sim_start_rows = np.argwhere(combined_df[['done']].isnull())[1:,0].tolist()
        n = len(sim_start_rows)+1

        simulation_arrays = np.split(combined_df[columns].to_numpy(), sim_start_rows)
        simulation_dfs = [pd.DataFrame(sim_array, columns=columns) for sim_array in simulation_arrays]

        # Making a combined numpy array with all the simulations
        sizes = np.array([len(sim_array) for sim_array in simulation_arrays])
        max_length = sizes.max()
        paddings = max_length - sizes

        padded_simulation_arrays = [np.pad(sim_arr, ((0,pad),(0,0)), constant_values=-1) for sim_arr, pad in zip(simulation_arrays, paddings)]
        all_simulation_arrays = np.array(padded_simulation_arrays).transpose((1,0,2))

        # Gathering start states
        start_points = all_simulation_arrays[0,:,1:3].astype(int)
        time_shift = all_simulation_arrays[0,:,0].astype(int)

        # Recreating action, state and observations
        positions = all_simulation_arrays[1:,:,1:3]
        actions = all_simulation_arrays[1:,:,3:5]
        observations = all_simulation_arrays[1:,:,5]
        done_at_step = np.where(all_simulation_arrays[sizes-1,np.arange(n),6], sizes-1, -1)

        # Building SimulationHistory instance
        hist = cls.__new__(cls)

        hist.n = len(start_points)
        hist.environment = environment.to_cpu() if isinstance(environment, Environment) else None
        hist.agent = agent.to_cpu() if isinstance(agent, Agent) else None
        hist.time_shift = time_shift
        hist.horizon = horizon
        hist.reward_discount = reward_discount
        hist.start_time = datetime.strptime(combined_df['timestamps'][0], '%Y%m%d_%H%M%S%f')

        hist.start_points = start_points
        hist._running_sims = None

        hist.positions = [*positions]
        hist.actions = [*actions]
        hist.observations = [*observations]
        hist.done_at_step = done_at_step
        hist.timestamps = [datetime.strptime(ts, '%H%M%S%f') for ts in combined_df['timestamps'][1:max_length]]
        # hist.timestamps = [datetime.strptime(str(int(ts)), '%H%M%S%f') for ts in combined_df['timestamps'][:max_length-1]]

        # Saving simulation dfs back
        hist._simulation_dfs = simulation_dfs
        
        return hist


    def plot(self,
             sim_id: int = 0,
             ax: plt.Axes = None
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
        # TODO: Make environment and agent optional for plotting
        assert (self.environment is not None) and (self.agent is not None), "Plot function not available as the environment and/or the agent used during the simulation is not linked to the simulation history."

        # Generate ax is not provided
        if ax is None:
            _, ax = plt.subplots(figsize=(18,3))

        # Retrieving sim
        sim = self.simulation_dfs[sim_id]

        # Plot setup
        env_shape = self.environment.shape
        ax.imshow(np.zeros(self.environment.shape), cmap='Greys', zorder=-100)
        ax.set_xlim(0, env_shape[1])
        ax.set_ylim(env_shape[0], 0)

        # Start
        start_coord = sim[['x', 'y']].to_numpy()[0]
        ax.scatter(start_coord[0], start_coord[1], c='green', label='Start')

        # Source circle
        goal_circle = plt.Circle(self.environment.source_position[::-1], self.environment.source_radius, color='r', fill=False, label='Source')
        ax.add_patch(goal_circle)

        # Until step
        seq = sim[['x','y']][1:].to_numpy()

        # Path
        ax.plot(seq[:,0], seq[:,1], zorder=-1, c='black', label='Path')

        # Something sensed
        if self.agent is not None:
            something_sensed = sim['o'][1:].to_numpy() > self.agent.threshold
            points_obs = seq[something_sensed,:]
            ax.scatter(points_obs[:,0], points_obs[:,1], zorder=1, label='Something observed')
        else:
            print('Agent used is not tracked')

        # Generate legend
        ax.legend()


    def plot_runtimes(self,
                      ax: plt.Axes = None
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


def run_test(agent: Agent,
             n: int | None = None,
             start_points: np.ndarray | None = None,
             environment: Environment | None = None,
             time_shift: int | np.ndarray = 0,
             time_loop: bool = True,
             horizon: int = 1000,
             reward_discount: float = 0.99,
             print_progress: bool = True,
             print_stats: bool = True,
             use_gpu: bool = False
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
    # Gathering n
    if n is None:
        if (start_points is None) or (len(start_points.shape) == 1):
            n = 1
        else:
            n = len(start_points)

    # Handle the case an specific environment is given
    if environment is not None:
        assert environment.shape == agent.environment.shape, "The provided environment's shape doesn't match the environment has been trained on..."
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

    # Move things to GPU if needed
    xp = np
    if use_gpu:
        assert gpu_support, f"GPU support is not enabled, the use_gpu option is not available."
        xp = cp

        # Move instances to GPU
        agent = agent.to_gpu()
        environment = environment.to_gpu()
        time_shift = cp.array(time_shift)

        if start_points is not None:
            start_points = cp.array(start_points)

    # Set start positions
    agent_position = None
    if start_points is not None:
        assert start_points.shape == (n, 2), 'The provided start_points are of the wrong shape'
        agent_position = start_points
    else:
        # Generating random starts
        agent_position = environment.random_start_points(n)

    # Initialize agent's state
    agent.initialize_state(n)

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
        new_agent_position = environment.move(agent_position, action)

        # Get an observation based on the new position of the agent
        observation = environment.get_observation(new_agent_position, time=(time_shift + i))

        # Check if the source is reached
        source_reached = environment.source_reached(new_agent_position)

        # Return the observation to the agent
        update_succeeded = agent.update_state(observation, source_reached)
        if update_succeeded is None:
            update_succeeded = xp.ones(len(source_reached) , dtype=bool)

        # Handling the case where simulations have reached the end
        sims_at_end = ((time_shift + i + 1) >= (math.inf if time_loop else environment.timesteps))

        # Agents to terminate
        to_terminate = source_reached | sims_at_end | ~update_succeeded

        # Interupt agents that reached the end
        agent_position = new_agent_position[~to_terminate]
        time_shift = time_shift[~to_terminate]
        agent.kill(simulations_to_kill=to_terminate)

        # Send the values to the tracker
        hist.add_step(
            actions=action,
            next_positions=new_agent_position,
            observations=observation,
            is_done=source_reached,
            interupt=to_terminate
        )

        # Early stopping if all agents done
        if len(agent_position) == 0:
            break

        # Update progress bar
        if print_progress:
            done_count = n-len(agent_position)
            iterator.set_postfix({'done ': f' {done_count} of {n} ({(done_count*100)/n:.1f}%)'})

    # If requested print the simulation start
    if print_stats:
        sim_end_ts = datetime.now()
        print(f'Simulations done in {(sim_end_ts - sim_start_ts).total_seconds():.3f}s:')
        print(hist.summary)

    return hist
