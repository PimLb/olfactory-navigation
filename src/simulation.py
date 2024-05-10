import math
import os
import inspect
import pandas as pd
import sys

from datetime import datetime
from matplotlib import pyplot as plt 
from tqdm.auto import trange

from src import Agent
from src.environment import Environment
from src.agent import *

import numpy as np
gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')


class SimulationHistory:
    '''
    Class to represent a list of the steps that happened during a simulation with:
        - the positions the agents pass by
        - the actions the agents take
        - the observations the agents receive ('observations')

    ...

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
    reward_discount : float (Default = 0.99)
        A discount to be applied to the rewards received by the agent. (eg: reward of 1 received at time n would be: 1 * reward_discount^n)
    
    Attributes
    ----------
    start_points : np.ndarray
    environment : Environment
    agent : Agent
    time_shift : np.ndarray
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
                 start_points:np.ndarray,
                 environment:Environment,
                 agent:Agent,
                 time_shift:np.ndarray,
                 reward_discount:float=0.99
                 ) -> None:
        # If only on state is provided, we make it a 1x2 vector
        if len(start_points.shape) == 1:
            start_points = start_points[None,:]

        # Fixed parameters
        self.n = len(start_points)
        self.environment = environment.to_cpu()
        self.agent = agent.to_cpu()
        self.time_shift = time_shift if gpu_support and cp.get_array_module(time_shift) == np else cp.asnumpy(time_shift)
        self.reward_discount = reward_discount
        self.start_time = datetime.now()

        # Simulation Tracking
        self.start_points = start_points if gpu_support and cp.get_array_module(start_points) == np else cp.asnumpy(start_points)
        self.actions = []
        self.positions = []
        self.observations = []
        self.timestamps = []
        # TODO: Add time tracking

        self._running_sims = np.arange(self.n)
        self.done_at_step = np.full(self.n, fill_value=-1)

        # Other parameters
        self._simulation_dfs = None


    def add_step(self,
                 actions:np.ndarray,
                 next_positions:np.ndarray,
                 observations:np.ndarray,
                 is_done:np.ndarray,
                 interupt:np.ndarray
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
         - steps_taken:         The amount of steps the agent took to reach the source, (horizon if the simulation did not reach the source)
         - discounted_rewards:  The discounted reward received by the agent over the course of the simulation
         - extra_steps:         The amount of extra steps compared to the optimal trajectory
         - t_min_over_t:         normalized version of the extra steps measure, where it tends to 1 the least amount of time the agent took to reach the source compared to an optimal trajectory.

        For the measures (converged, steps_taken, discounted_rewards, extra_steps, t_min_over_t), the average and standard deviations are computed in rows at the top.
        '''
        # Dataframe creation
        df = pd.DataFrame(self.start_points, columns=['y_start', 'x_start'])
        df['optimal_steps_count'] = self.environment.distance_to_source(self.start_points)
        df['converged'] = self.done_at_step >= 0
        df['steps_taken'] = np.where(df['converged'], self.done_at_step, len(self.positions))
        df['discounted_rewards'] = self.reward_discount ** df['steps_taken']
        df['extra_steps'] = df['steps_taken'] - df['optimal_steps_count']
        df['t_min_over_t'] = df['optimal_steps_count'] / df['steps_taken']

        # Reindex
        runs_list = [f'run_{i}' for i in range(self.n)]
        df.index = runs_list

        # Analysis aggregations
        columns_to_analyze = ['converged', 'steps_taken', 'discounted_rewards', 'extra_steps', 't_min_over_t']
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
        summary_str = f'Simulations reached goal: {done_sim_count}/{self.n} ({self.n-done_sim_count} failures) ({(done_sim_count*100)/self.n:.2f}%)'

        if done_sim_count == 0:
            return summary_str
        
        # Metrics
        df = self.analysis_df

        summary_str += f"\n\t- Average step count: {df.loc['mean','steps_taken']:.3f} +- {df.loc['standard_deviation','steps_taken']:.2f} "
        summary_str += f"(Successfull only: {df.loc['success_mean','steps_taken']:.3f} +- {df.loc['success_standard_deviation','steps_taken']:.2f})"

        summary_str += f"\n\t- Extra steps: {df.loc['mean','extra_steps']:.3f} +- {df.loc['standard_deviation','extra_steps']:.2f} "
        summary_str += f"(Successful only: {df.loc['success_mean','extra_steps']:.3f} +- {df.loc['success_standard_deviation','extra_steps']:.2f})"

        summary_str += f"\n\t- Average discounted rewards (ADR): {df.loc['mean','discounted_rewards']:.3f} +- {df.loc['standard_deviation','discounted_rewards']:.2f} "
        summary_str += f"(Successfull only: {df.loc['success_mean','discounted_rewards']:.3f} +- {df.loc['success_standard_deviation','discounted_rewards']:.2f})"

        summary_str += f"\n\t- Tmin/T: {df.loc['mean','t_min_over_t']:.3f} +- {df.loc['standard_deviation','t_min_over_t']:.2f} "
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
             file:str|None=None,
             folder:str|None=None,
             save_analysis:bool=True,
             save_components:bool=False
             ) -> None:
        '''
        Function to save the simulation history to a csv file in a given folder.
        Additionally, an analysis of the runs can be saved if the save_analysis is enabled.
        The environment and agent used can be saved in the saved folder by enabling the 'save_component' parameter.

        Parameters
        ----------
        file : str (optional)
            The name of the file the simulation histories will be saved to.
            If it is not provided, it will be by default "Simulations-<env_name>-n_<sim_count>-<sim_start_timestamp>-horizon_<max_sim_length>.csv"
        folder : str (optional)
            Folder to save the simulation histories to.
            If the folder name is not provided the current folder will be used.
        save_analysis : bool (Default = True)
            Whether to save an additional csv file with an analysis of the runs of the simulation.
            It will contain the amount of steps taken, the amount of extra steps compared to optimality, the discounted rewards and the ratio between optimal trajectory and the steps taken.
            The means and standard deviations of all the runs are also computed.
            The file will have the same name as the simulation history file with an additional '-analysis' tag at the end.
        save_components : bool (Default = False)
            Whether or not to save the environment and agent along with the simulation histories in the given folder.
        '''
        # Handle file name
        if file is None:
            env_name = f's_{self.environment.shape[0]}_{self.environment.shape[1]}'
            file = f'Simulations-{env_name}-n_{self.n}-{self.start_time.strftime("%m%d%Y_%H%M%S")}-horizon_{len(self.positions)}.csv'

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
                       file:str,
                       environment:Environment|None=None,
                       agent:Agent|None=None
                       ) -> 'SimulationHistory':
        '''
        Function to load the simulation history from a file.
        This can be useful to use the plot functions on the simulations saved in succh file.

        The environment and agent can provided as a backup in the case they cannot be loaded from the file.
        
        Parameters
        ----------
        file : str
            A file (with the path) of the simulation histories csv. (the analysis file cannot be used for this)
        environment : Environment (optional)
            An environment instance to be linked with the simulation history object.
            If an environment can be loaded from the path found in the file, this parameter will be ignored.
            But if this loading fails and no environment is provided, the loading will fail.
        agent : Agent (optional)
            An agent instance to be linked with the simulation history object.
            If an agent can be loaded from the path found in the file, this parameter will be ignored.
            But if this loading fails and no agent is provided, the loading will fail.

        Returns
        -------
        hist : SimulationHistory
            The loaded instance of a simulation history object.
        '''
        combined_df = pd.read_csv(file)

        # Retrieving reward discount
        reward_discount = combined_df['reward_discount'][0]

        # Retrieving environment
        loaded_environment = None
        environment_name = combined_df['environment'][0]
        if combined_df['environment'][1] is not None:
            try:
                loaded_environment = Environment.load(combined_df['environment'][1])
            except:
                print(f'Failed to retrieve "{environment_name}" environment from memory')

        if loaded_environment is not None:
            print(f'Environment "{environment_name}" loaded from memory' + (' (Ignoring environment provided as a parameter)' if environment is not None else ''))
            environment = loaded_environment

        if environment is None:
            raise Exception('No environment could be linked, the simulation hisotry cannot be instanciated. Provide an environment to resolve this.')

        # Retrieving agent
        loaded_agent = None
        agent_name = combined_df['agent'][0]
        agent_class = combined_df['agent'][1]
        if combined_df['agent'][2] is not None:
            try:
                for (class_name, class_obj) in inspect.getmembers(sys.modules[__name__], inspect.isclass):
                    if class_name == agent_class:
                        loaded_agent = class_obj.load(combined_df['agent'][2])
                        break
            except:
                print(f'Failed to retrieve "{agent_name}" agent from memory')

        if loaded_agent is not None:
            print(f'Agent "{agent_name}" loaded from memory' + (' (Ignoring agent provided as a parameter)' if agent is not None else ''))
            agent = loaded_agent

        if agent is None:
            raise Exception('No agent could be linked, the simulation hisotry cannot be instanciated. Provide an agent to resolve this.')

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
        simulation_dfs = []
        sim_start_rows = [None] + np.argwhere(combined_df[['done']].isnull())[1:,0].tolist() + [None]
        n = len(sim_start_rows)-1

        for i in range(n):
            simulation_dfs.append(combined_df[columns].iloc[sim_start_rows[i]:sim_start_rows[i+1]])

        # Gathering start states
        start_points = np.array([sim[['y', 'x']].iloc[0] for sim in simulation_dfs])
        time_shift = np.array([sim['time'].iloc[0] for sim in simulation_dfs])

        # Generation of SimHist instance
        hist = SimulationHistory(
            start_points=start_points,
            environment=environment,
            agent=agent,
            time_shift=time_shift,
            reward_discount=reward_discount
        )

        max_length = max(len(sim) for sim in simulation_dfs)

        # Recreating action, state and observations
        positions = np.full((max_length-1, n, 2), -1)
        actions = np.full((max_length-1, n, 2), -1)
        observations = np.full((max_length-1, n), -1)
        done_at_step = np.full((n,), -1)

        for i, sim in enumerate(simulation_dfs):
            positions[:len(sim)-1, i, :] = sim[['y','x']].to_numpy()[1:]
            actions[:len(sim)-1, i, :] = sim[['dy','dx']].to_numpy()[1:]
            observations[:len(sim)-1, i] = sim[['o']].to_numpy()[1:,0]
            done_at_step[i] = len(sim)-1 if sim['done'].iloc[-1] == 1 else -1

        hist.positions = [arr for arr in positions]
        hist.actions = [arr for arr in actions]
        hist.observations = [arr for arr in observations]
        hist.done_at_step = done_at_step

        # Saving simulation dfs back
        hist._simulation_dfs = simulation_dfs
        
        return hist


    def plot(self,
            sim_id:int=0,
            ax=None
            ) -> None:
        '''
        Function to plot a the trajectory of a given simulation.
        An ax can be use to plot it on.

        Parameters
        ----------
        sim_id : int (default=0)
            The id of the simulation to plot.
        ax : (Optional)
            The ax on which to plot the path.
        '''
        # Generate ax is not provided
        if ax is None:
            _, ax = plt.subplots(figsize=(18,3))

        # Initial clearing
        ax.clear()

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
            something_sensed = sim['o'][1:].to_numpy() > self.agent.treshold
            points_obs = seq[something_sensed,:]
            ax.scatter(points_obs[:,0], points_obs[:,1], zorder=1, label='Something observed')
        else:
            print('Agent used is not tracked')

        # Generate legend
        ax.legend()


def run_test(agent:Agent,
             n:int|None=None,
             start_points:np.ndarray|None=None,
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
    n : int (optional)
        How many simulation to run in parallel.
        n is optional but it needs to match with what is provided in start_points.
    start_points : np.ndarray (optional)
        The starting points of the simulation in 2d space.
        If not provided, n random points will be generated based on the start probabilities of the environment.
        Else, the amount of start_points need to match to n, if it is provided.
    environment : Environment (optional)
        The environment to run the simulations in.
        By default, the environment linked to the agent will used.
        This parameter is intended if the environment needs to be modified compared to environment the agent was trained on.
    time_shift : int or np.ndarray (default = 0)
        The time at which to start the olfactory simulation array.
        It can be either a single value, or n values.
    time_loop : bool (default = True)
        Whether to loop the time if reaching the end. (starts back at 0)
    horizon : int (default = 1000)
        The amount of steps to run the simulation for before killing the remaining simulations.
    reward_discount : float (default = 0.99)
        How much a given reward is discounted based on how long it took to get it.
        It is purely used to compute the Average Discount Reward (ADR) after the simulation.
    print_progress : bool (default = True)
        Wheter to show a progress bar of what step the simulations are at.
    print_stats : bool (default = True)
        Wheter to print the stats at the end of the run.
    use_gpu : bool (default = False)
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
    if use_gpu:
        assert gpu_support, f"GPU support is not enabled, the use_gpu option is not available."

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
        agent.update_state(observation, source_reached)

        # Handling the case where simulations have reached the end
        sims_at_end = ((time_shift + i + 1) >= (math.inf if time_loop else len(environment.grid)))

        # Interupt agents that reached the end
        agent_position = new_agent_position[~source_reached & ~sims_at_end]
        time_shift = time_shift[~source_reached & ~sims_at_end]
        agent.kill(simulations_to_kill=sims_at_end[~source_reached])

        # Send the values to the tracker
        hist.add_step(
            actions=action,
            next_positions=new_agent_position,
            observations=observation,
            is_done=source_reached,
            interupt=sims_at_end
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
