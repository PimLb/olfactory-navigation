import os
import numpy as np
import pandas as pd

from datetime import datetime
from matplotlib import pyplot as plt 
from tqdm.auto import trange

from src import Agent
from src.environment import Environment


class SimulationHistory:
    '''
    Class to represent a list of the steps that happened during a simulation or set of simulations with:
        - the state the agent passes by ('state')
        - the action the agent takes ('action')
        - the observation the agent takes ('observation')

    # TODO Reword this
    ...

    Parameters
    ----------
    start_state : int
        The initial state in the simulation.
    
    Attributes
    ----------
    states : list[int]
        A list of recorded states through which the agent passed by during the simulation process.
    actions : list[int]
        A list of recorded actions the agent took during the simulation process.
    observations : list[int]
        A list of recorded observations gotten by the agent during the simulation process.
    '''
    def __init__(self,
                 start_state:np.ndarray,
                 environment:Environment,
                 agent:Agent,
                 time_shift:np.ndarray,
                 reward_discount:float=0.99
                 ) -> None:
        # If only on state is provided, we make it a 1x2 vector
        if len(start_state.shape) == 1:
            start_state = start_state[None,:]

        # Fixed parameters
        self.environment = environment
        self.agent = agent
        self.time_shift = time_shift
        self.reward_discount = reward_discount
        self.start_time = datetime.now()

        # Simulation Tracking
        self.start_state = start_state
        self.actions = []
        self.states = []
        self.observations = []

        self._running_sims = np.arange(len(start_state))
        self.done_at_step = np.full(len(start_state), fill_value=-1)

        # Other parameters
        self._simulation_dfs = None


    def add_step(self,
                 action:np.ndarray,
                 next_state:np.ndarray,
                 observation:np.ndarray,
                 is_done:np.ndarray
                 ) -> None:
        '''
        # TODO: Update this
        Function to add a step in the simulation history

        Parameters
        ----------
        action : int
            The action that was taken by the agent.
        next_state : int
            The state that was reached by the agent after having taken action.
        next_belief : Belief
            The new belief of the agent after having taken an action and received an observation.
        observation : int
            The observation the agent received after having made an action.
        '''
        self._simulation_dfs = None

        # Actions tracking
        action_all_sims = np.full((len(self.start_state),2), fill_value=-1)
        action_all_sims[self._running_sims] = action
        self.actions.append(action_all_sims)

        # Next states tracking
        next_state_all_sims = np.full((len(self.start_state), 2), fill_value=-1)
        next_state_all_sims[self._running_sims] = next_state
        self.states.append(next_state_all_sims)

        # Observation tracking
        observation_all_sims = np.full((len(self.start_state),), fill_value=-1, dtype=float)
        observation_all_sims[self._running_sims] = observation
        self.observations.append(observation_all_sims)

        # Recording at which step the simulation is done if it is done
        self.done_at_step[self._running_sims[is_done]] = len(self.states)

        # Updating the list of running sims
        self._running_sims = self._running_sims[~is_done]


    @property
    def summary(self) -> str:
        n = len(self.start_state)
        done_sim_count = np.sum(self.done_at_step >= 0)
        summary_str = f'Simulations reached goal: {done_sim_count}/{n} ({n-done_sim_count} failures) ({(done_sim_count*100)/n:.2f})'

        if done_sim_count == 0:
            return summary_str
        
        # Metrics
        sim_is_done = self.done_at_step >= 0
        done_at_step_with_max = np.where(self.done_at_step < 0, len(self.states), self.done_at_step)
        discounted_rewards = (self.reward_discount) ** done_at_step_with_max

        t_min = self.environment.distance_to_source(self.start_state)
        extra_steps = done_at_step_with_max - t_min
        t_min_over_t = t_min / done_at_step_with_max

        summary_str += f'\n\t- Average step count: {np.average(done_at_step_with_max)} (Successfull only: {np.average(self.done_at_step[sim_is_done])})'
        summary_str += f'\n\t- Extra steps: {np.average(extra_steps)} (Successful only: {np.average(extra_steps[sim_is_done])})'
        summary_str += f'\n\t- Average discounted rewards (ADR): {np.average(discounted_rewards):.3f} (Successfull only: {np.average(discounted_rewards[sim_is_done]):.3f}) (discount: {self.reward_discount})'
        summary_str += f'\n\t- Tmin/T: {np.average(t_min_over_t):.3f} (Successful only: {np.average(t_min_over_t[sim_is_done]):.3f})'

        return summary_str


    @property
    def simulation_dfs(self) -> list[pd.DataFrame]:
        if self._simulation_dfs is None:
            self._simulation_dfs = []

            # Converting state, actions and observation to numpy arrays
            states_array = np.array(self.states)
            action_array = np.array(self.actions)
            observation_array = np.array(self.observations)

            for i in range(len(self.start_state)):
                length = self.done_at_step[i] if self.done_at_step[i] >= 0 else len(states_array)

                df = {
                    'time':  np.arange(length+1) + self.time_shift[i],
                    'y':     np.hstack([self.start_state[i,0], states_array[:length, i, 0]]),
                    'x':     np.hstack([self.start_state[i,1], states_array[:length, i, 1]]),
                    'dy':    np.hstack([[0], action_array[:length, i, 0]]),
                    'dx':    np.hstack([[0], action_array[:length, i, 1]]),
                    'o':     np.hstack([[0], observation_array[:length, i]]),
                    'done':  np.where(np.arange(length+1) == self.done_at_step[i], 1, 0)
                }

                # Append
                self._simulation_dfs.append(pd.DataFrame(df))

        return self._simulation_dfs


    def save(self,
             file:str|None=None,
             folder:str|None=None
             ) -> None:
        '''
        Function to save the simulation history to a given folder.

        Parameters
        ----------
        folder : str (optional)
            Folder to save the simulation histories to.
            The folder has to be empty, or, if it doesnt exist yet, it will be created.
            If the folder name is not provided a combination of the environment name, the current timestamp and the amount of simulations run.
        '''
        if file is None:
            env_name = f's_{self.environment.shape[0]}_{self.environment.shape[1]}'
            file = f'Simulations-{env_name}-n_{len(self.start_state)}-horizon_{len(self.states)}-{self.start_time.strftime("%m%d%Y_%H%M%S")}.csv'

        if folder is None:
            folder = './'

        if '/' not in folder:
            folder = './' + folder

        if not os.path.exists(folder):
            os.mkdir(folder)

        if not folder.endswith('/'):
            folder += '/'

        # Create csv file
        combined_df = pd.concat(self.simulation_dfs)

        # Adding Environment and Agent info
        padding = [None] * len(combined_df)
        combined_df['reward_discount'] = [self.reward_discount] + padding[:-1]
        combined_df['environment'] = [self.environment.name, self.environment.saved_at] + padding[:-2]
        combined_df['agent'] = [] + padding[:] # TODO

        # Saving csv
        combined_df.to_csv(folder + file, index=False)

        print(f'Simulations saved to: {folder + file}')


    @classmethod
    def load_from_file(cls,
                       file:str,
                       environment:Environment|None=None
                       ) -> 'SimulationHistory':
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
        sim_start_rows = [None] + np.argwhere(combined_df[['time']].diff() < 1)[:,0].tolist() + [None]
        n = len(sim_start_rows)-1

        for i in range(n):
            simulation_dfs.append(combined_df[columns].iloc[sim_start_rows[i]:sim_start_rows[i+1]])

        # Gathering start states
        start_states = np.array([sim[['y', 'x']].iloc[0] for sim in simulation_dfs])
        time_shift = np.array([sim['time'].iloc[0] for sim in simulation_dfs])

        # Generation of SimHist instance
        hist = SimulationHistory(
            start_state=start_states,
            environment=environment,
            agent=None, # TODO
            time_shift=time_shift,
            reward_discount=reward_discount
        )

        max_length = max(len(sim) for sim in simulation_dfs)

        # Recreating action, state and observations
        states = np.full((max_length-1, n, 2), -1)
        actions = np.full((max_length-1, n, 2), -1)
        observations = np.full((max_length-1, n), -1)
        done_at_step = np.full((n,), -1)

        for i, sim in enumerate(simulation_dfs):
            states[:len(sim)-1, i, :] = sim[['y','x']].to_numpy()[1:]
            actions[:len(sim)-1, i, :] = sim[['dy','dx']].to_numpy()[1:]
            observations[:len(sim)-1, i] = sim[['o']].to_numpy()[1:,0]
            done_at_step[i] = len(sim)-1 if sim['done'].iloc[-1] == 1 else -1

        hist.states = [arr for arr in states]
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
        if self.agent is not None: # TODO: Agent to save
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
             horizon:int=1000,
             reward_discount:float=0.99,
             print_progress:bool=True,
             print_stats:bool=True
             ) -> SimulationHistory:
    '''
    Function to run n simulations for a given agent in its environment (or a given agent).
    The simulations start either from random start points or provided trough the start_points parameter.
    The simulation can have a shifted initial time (in the olfactory simulation).

    # TODO Explain sim loop

    Parameters
    ----------

    Returns
    -------
    hist : SimulationHistory
        A SimulationHistory object that tracked all the positions, actions and observations.
    '''
    # Gathering n
    if n is None:
        if start_points is None:
            n = 1
        else:
            n = len(start_points)

    # Handle the case an specific environment is given
    if environment is not None:
        assert environment.shape == agent.environment.shape
        print('Using the provided environment, not the agent environment.')
    else:
        environment = agent.environment

    # Set start positions
    agent_position = None
    if start_points is not None:
        assert start_points.shape == (n, 2), 'The provided start_points are of the wrong shape'
        agent_position = start_points
    else:
        # Generating random starts
        agent_position = environment.random_start_points(n)

    # Timeshift
    if isinstance(time_shift, int):
        time_shift = np.ones(n) * time_shift
    else:
        time_shift = np.array(time_shift)
        assert time_shift.shape == (n,), f"time_shift array has a wrong shape (Given: {time_shift.shape}, expected ({n},))"
    time_shift = time_shift.astype(int)

    # Initialize agent's state
    agent.initialize_state(n)

    # Create simulation history tracker
    hist = SimulationHistory(
        start_state=agent_position,
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

        # Send the values to the tracker
        hist.add_step(
            action=action,
            next_state=new_agent_position,
            observation=observation,
            is_done=source_reached
        )

        # Updating the list of agent positions and filtering to only the ones still running
        agent_position = new_agent_position[~source_reached]

        # Filtering time_shift list
        time_shift = time_shift[~source_reached]

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
