import numpy as np

from datetime import datetime
from tqdm.auto import trange

from src import Agent
from src.environment import Environment


class SimulationHistory:
    '''
    Class to represent a list of the steps that happened during a simulation or set of simulations with:
        - the state the agent passes by ('state')
        - the action the agent takes ('action')
        - the observation the agent takes ('observation')

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
    def __init__(self, start_state:np.ndarray) -> None:
        # If only on state is provided, we make it a 1x2 vector
        if len(start_state.shape) == 1:
            start_state = start_state[None,:]
        
        self.start_state = start_state
        self.actions = []
        self.states = []
        self.observations = []

        self._running_sims = np.arange(len(start_state))
        self.done_at_step = np.full(len(start_state), fill_value=-1)


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
        # Actions tracking
        action_all_sims = np.full((len(self.start_state),2), fill_value=-1)
        action_all_sims[self._running_sims] = action
        self.actions.append(action_all_sims)

        # Next states tracking
        next_state_all_sims = np.full((len(self.start_state), 2), fill_value=-1)
        next_state_all_sims[self._running_sims] = next_state
        self.states.append(next_state_all_sims)

        # Observation tracking
        observation_all_sims = np.full((len(self.start_state),), fill_value=-1)
        observation_all_sims[self._running_sims] = observation
        self.observations.append(action_all_sims)

        # Recording at which step the simulation is done if it is done
        self.done_at_step[self._running_sims[is_done]] = len(self.states)

        # Updating the list of running sims
        self._running_sims = self._running_sims[~is_done]


def run_test(agent:Agent,
             n:int=1,
             environment:Environment|None=None,
             horizon:int=1000,
             reward_discount:float=0.99,
             print_progress:bool=True,
             print_stats:bool=True
             ) -> SimulationHistory:
    
    if environment is None and print_progress:
        print('Environment not provided, using the agent\'s environment')
        environment = agent.environment
    else:
        assert environment.shape == agent.environment.shape

    # Set position at random
    agent_position = environment.random_start_points(n)

    # Initialize agent's state
    agent.initialize_state(n)

    # Create simulation history tracker
    hist = SimulationHistory(agent_position)

    # Track begin of simulation ts
    sim_start_ts = datetime.now()

    # Simulation loop
    iterator = trange(horizon) if print_progress else range(horizon)
    for _ in iterator:
        # Letting agent choose the action to take based on it's curent state
        action = agent.choose_action()

        # Updating the agent's actual position (hidden to him)
        new_agent_position = environment.move(agent_position, action)

        # Get an observation based on the new position of the agent
        observation = environment.get_observation(new_agent_position)

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

        # Early stopping if all agents done
        if len(agent_position) == 0:
            break

        # Update progress bar
        if print_progress:
            done_count = n-len(agent_position)
            iterator.set_postfix({'done ': f' {done_count} of {n} ({(done_count*100)/n:.1f}%)'})

    if print_stats:
        sim_end_ts = datetime.now()

        # Computing stats
        done_sim_count = np.sum(hist.done_at_step >= 0)
        done_at_step_sum = np.sum(hist.done_at_step[hist.done_at_step >= 0])
        discounted_rewards = ((hist.done_at_step >= 0).astype(int) * reward_discount) ** hist.done_at_step[hist.done_at_step >= 0]

        t_min = environment.distance_to_source(hist.start_state)
        t_min_over_t = t_min[hist.done_at_step >= 0] / hist.done_at_step[hist.done_at_step > 0]

        # Printing stats
        print(f'All {n} simulations done in {(sim_end_ts - sim_start_ts).total_seconds():.3f}s:')
        print(f'\t- Simulations reached goal: {done_sim_count}/{n} ({n-done_sim_count} failures) ({(done_sim_count*100)/n:.2f})')
        print(f'\t- Average step count: {(done_at_step_sum / n)}')
        print(f'\t- Average discounted rewards (ADR): {np.average(discounted_rewards):.3f} (discount: {reward_discount})')
        print(f'\t- Tmin/T: {np.average(t_min_over_t):.3f}')

    return hist
