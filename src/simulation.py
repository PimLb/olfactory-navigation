import numpy as np
import random

from src import Environment
from src import Agent


class SimulationHistory:
    def __init__(self) -> None:
        pass


class Simulation:
    def __init__(self
                 ) -> None:
        # Should we have rewards discount as a param here?
        pass


    def run_test(self,
                 n:int,
                 environment:Environment,
                 agent:Agent,
                 horizon:int
                 ) -> SimulationHistory:
        
        # Set position at random
        rand_x = int(random.random() * environment.padded_width)
        rand_y = int(random.random() * environment.padded_height)
        agent_position = np.array([rand_x, rand_y])

        # Initialize agent's state
        agent.initialize_state()

        # Simulation loop
        for iter in range(horizon):
            # Letting agent choose the action to take based on it's curent state
            action = agent.choose_action()

            # Updating the agent's actual position (hidden to him)
            new_agent_position = environment.move(agent_position, action)

            # Get an observation based on the new position of the agent
            observation = environment.get_observation(new_agent_position)

            # Check if the source is reached
            source_reached = environment.source_reached(new_agent_position)

            # Return the observation to the agent
            agent_done = agent.update_state(observation, source_reached)

            # Early stopping if the agent is done
            if agent_done:
                break

        # TODO: Implement history
        return SimulationHistory()
