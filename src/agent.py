import numpy as np

from src.environment import Environment


class AgentState:
    def __init__(self) -> None:
        pass


class Agent:
    '''
    Generic agent class
    '''
    def __init__(self,
                 environment:Environment
                 ) -> None:
        self.environment = environment
        self.state = None


    def train(self) -> None:
        '''
        Function to call the particular flavour of training of the agent.
        '''
        raise NotImplementedError('The train function is not implemented, make an agent subclass to implement the method')


    def initialize_state(self, n:int=1) -> None:
        '''
        Function to initialize the state of the agent. Which is meant to contain concepts such as the "memory" or "belief" of the agent.

        Parameters
        ----------
        n : int, default=1
            How many agents to initialize.
        '''
        raise NotImplementedError('The initialize_state function is not implemented, make an agent subclass to implement the method')


    def choose_action(self) -> np.ndarray:
        '''
        Function to allow for the agent to choose an action to take based on its current state.
        It then stores this action in its state.

        Returns
        -------
        movement_vector : np.ndarray
            A vector in 2D space of the movement the agent will take
        '''
        raise NotImplementedError('The choose_action function is not implemented, make an agent subclass to implement the method')


    def update_state(self,
                     observation:int|np.ndarray,
                     source_reached:bool|np.ndarray
                     ) -> None:
        '''
        Function to update the internal state of the agent based on the previous action taken and the observation received.
        '''
        raise NotImplementedError('The update_state function is not implemented, make an agent subclass to implement the method')
