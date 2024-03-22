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
                 environment:Environment,
                 treshold:float|None=3e-6,
                 name:str|None=None
                 ) -> None:
        self.environment = environment
        self.treshold = treshold
        self.name = name

        self.saved_at = None

        self.on_gpu = False
        self._alternate_version = None


    def to_gpu(self) -> 'Agent':
        '''
        Function to send the numpy arrays of the agent to the gpu.
        It returns a new instance of the Agent class with the arrays on the gpu
        '''
        raise NotImplementedError('The to_gpu function is not implemented, make an agent subclass to implement the method')


    def to_cpu(self) -> 'Agent':
        '''
        Function to send the numpy arrays of the agent to the gpu.
        It returns a new instance of the Agent class with the arrays on the gpu

        Returns
        -------
        cpu_agent : Agent
            A new environment instance where the arrays are on the cpu memory.
        '''
        if self.on_gpu:
            assert self._alternate_version is not None, "Something went wrong"
            return self._alternate_version

        return self


    def train(self) -> None:
        '''
        Function to call the particular flavour of training of the agent.
        '''
        raise NotImplementedError('The train function is not implemented, make an agent subclass to implement the method')


    def save(self,
             folder:str|None=None,
             force:bool=False
             ) -> None:
        '''
        Function to save a trained agent to memory.
        '''
        raise NotImplementedError('The save function is not implemented, make an agent subclass to implement the method')


    @classmethod
    def load(cls, folder:str):
        '''
        Function to save a trained agent to memory.
        '''
        raise NotImplementedError('The load function is not implemented, make an agent subclass to implement the method')


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

    def kill(self,
             simulations_to_kill:np.ndarray
             ) -> None:
        '''
        Function to kill any simulations that still haven't reached the source
        '''
        raise NotImplementedError('The kill function is not implemented, make an agent subclass to implement the method')
