import inspect
import numpy as np

from olfactory_navigation.environment import Environment


class AgentState:
    def __init__(self) -> None:
        pass


class Agent:
    '''
    Generic agent class
    '''
    def __init__(self,
                 environment: Environment,
                 threshold: float | None = 3e-6,
                 name: str | None = None
                 ) -> None:
        self.environment = environment
        self.threshold = threshold

        # setup name
        if name is None:
            self.name = self.class_name
            self.name += f'-tresh_{self.threshold}'
        else:
            self.name = name

        self.saved_at = None

        self.on_gpu = False
        self._alternate_version = None


    @property
    def class_name(self):
        return self.__class__.__name__


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
             folder: str | None = None,
             force: bool = False,
             save_environment: bool = False
             ) -> None:
        '''
        Function to save a trained agent to memory.
        '''
        raise NotImplementedError('The save function is not implemented, make an agent subclass to implement the method')


    @classmethod
    def load(cls,
             folder: str
             ) -> 'Agent':
        '''
        Function to load a trained agent from memory.
        '''
        from olfactory_navigation import agents

        for name, obj in inspect.getmembers(agents):
            if inspect.isclass(obj) and (name in folder) and issubclass(obj, cls) and (obj != cls):
                return obj.load(folder)

        raise NotImplementedError('The load function is not implemented, make an agent subclass to implement the method')


    def modify_environment(self,
                           new_environment: Environment
                           ) -> 'Agent':
        '''
        Function to modify the environment of the agent.
        If the agent is already trained, the trained element should also be adapted to fit this new environment.
        '''
        raise NotImplementedError('The modify_environment function is not implemented, make an agent subclass to implement the method')


    def initialize_state(self,
                         n: int = 1
                         ) -> None:
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
                     observation: int | np.ndarray,
                     source_reached: bool | np.ndarray
                     ) -> None | np.ndarray:
        '''
        Function to update the internal state of the agent based on the previous action taken and the observation received.
        '''
        raise NotImplementedError('The update_state function is not implemented, make an agent subclass to implement the method')


    def kill(self,
             simulations_to_kill: np.ndarray
             ) -> None:
        '''
        Function to kill any simulations that still haven't reached the source
        '''
        raise NotImplementedError('The kill function is not implemented, make an agent subclass to implement the method')
