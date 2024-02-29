from numpy import ndarray
from src.environment import Environment
from ..agent import Agent


class PBVI_Agent(Agent):
    def __init__(self, environment: Environment) -> None:
        super().__init__(environment)

    
    def train(self) -> None:
        return super().train()
    

    def expand(self) -> None:
        raise NotImplementedError('PBVI class is abstract so expand function is not implemented, make an PBVI_agent subclass to implement the method')


    def backup(self) -> None:
        # TODO
        pass


    def choose_action(self) -> ndarray:
        return super().choose_action()


    def update_state(self, observation: int, source_reached: bool) -> None:
        return super().update_state(observation, source_reached)