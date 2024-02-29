from ..environment import Environment
from ..agents.pbvi_agent import PBVI_Agent


class FSVI_Agent(PBVI_Agent):
    def __init__(self, environment: Environment) -> None:
        super().__init__(environment)