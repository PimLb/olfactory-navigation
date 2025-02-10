import importlib
from olfactory_navigation.agent import Agent
from olfactory_navigation.environment import Environment
from olfactory_navigation.simulation import run_test, SimulationHistory

DEFAULT_VERSION = "0.0.0"
__all__ = (
    'Agent',
    'Environment',
    'run_test',
    'SimulationHistory',
)

try:
    from importlib.metadata import version
    __version__ = version("olfactory-navigation")
except importlib.metadata.PackageNotFoundError:
    __version__ = DEFAULT_VERSION