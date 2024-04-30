import warnings
from ..environment import Environment
from ..agent import Agent
from .model_based_util.pomdp import Model
from .model_based_util.belief import Belief, BeliefSet

import numpy as np
gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')


class Infotaxis_Agent(Agent):
    '''
    # TODO
    '''
    def __init__(self,
                 environment:Environment,
                 treshold:float|None=3e-6,
                 name:str|None=None
                 ) -> None:
        super().__init__(environment)

        self.model = Model.from_environment(environment, treshold)
        self.treshold = treshold

        # setup name
        if name is None:
            self.name = self.class_name
            self.name += f'-tresh_{self.treshold}'
        else:
            self.name = name

        # Status variables
        self.beliefs = None
        self.action_played = None


    def to_gpu(self) -> Agent:
        '''
        Function to send the numpy arrays of the agent to the gpu.
        It returns a new instance of the Agent class with the arrays on the gpu

        Returns
        -------
        gpu_agent
        '''
        # Generating a new instance
        cls = self.__class__
        gpu_agent = cls.__new__(cls)

        # Copying arguments to gpu
        for arg, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                setattr(gpu_agent, arg, cp.array(val))
            elif isinstance(val, Model):
                gpu_agent.model = self.model.gpu_model
            elif isinstance(val, BeliefSet):
                gpu_agent.beliefs = self.beliefs.to_gpu()
            else:
                setattr(gpu_agent, arg, val)

        # Self reference instances
        self._alternate_version = gpu_agent
        gpu_agent._alternate_version = self

        gpu_agent.on_gpu = True
        return gpu_agent


    def initialize_state(self,
                         n:int=1
                         ) -> None:
        '''
        To use an agent within a simulation, the agent's state needs to be initialized.
        The initialization consists of setting the agent's initial belief.
        Multiple agents can be used at once for simulations, for this reason, the belief parameter is a BeliefSet by default.
        
        Parameters
        ----------
        n : int, default=1
            How many agents are to be used during the simulation.
        '''
        self.beliefs = BeliefSet(self.model, [Belief(self.model) for _ in range(n)])


    def choose_action(self) -> np.ndarray:
        '''
        Function to let the agent or set of agents choose an action based on their current belief.
        As for the Infotaxis principle, it will choose an action that will minimize the sum of next entropies.

        Returns
        -------
        movement_vector : np.ndarray
            A single or a list of actions chosen by the agent(s) based on their belief.
        '''
        xp = np if not self.on_gpu else cp

        n = len(self.beliefs)
        
        best_entropy = xp.ones(n) * -1
        best_action = xp.ones(n, dtype=int) * -1

        current_entropy = self.beliefs.entropies

        for a in self.model.actions:
            total_entropy = xp.zeros(n)

            for o in self.model.observations:
                b_ao = self.beliefs.update(actions=np.ones(n, dtype=int)*a,
                                           observations=np.ones(n, dtype=int)*o,
                                           throw_error=False)

                # Computing entropy
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    b_ao_entropy = b_ao.entropies

                b_prob = xp.dot(self.beliefs.belief_array, xp.sum(self.model.reachable_transitional_observation_table[:,a,o,:], axis=1))

                total_entropy += (b_prob * (current_entropy - b_ao_entropy))
            
            # Checking if action is superior to previous best
            superiority_mask = best_entropy < total_entropy
            best_action[superiority_mask] = a
            best_entropy[superiority_mask] = total_entropy[superiority_mask]
        
        # Recording the action played
        self.action_played = best_action

        # Converting action indexes to movement vectors
        movemement_vector = self.model.movement_vector[best_action,:]

        return movemement_vector


    def update_state(self,
                     observation:int|np.ndarray,
                     source_reached:bool|np.ndarray
                     ) -> None:
        '''
        Function to update the internal state(s) of the agent(s) based on the previous action(s) taken and the observation(s) received.

        Parameters
        ----------
        observation : np.ndarray
            The observation(s) the agent(s) made.
        source_reached : np.ndarray
            A boolean array of whether the agent(s) have reached the source or not.
        '''
        assert self.beliefs is not None, "Agent was not initialized yet, run the initialize_state function first"

        # Binarize observations
        observation_ids = np.where(observation > self.treshold, 1, 0).astype(int)
        observation_ids[source_reached] = 2 # Observe source

        # Update the set of beliefs
        self.beliefs = self.beliefs.update(actions=self.action_played, observations=observation_ids)

        # Remove the beliefs of the agents having reached the source
        self.beliefs = BeliefSet(self.model, self.beliefs.belief_array[~source_reached])


    def kill(self,
             simulations_to_kill:np.ndarray
             ) -> None:
        '''
        Function to kill any simulations that have not reached the source but can't continue further

        Parameters
        ----------
        simulations_to_kill : np.ndarray
            A boolean array of the simulations to kill.
        '''
        if all(simulations_to_kill):
            self.beliefs = None
        else:
            self.beliefs = BeliefSet(self.beliefs.model, self.beliefs.belief_array[~simulations_to_kill])