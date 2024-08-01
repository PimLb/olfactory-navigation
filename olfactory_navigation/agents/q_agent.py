import os
from olfactory_navigation.environment import Environment
from olfactory_navigation.agent import Agent

from math import sqrt
from tqdm import tqdm
import json

from typing import Tuple

import numpy as np
gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')

class QAgent(Agent):
    def __init__(self, environment: Environment, 
                 threshold: float  = 0.000003, 
                 horizon : int = 100,
                 num_episodes : int = 1000,
                 learning_rate  = lambda t : 1/sqrt(t + 1),
                 eps_greedy = lambda t : 0.3,
                 gamma : float = 1.0,
                 delta : int = 100,
                 seed : int = 121314,
                 checkpoint_folder : str | None = None,
                 checkpoint_frequency : int | None = None
                 ) -> None:
        assert checkpoint_folder is None or (checkpoint_folder is not None and checkpoint_frequency is not None and checkpoint_frequency > 0)
        super().__init__(environment, threshold, "QAgent")
        self.xp = cp if self.on_gpu else np
        self.learning_rate = learning_rate
        self.horizon = horizon
        self.eps_greedy = eps_greedy
        self.gamma = gamma
        self.delta = delta
        self.deterministic = True
        self.MAX_T = environment.timesteps
        self.num_episodes = num_episodes
        self.Q = np.zeros((2, 4))
        self.action_set = self.xp.array([
            [ 0, -1],
            [ 1,  0],
            [ 0,  1],
            [-1,  0],
        ]).reshape(-1, 2)
        self.episode = 0
        self.checkpoint_folder = checkpoint_folder
        self.checkpoint_frequency = checkpoint_frequency
        if checkpoint_folder is not None:
            os.makedirs(checkpoint_folder, exist_ok=True)        


    def initialize_state(self, n:int=1) -> None:
        self.current_state = self.xp.zeros((n, ), dtype=np.int32)


    def to_gpu(self) -> Agent:
        cls = self.__class__
        gpu_agent = cls.__new__(cls)

        # Copying arguments to gpu
        for arg, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                setattr(gpu_agent, arg, cp.array(val))
            else:
                setattr(gpu_agent, arg, val)

        # Self reference instances
        self._alternate_version = gpu_agent
        gpu_agent._alternate_version = self

        gpu_agent.on_gpu = True
        return gpu_agent




    def update_state(self,
                     observation : float|np.ndarray,
                     source_reached : bool|np.ndarray
                     ) -> None:
        filtered_observations = (observation >= self.threshold).astype(bool)#.reshape(self.memory.shape[0]) # type: ignore
        self.current_state[filtered_observations] += 1
        self.current_state[~filtered_observations] = 0
        if self.deterministic:
            self.current_state = self.current_state[~source_reached]

    def _update_q(self, s, a, s_prime, r):
        alpha = self.learning_rate(self.episode)
        self.Q[s, a] = (1 - alpha) * self.Q[s, a] + alpha * (r + self.gamma * self.Q[s_prime].max())



    def choose_action(self) -> np.ndarray:
        if self.deterministic:
            return self.action_set[self.Q[self.current_state, :].argmax(axis=1)]
        eps_mask = (self.rnd_state.rand(self.memory.shape[0]) < self.eps_greedy(self.episode) ).astype(bool)    
        if np.any(eps_mask):
            action_indices = self.xp.zeros((self.memory.shape[0], ), dtype=np.int32)
            action_indices[eps_mask] = self.rnd_state.randint(0, 4, eps_mask.sum())
            action_indices[~eps_mask] = self.Q[self.current_state[~eps_mask], :].argmax(axis=1)
            return self.action_set[action_indices]

        return self.action_set[self.Q[self.current_state, :].argmax(axis=1)]
    
    def _get_agent_state(self):
        return dict(threshold=self.threshold, 
                    horizon = self.horizon, 
                    gamma=self.gamma,
                    episode = self.episode,
                    num_episodes = self.num_episodes)

    def _save_checkpoint(self):
        np.save(f"{self.checkpoint_folder}/QFunction_{self.episode}", self.Q)
        with open(f"{self.checkpoint_folder}/agent_state_{self.episode}.json", 'w') as f:
            json.dump(self._get_agent_state(), f)

    def _perform_single_step(self, pos : np.ndarray, time_idx : int):
        if self.current_state[0] == 0:
            a = 0
        elif self.rnd_state.rand() < self.eps_greedy(self.episode):
            a = self.rnd_state.choice(4, size=1)[0]
        else: 
            a = self.Q[self.current_state[0], :].argmax()
        new_pos = self.environment.move(pos, self.action_set[a])
        terminated = self.environment.source_reached(new_pos)
        r = 1.0 if terminated else 0.0
        time_idx = (time_idx + 1) % self.MAX_T
        obs = self.environment.get_observation(new_pos, time_idx)
        self.update_state(obs, terminated)
        s_prime = self.current_state[0]
        if s_prime >= self.Q.shape[0]:
            self.Q = np.vstack((self.Q, np.zeros(4, dtype=np.float32)))

        return new_pos, a, r, terminated, s_prime, time_idx




    def save(self, folder: str | None = None, force: bool = False) -> None:
        np.save(f"{folder}/QFunction", self.Q)
        with open(f"{folder}/agent_state.json", 'w') as f:
            json.dump(self._get_agent_state(), f)


    def load(self, folder:str):
        self.Q = np.load(f"{folder}/QFunction.npy")
        if os.path.isfile(f"{folder}/agent_state.json"):
            with open(f"{folder}/agent_state.json", 'r') as f:
                agent_state = json.load(f)
            for (k, v) in agent_state.items():
                setattr(self, k, v)


    def train(self):
        cumulative_rewards = []
        average_crewards = []
        iterator = tqdm(range(self.num_episodes))
        self.deterministic = False
        for _ in iterator:
            # Initialization
            self.initialize_state(1)
            
            init_time_idx = self.rnd_state.randint(0, self.environment.timesteps)
            time_idx = init_time_idx
            init_pos = self.environment.random_start_points(1)
            pos = init_pos.copy()
            obs = self.environment.get_observation(pos, time_idx)
            self.update_state(obs, source_reached=self.environment.source_reached(pos))
            c_reward = 0.0
            s = self.current_state[0]
            alpha = self.learning_rate(self.episode)
            for t in range(self.horizon):
                pos, a, r, terminated, s_prime, time_idx = self._perform_single_step(pos, time_idx)
                c_reward += r * (self.gamma**t) if t > 0 else r        
                self.Q[s, a] = (1 - alpha) * self.Q[s, a] + alpha * (r + self.gamma * self.Q[s_prime].max())
                if terminated:
                    break
                s = s_prime
            self.episode += 1
            if self.checkpoint_folder is not None and self.episode % self.checkpoint_frequency == 0: # type: ignore
                self._save_checkpoint()
            cumulative_rewards.append(c_reward)
            iterator.set_postfix({
                                'episode' : self.episode,
                                'init_pos' : init_pos.flatten(),
                                'init time slice' : init_time_idx,
                                'avg R_t' : np.mean(cumulative_rewards[-self.delta:]), 
                                'eps' : f'{self.eps_greedy(self.episode)}',
                                'alpha' : f'{self.learning_rate(self.episode )}'})
            average_crewards.append(np.mean(cumulative_rewards[-self.delta:]))
        self.deterministic = True
        return {'average_cumulative_reward' : np.array(average_crewards), 'cumulative_rewards' : np.array(cumulative_rewards)}

    def kill(self, simulations_to_kill: np.ndarray) -> None:
        self.memory = self.memory[~simulations_to_kill]
        self.current_state = self.current_state[~simulations_to_kill]
#        self.current_memory_len = self.current_memory_len[~simulations_to_kill]
