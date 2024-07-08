import os
from ..environment import Environment
from ..agent import Agent

from math import sqrt, exp
from tqdm import tqdm
import json
import time
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython import display
import numpy as np
gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')



class QMemAgent(Agent):
    def __init__(self, environment: Environment, 
                 threshold: float  = 3e-6, 
                 horizon : int = 100,
                 num_episodes : int = 1000,
                 learning_rate  = lambda t : 1/sqrt(t + 1),
                 eps_init = 0.8,
                 eps_end = 0.2,
                 eps_decay = 10000,
                 memory_size = 1,
                 deterministic : bool = True,
                 gamma : float = 1.0,
                 delta : int = 100,
                 seed : int = 121314,
                 rewards : Tuple[float, float] = (1.0, 0.0),
                 checkpoint_folder : str | None = None,
                 checkpoint_frequency : int | None = None
                 ) -> None:
        assert memory_size >= 1, "Size of memory must be >= 1!"
        assert checkpoint_folder is None or (checkpoint_folder is not None and checkpoint_frequency is not None and checkpoint_frequency > 0)
        super().__init__(environment, threshold, "QAgent")
        self.xp = cp if self.on_gpu else np
        self.learning_rate = learning_rate
        self.horizon = horizon
        self.seed = seed
        self.rnd_state = np.random.RandomState(seed = seed)
        self.eps_init = eps_init
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.memory_size = memory_size
        self.gamma = gamma
        self.delta = delta
        self.rewards = rewards
        self.deterministic = deterministic
        self.MAX_T = environment.data.shape[0] 
        self.num_episodes = num_episodes
        self.Q = np.zeros((2, 4))
        self.action_set = self.xp.array([
            [ 0, -1],
            [ 1,  0],
            [-1,  0],
            [ 0,  1],
        ]).reshape(-1, 2)
        self.action_labels = ['left', 'down', 'up', 'right']
        self.episode = 0
        self.checkpoint_folder = checkpoint_folder
        self.checkpoint_frequency = checkpoint_frequency
        if checkpoint_folder is not None:
            os.makedirs(checkpoint_folder, exist_ok=True)        


    def initialize_state(self, n:int=1) -> None:
        self.current_state = self.xp.zeros((n, ), dtype=np.int32)
        self.memory = self.xp.zeros((n, self.memory_size), dtype=np.int32)


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
        filtered_observations = (observation >= self.threshold).astype(bool).reshape(-1) # type: ignore
        self.memory = np.concatenate((self.memory, filtered_observations.reshape(-1, 1)), axis=1)
        observed_mask = np.any(self.memory, axis=1)
        self.current_state[observed_mask] = 0
        self.current_state[~observed_mask] += 1
        self.current_state = self.current_state[~source_reached]
        self.memory = self.memory[~source_reached, :]

    def choose_action(self) -> np.ndarray:
        if self.deterministic:
            return self.action_set[self.Q[self.current_state, :].argmax(axis=1)]
        
        eps_mask = (self.rnd_state.rand(self.current_state.shape[0]) < self.eps_end ).astype(bool)    
        if np.any(eps_mask):
            action_indices = self.xp.zeros((self.current_state.shape[0], ), dtype=np.int32)
            action_indices[eps_mask] = self.rnd_state.randint(0, 4, eps_mask.sum())
            action_indices[~eps_mask] = self.Q[self.current_state[~eps_mask], :].argmax(axis=1)
            return self.action_set[action_indices]

        return self.action_set[self.Q[self.current_state, :].argmax(axis=1)]
    
    def _get_agent_state(self):
        return dict(threshold=self.threshold, 
                    horizon = self.horizon, 
                    gamma=self.gamma,
                    memory_size = self.memory_size,
                    episode = self.episode,
                    num_episodes = self.num_episodes)

    def _save_checkpoint(self):
        np.save(f"{self.checkpoint_folder}/QFunction_{self.episode}", self.Q)
        with open(f"{self.checkpoint_folder}/agent_state_{self.episode}.json", 'w') as f:
            json.dump(self._get_agent_state(), f)


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


    def train(self, set_best_Q = False, draw_Q = False, draw_iters = 1, delta_Q_draw = 50, delta_draw = 10):#, init_sampling_region):
        cumulative_rewards = []
        average_crewards = []
        avg_max_void_steps = []
        max_void_steps = []
        episode_times = []
        speed = []
        avg_speed = []
        best_Q, best_avg = self.Q.copy(), -np.inf
        iterator = tqdm(range(self.num_episodes))
        if draw_Q:
            fig = plt.figure(figsize=(10, 8))
            fig.suptitle(f"Q-Agent Training (Memory Size = {self.memory_size})", fontsize=20)
            ax1 = fig.add_subplot(2, 3, 1)
            ax2 = fig.add_subplot(2, 3, 2)
            ax4 = fig.add_subplot(2, 3, 3)
            ax3 = fig.add_subplot(2, 1, 2)
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            dh = display.display(fig, display_id=True)

            ax3.set_xlabel("time since observation")
            ax1.set_title(f"Average Cumulative Reward [$\\Delta = {self.delta}$]", fontsize=10)
            ax2.set_title(f"Average Maximum Steps in Void [$\\Delta = {self.delta}$]", fontsize=10)
            ax4.set_title(f"Average T_min / T [$\\Delta = {self.delta}$]", fontsize=10)
            ax3.set_title(f"Normalized Q-Function", fontsize=10)
            p1, = ax1.plot(average_crewards[-delta_draw:], '-', lw=1, c='darkgreen')
            p2, = ax2.plot(avg_max_void_steps[-delta_draw:], '-', lw=1, c='darkred')
            p3, = ax4.plot(avg_speed[-delta_draw:], '-', lw=1, c='orange')
            delta = min(delta_Q_draw, self.Q.shape[0])
            im = ax3.imshow((self.Q.T[:, :delta] - self.Q.T[:, :delta].min(axis=0)) / (self.Q.T[:, :delta].max(axis=0) - self.Q.T[:, :delta].min(axis=0)), origin='lower', interpolation='bilinear', vmin=0.0, vmax=1.0)
            fig.colorbar(im, cax=cax, orientation='vertical')

            axes = [ax1, ax2, ax3, ax4]
            fig.tight_layout()

        for episode in iterator:
            # Initialization
            self.initialize_state(1)            
            init_time_idx = self.rnd_state.randint(0, self.environment.data.shape[0])
            time_idx = init_time_idx
            init_pos = self.environment.random_start_points(1) #self.rnd_state.randint(sampling_region[:, 0], sampling_region[:, 1])
            pos = init_pos.copy()
            obs = self.environment.get_observation(pos, time_idx)
            self.memory = self.memory[0, :]
            self.memory[-1] = int(obs >= self.threshold)
            s = 0 if obs >= self.threshold else 1 # current state (0 if odor observed, 1 otherwise)
            c_reward = 0.0 # cumulative reward in episode
            alpha = self.learning_rate(self.episode) 
            eps = max(self.eps_end + (self.eps_init - self.eps_end) * exp(- episode / self.eps_decay), self.eps_end)
            current_max_void = s
            episode_time = time.time()
            for t in range(self.horizon):
                # compute and play action
                if s == 0:
                    a = 0
                else:
                    a = self.rnd_state.choice(self.action_set.shape[0]) if self.rnd_state.rand() <= eps else np.argmax(self.Q[s, :])
                new_pos = self.environment.move(pos, self.action_set[a])
                terminated = self.environment.source_reached(new_pos)
                # receive reward and update Q
                r = self.rewards[0] if terminated else self.rewards[1]
                time_idx = (time_idx + 1) % self.MAX_T
                obs = self.environment.get_observation(new_pos, time_idx)
                self.memory = np.concatenate((self.memory[1:], [obs[0] >= self.threshold])) 
                s_prime = 0 if np.any(self.memory) else s + 1
                if terminated:
                    s_prime = None
                if s_prime is not None and s_prime > current_max_void:
                    current_max_void = s_prime
                if s_prime is not None and s_prime >= self.Q.shape[0]:
                    self.Q = np.vstack((self.Q, np.zeros(4, dtype=np.float32)))
                c_reward += r * (self.gamma**t) if t > 0 else r        
                if s_prime is None:
                    self.Q[s, a] = (1 - alpha) * self.Q[s, a] + alpha * r
                else:
                    self.Q[s, a] = (1 - alpha) * self.Q[s, a] + alpha * (r + self.gamma * self.Q[s_prime].max())
                if terminated:
                    break
                s = s_prime
                pos = new_pos
            episode_speed = self.environment.distance_to_source(init_pos)[0] / (t + 1) if terminated else 0.0
            episode_time = time.time() - episode_time
            episode_times.append(episode_time)
            self.episode += 1
            if self.checkpoint_folder is not None and self.episode % self.checkpoint_frequency == 0: # type: ignore
                self._save_checkpoint()
            cumulative_rewards.append(c_reward)
            average_crewards.append(np.mean(cumulative_rewards[-self.delta:]))
            max_void_steps.append(current_max_void)
            avg_max_void_steps.append(np.mean(max_void_steps[-self.delta:]))
            speed.append(episode_speed)
            avg_speed.append(np.mean(speed[-self.delta:]))
            iterator.set_postfix({
                                'episode' : self.episode,
                                'init_pos' : init_pos.flatten(),
                                'init time slice' : init_time_idx,
                                'avg R_t' : average_crewards[-1], 
                                'avg void' : avg_max_void_steps[-1],
                                'eps' : f'{eps}',
                                'alpha' : f'{self.learning_rate(self.episode )}'})
            if draw_Q and episode % draw_iters == 0 and episode > 0:
                ax3.clear()
                delta = min(delta_Q_draw, self.Q.shape[0])
                ax1.set_title(f"Average Cumulative Reward [$\\Delta = {self.delta}$]", fontsize=10)
                ax2.set_title(f"Average Maximum Steps in Void [$\\Delta = {self.delta}$]", fontsize=10)
                ax3.set_title(f"Normalized Q-Function", fontsize=10)
                p1.set_data(range(min(delta_draw, len(average_crewards[-delta_draw:]))), average_crewards[-delta_draw:])
                p2.set_data(range(min(delta_draw, len(avg_max_void_steps[-delta_draw:]))), avg_max_void_steps[-delta_draw:])
                p3.set_data(range(min(delta_draw, len(avg_speed[-delta_draw:]))), avg_speed[-delta_draw:])
                delta = min(delta_Q_draw, self.Q.shape[0])
                im = ax3.imshow((self.Q.T[:, :delta] - self.Q.T[:, :delta].min(axis=0)) / (self.Q.T[:, :delta].max(axis=0) - self.Q.T[:, :delta].min(axis=0)), origin='lower', interpolation='bilinear', vmin=0.0, vmax=1.0)
                for ax in axes:
                    ax.relim()
                    ax.autoscale_view()
    
                ax3.set_xticks(range(delta))
                ax3.set_yticks(range(4), labels=self.action_labels)
                dh.update(fig)

            if best_avg < average_crewards[-1] and episode >= self.delta:
                best_Q = self.Q.copy()
                best_avg = average_crewards[-1]
        if set_best_Q:
            self.Q = best_Q
        return {'average_cumulative_reward' : np.array(average_crewards), 
                'cumulative_rewards' : np.array(cumulative_rewards), 
                'best_Q' : best_Q,
                'memory_size' : self.memory_size, 
                'eps_decay' : self.eps_decay,
                'avg_speed' : np.array(avg_speed),
                'episode_times' : np.array(episode_times),
                'average_max_void_steps' : np.array(avg_max_void_steps)}

    def kill(self, simulations_to_kill: np.ndarray) -> None:
        self.current_state = self.current_state[~simulations_to_kill]
        self.memory = self.memory[~simulations_to_kill, :]

