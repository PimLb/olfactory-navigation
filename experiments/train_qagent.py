import sys
sys.path.append('..')
from olfactory_navigation.agents import QAgent

from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import cupy as cp


from olfactory_navigation import Environment

data_path = "/storage/rando/data/nose_data_27_123.npy" # INSERT YOUR PATH

env = Environment(data_file=data_path,
                  data_source_position=[13, 0],
                  source_radius=2,
                  margins=[14, 62],
                  boundary_condition='stop',
                  start_zone='odor_present',
                  odor_present_treshold=3e-6)


memory_size = 10
time_disc = 1000
horizon = 1000
num_episodes=100000
delta = 500
gamma = 1.0
eps_decay = 0.0001
alpha_decay = 0.0001
eps = lambda t : 0.9 * np.exp(-eps_decay * t)  #if np.exp(-eps_decay * t)  > 0.3 else 0.3
alpha = lambda t : 0.9 * np.exp(-alpha_decay * t) # if np.exp(-alpha_decay * t) > 0.0001 else 0.0001 #0.3 * np.exp(-alpha_decay * t) > 0.001 else 0.001

checkpoint_folder = "./q_agent_training/clip/checkpoints"
checkpoint_frequency = 5000


ag = QAgent(env, 
            memory_size=memory_size, 
            treshold=3e-6,
            time_disc=time_disc, 
            horizon=horizon,
            num_episodes=num_episodes,
            delta=delta,
            eps_greedy= eps,
            learning_rate=alpha,
            gamma = gamma,
            seed=13141516,
            checkpoint_folder=checkpoint_folder,
            checkpoint_frequency=checkpoint_frequency
            )


training_result = ag.train()

average_cumulative_reward = training_result['average_cumulative_reward']

fig, ax = plt.subplots()

ax.plot(range(average_cumulative_reward.shape[0]), average_cumulative_reward, '-', lw=3, c='black')

ax.set_xlabel("episode")
ax.set_ylabel("average cumulative reward")

fig.tight_layout()
fig.savefig("./training_result.pdf", bbox_inches='tight')
plt.close(fig)


ag.save("./")