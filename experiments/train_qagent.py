
import sys
sys.path.append('..')
import time
from olfactory_navigation import Environment
from olfactory_navigation.agents import QAgent
from olfactory_navigation.simulation import run_test
from olfactory_navigation.test_setups import run_all_starts_test

from matplotlib import pyplot as plt
#%matplotlib inline

import pandas as pd
import numpy as np
env = Environment(data_file="/storage/rando/data/nose_data_27_123.npy",
                  data_source_position=[13, 0],
                  source_radius=2,
                  margins=[14, 62],
                  boundary_condition='wrap_vertical',
                  start_zone='odor_present',
                  odor_present_threshold=3e-6)

horizon = 1000
num_episodes = 1000000
eps_decay = 100000
rewards = (1.0, -0.001)
ag = QAgent(
    environment=env,
    horizon=horizon, 
    eps_init=1.0,
    eps_end=1e-5,
    eps_decay=eps_decay,#int(num_episodes - num_episodes//10),
    learning_rate=lambda t : 0.001 + (0.9 - 0.001) * np.exp(- t / eps_decay),
#    learning_rate=lambda t : 1e-5 + (0.9 - 1e-5) * np.exp(- t / eps_decay),
    gamma=1.0,
    threshold=3e-6,
    delta=500,
    rewards=rewards,
    num_episodes=num_episodes,
    deterministic=True,
    checkpoint_folder  = "./q_agent_training2/checkpoints",
    checkpoint_frequency  = 10000
)


tr_time = time.time()
training_result = ag.train(set_best_Q=False, draw_Q=False)
tr_time = time.time() - tr_time
ag.save("./q_agent_training2/")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
ax1.set_title("Average Cumulative Reward")
ax2.set_title("Average Speed")

ax1.plot(range(len(training_result['average_cumulative_reward'])), training_result['average_cumulative_reward'], '-', c='black')
ax2.plot(range(len(training_result['avg_speed'])), training_result['avg_speed'], '-', c='orange')

ax1.set_xlabel("episode")
ax2.set_xlabel("episode")
ax1.set_ylabel("average cumulative reward")
ax2.set_ylabel("average speed")
fig.tight_layout()
fig.savefig("./q_agent_training2/avg_crew.pdf")
plt.close(fig)

with open("./q_agent_training2/tr_info.log", 'w') as f:
    f.write(f"decay: {eps_decay}\n")
    f.write(f"episodes: {num_episodes}\n")
    f.write(f"horizon: {horizon}\n")
    f.write(f"rewards: {rewards}\n")
    f.write(f"training time: {tr_time}\n")

with open("./q_agent_training2/tr_trace.log", 'w') as f:
    for i in range(len(training_result['average_cumulative_reward'])):
        f.write("{},{}\n".format(training_result['average_cumulative_reward'][i],training_result['avg_speed'][i]))
