
import sys
sys.path.append('..')
import time
from olfactory_navigation import Environment
from olfactory_navigation.agents import QMemAgent
from olfactory_navigation.simulation import run_test
from olfactory_navigation.test_setups import run_all_starts_test
from olfactory_navigation.agents.qagent_util import PositionSampler
from typing import Tuple

from matplotlib import pyplot as plt
#%matplotlib inline

from concurrent.futures import ProcessPoolExecutor as PPE

import pandas as pd
import numpy as np
import itertools

horizon = 1000
num_episodes = 1000000
#eps_decay = 100000
rewards = (1.0, -0.001)
eps_init, eps_end = 1.0, 0.01#e-5
seed = 12131415
noise_thr = 3e-6
margins = [14, 62]
out_dir = "./q_agent_training2"


env = Environment(data_file="/storage/rando/data/nose_data_27_123.npy",
                  data_source_position=[13, 0],
                  source_radius=2,
                  margins=margins,
                  seed=seed,
                  boundary_condition='wrap_vertical',
                  start_zone='odor_present',
                  odor_present_threshold=noise_thr)
level = 0
rnd_state = np.random.RandomState(121314)

sampling_regions = [
#    np.array([[17, 72], [37, 82]], dtype=np.int32),
    np.array([[0, env.padded_height], [55, env.padded_width]], dtype=np.int32),
    ]
#for i in range(15):
#    sampling_regions.append(np.array([[sampling_regions[-1][0][0], sampling_regions[-1][0][1] + 10], [sampling_regions[-1][1][0], sampling_regions[-1][1][1] + 10]]) )

#sampling_regions.append(np.array([[0, 37], [env.padded_height, env.padded_width]]))

#print(sampling_regions)
#print(env.padded_height, env.padded_width)
#exit()


def run_agent(params : Tuple):
    eps_decay, mem_size = params
#    position_sampler = PositionSampler(sampling_regions, delta = 5, quality_threshold=0.9, minimum_quality=-0.1, seed = 12131415)
    path = "{}/memory_size_{}/decay_{}".format(out_dir,mem_size, eps_decay)
    ag = QMemAgent(
        environment=env,
        horizon=horizon, 
        memory_size=mem_size,
        eps_init=eps_init,
        eps_end=eps_end,
        eps_decay=eps_decay,
        alpha_init=0.9,
        alpha_decay=eps_decay,
        alpha_end= 0.01,
#        learning_rate=lambda t : 1e-5 + (0.9 - 1e-5) * np.exp(- t / eps_decay),
        gamma=1.0,
        delta=500,
        threshold=3e-6,
        rewards=rewards,
        num_episodes=num_episodes,
        deterministic=True,
        checkpoint_folder  = f"{path}/checkpoints",
        checkpoint_frequency  = 50000
    )
    tr_time = time.time()
    training_result = ag.train(set_best_Q=False, draw_Q=False)
    tr_time = time.time() - tr_time
    ag.save(f"{path}/")
    fig, ax = plt.subplots()
    ax.plot(range(len(training_result['average_cumulative_reward'])), training_result['average_cumulative_reward'], '-')
    fig.savefig(f"{path}/avg_crew.pdf")
    plt.close(fig)

    with open(f"{path}/tr_info.log", 'w') as f:
        f.write(f"decay: {eps_decay}\n")
        f.write(f"episodes: {num_episodes}\n")
        f.write(f"horizon: {horizon}\n")
        f.write(f"rewards: {rewards}\n")
        f.write(f"training time: {tr_time}\n")
    return training_result


    
    
    

max_workers = 15

eps_decays = [5000, 100000, 150000]
memory_sizes = [1, 5, 10, 15, 20]
results = []
grid = list(itertools.product(eps_decays, memory_sizes))
with PPE(max_workers=max_workers) as executor:
    for tr_result in executor.map(run_agent, grid, chunksize=len(grid) // max_workers):
        print("[--] Completed!")
        results.append(tr_result)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
ax1.set_title("Average Cumulative Reward", fontsize=18)
ax2.set_title("Average Speed", fontsize=18)
for i in range(len(results)):
    avg_crew = results[i]['average_cumulative_reward']
    avg_speed = results[i]['avg_speed']
    mem_size, eps_decay = results[i]['memory_size'], results[i]['eps_decay']
    ax1.plot(range(avg_crew.shape[0]), avg_crew, '-', label='$M = {}$ and $\\epsilon = {}$'.format(mem_size, eps_decay))    
    ax2.plot(range(avg_speed.shape[0]), avg_speed, '-', label='$M = {}$ and $\\epsilon = {}$'.format(mem_size, eps_decay))
ax1.set_xlabel("episode", fontsize=14)
ax2.set_xlabel("episode", fontsize=14)
ax1.legend()
ax2.legend()

fig.savefig(f"{out_dir}/result_comparison.pdf", bbox_inches='tight')



