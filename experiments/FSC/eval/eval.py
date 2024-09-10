import numpy as np
from scipy.special import softmax
import sys
import time
import multiprocessing as mp
import matplotlib.pyplot as plt

cSource = 45.5
rSource = 91
find_range = 1.1
gamma = 0.99975
cols = 92
rows = 131
traj = 10000
maxStep = 10000
SC = 92 * 131 # Number of States
ActionDict = np.asarray([
            [-1,  0], # North
            [ 0,  1], # East
            [ 1,  0], # South
            [ 0, -1]  # West
        ])
reward = -(1 -gamma)
procNumber = 50

def isEnd(s):
    return (s[0] - rSource) ** 2 + (s[1] -cSource) **2 < find_range**2

def get_observation(s, pObs):
    singleD = s[0] * cols + s[1]
    return np.random.choice(2, p=pObs[:, singleD])

def choose_action(pi, o):
    return np.random.choice(4, p = pi[o, 0])

def move(states, actions):
    ret = states + actions
    ret[:, 0] = np.clip(ret[:, 0], 0, rows -1)
    ret[:, 1] = np.clip(ret[:, 1], 0, cols -1)
    return ret

def getTrajectories(start, pObs, max_MC_steps, pi):
    num = start.shape[0]
    mapObs = [pObs for i in range(num)]
    step = start.astype(int)
    done = np.array([a for a in map(isEnd, step)])
    obs = np.array([a for a in map(get_observation, step, mapObs)])
    t = 0
    actions = np.zeros((num, 2), dtype=int)
    rewards = np.zeros(num)
    while np.any(~done) and t < max_MC_steps:
        for i in range(num): # TODO: sequenziale, sarebbe da parallelizare
            if not done[i]:
                a = choose_action(pi, obs[i])
                actions[i] = ActionDict[a]
            else:
                actions[i] = np.array([0, 0])
        t+= 1
        step = move(step, actions)
        obs = np.array([a for a in map(get_observation, step, mapObs)])
        done = np.array([a for a in map(isEnd, step)])
        rewards[~done] += reward
    return np.mean(rewards)

if __name__ == "__main__":
    thetaPath = sys.argv[1]
    theta = np.load(thetaPath)
    thetaName = thetaPath[thetaPath.rfind("/")+1:]
    pi = softmax(theta, axis = 2)
    print("PI to be evaluated: ", pi, flush= True)
    dataFile = sys.argv[2]
    dataC = np.load(f"../celaniData/{dataFile}.npy")
    rho = np.zeros(SC)
    rho[:cols] = (1-dataC[0,:cols])/np.sum((1-dataC[0,:cols]))
    results = []
    print("Inizio: ", time.ctime(), flush=True)
    tmp = np.random.choice(range(SC), size=traj, replace=True, p = rho)
    starts = np.array(np.unravel_index(tmp, (131, 92))).T
    # print("Starting points at iteration 0", starts, flush=True)

    N = traj // procNumber
    remainder = traj % procNumber
    for i in range(N):
        args = [(starts[i: (i+1)* procNumber], dataC, maxStep, pi) ] * procNumber
        with mp.Pool(procNumber) as p:
            rewardList = p.starmap(getTrajectories, args)
        results += rewardList
        print(f"Fine iterazione {i}/{N}: ", time.ctime(), flush=True)
    print("Mean: ", np.mean(results)," STD: ", np.std(results))
    plt.hist(results, 50)
    plt.savefig(f"{thetaName}.png")
