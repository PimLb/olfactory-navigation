import numpy as np
from scipy.special import softmax
import sys
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import argparse as ap

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

def isEnd(sm):
    s = sm % SC
    r, c = s // cols, s % cols
    return (r - rSource) ** 2 + (c -cSource) **2 < find_range**2


def get_observation(sm, pObs):
    s = sm % SC
    return np.random.choice(2, p=pObs[:, s])

def choose_action(pi, o, M, curMem):
    return np.random.choice(4 * M, p = pi[o, curMem])

def takeAction(sm, am):
    s = sm % SC
    a = am % 4
    newM = am // 4
    r, c = s // cols, s % cols # forse usare unravel_index
    action = ActionDict[a]
    rNew = r + action[0]
    cNew = c + action[1]
    r = rNew if rNew >= 0 and rNew < 131 else r
    c = cNew if cNew >= 0 and cNew < 92 else c
    return r * 92 + c + newM * SC, newM

def move(states, actions):
    ret = states + actions
    ret[:, 0] = np.clip(ret[:, 0], 0, rows -1)
    ret[:, 1] = np.clip(ret[:, 1], 0, cols -1)
    return ret

def getTrajectories(start, pObs, max_MC_steps, pi, M):
    num = start.shape[0]
    mapObs = [pObs for i in range(num)]
    step = start.astype(int)
    curMem = np.zeros(num,int)
    done = np.array([a for a in map(isEnd, step)])
    obs = np.array([a for a in map(get_observation, step, mapObs)])
    t = 0
    stepsDone = np.zeros(num)
    while np.any(~done) and t < max_MC_steps:
        for i in range(num): # TODO: sequenziale, sarebbe da parallelizare
            if not done[i]:
                a = choose_action(pi, obs[i], M, curMem[i])
                step[i], curMem[i] = takeAction(step[i], a)
        t+= 1
        obs = np.array([a for a in map(get_observation, step, mapObs)])
        stepsDone[~done] += 1
        done = np.array([a for a in map(isEnd, step)])
    return stepsDone

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("theta_path", help="the path to the theta values")
    parser.add_argument("obs_path",  help="the path to the obsservation probability file")
    parser.add_argument("memories", type=int, help="The memories of the FSC")
    parser.add_argument("name", help="the name of the file to save into")
    args = parser.parse_args()
    thetaPath = args.theta_path
    dataFile = args.obs_path
    M = args.memories
    thetaName = args.name
    
    theta = np.load(thetaPath)
    assert theta.shape == (2, M, 4 * M)
    pi = softmax(theta, axis = 2)

    print("PI to be evaluated: ", pi, flush= True)
    dataC = np.load(dataFile)
    rho = np.zeros(SC * M)
    rho[:cols] = (1-dataC[0,:cols])/np.sum((1-dataC[0,:cols]))
    results = np.zeros(traj)
    print("Inizio: ", time.ctime(), flush=True)
    starts = np.random.choice(range(SC * M), size=traj, replace=True, p = rho)
    # starts = np.array(np.unravel_index(tmp, (131, 92))).T
    # print("Starting points at iteration 0", starts, flush=True)

    N = traj // procNumber
    remainder = traj % procNumber
    print("prima: ", pi[0, 0].shape)
    args = [(starts[i * procNumber: (i+1)* procNumber], dataC, maxStep, pi, M) for i in range(N)]
    with mp.Pool(procNumber) as p:
        rewardList = p.starmap(getTrajectories, args)
    print(time.ctime())
    for i, rl in enumerate(rewardList):
        results[procNumber*i:procNumber*(i+1)] += rl
    # results = getTrajectories(starts, dataC, 10000, pi)
    np.save(f"results/{thetaName}", results)
    # print([b * 200 for b in range(51)])
    n, b, patch = plt.hist(results, 50, range = (0, maxStep))
    patch[-1].set_facecolor('red')
    plt.ylim(0, 5000)
    finished = results[results != maxStep]
    plt.yticks([i*500 for i in range(0, 11)] + [np.count_nonzero(results == maxStep)])
    plt.title(thetaName)
    plt.savefig(f"pngs/{thetaName}.png")
    print("Mean Reached: ", np.mean(finished )," STD Reached: ", np.std(finished ), " Finished", np.count_nonzero(results != maxStep) / traj * 100, "%")
    print("Mean Overall: ", np.mean(results )," STD Overall: ", np.std(results) , "Not finished", np.count_nonzero(results == maxStep) / traj * 100, "%")
