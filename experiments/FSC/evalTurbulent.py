import numpy as np
from scipy.special import softmax
import sys
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import argparse as ap
import h5py
import os

gamma = 0.99975
ActionDict = np.asarray([
            [-1,  0], # Downwind
            [ 0,  1], # Crosswind
            [ 1,  0], # Upwind
            [ 0, -1]  # Crosswind
        ])
reward = -(1 -gamma)
procNumber = 50

odor = h5py.File("storage/odor_data_r800/odor_new.h5")['odor']
rows, cols = odor['0'].shape
SC = rows * cols

traj = 10000
maxStep = 10000
threshold = 1e-4 #TODO: how to choose the threshold?
rho = np.zeros(SC) # TODO: where to put the starting points? How to weight them?
rho[:cols] = (odor['0'][0])/np.sum((odor['0'][0]))

rSource = 950 
cSource = 125
find_range = 1.1 # TODO: are these correct?

def isEnd(sm):
    s = sm % SC
    r, c = s // cols, s % cols
    return (r - rSource) ** 2 + (c -cSource) **2 < find_range**2

def get_observation(sm, frame, threshold = 1e-4):
    s = sm % SC
    r,c = s // cols, s % cols
    return int(odor[str(frame)][r,c] >= threshold)

def choose_action(pi, o, M, curMem):
    return np.random.choice(4 * M, p = pi[o, curMem])

def takeAction(sm, am):
    s = sm % SC
    a = am % 4
    newM = am // 4
    r, c = s // cols, s % cols # forse usare unravel_index
    action = ActionDict[a]
    r = np.clip(r + action[0], 0, rows-1)
    c = np.clip(c + action[1], 0, cols-1)
    return r * cols + c + newM * SC, newM

def getTrajectories(start, max_MC_steps, pi, M, threshold = 1e-4):
    num = start.shape[0]
    curStates = start.astype(int)
    curMem = np.zeros(num,int)
    done = np.array([a for a in map(isEnd, curStates)])
    # obs = np.array([a for a in map(get_observation, step, [0 for i in range(num)])])
    t = 0
    stepsDone = np.zeros(num)
    lt = [threshold for i in range(num)]
    while np.any(~done) and t < max_MC_steps:
        obs = np.array([a for a in map(get_observation, curStates, [t for i in range(num)], lt)])
        for i in range(num): # TODO: sequenziale, sarebbe da parallelizare
            if not done[i]:
                a = choose_action(pi, obs[i], M, curMem[i])
                curStates[i], curMem[i] = takeAction(curStates[i], a)
        t+= 1
        stepsDone[~done] += 1
        done = np.array([a for a in map(isEnd, curStates)])
    return stepsDone

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("name", help="the name of the file to save into")
    parser.add_argument("theta_path", help="the path to the theta values")
    parser.add_argument("--threshold", help="The threshold abov which the agent will receive a positve observation. Default 1e-4", default=1e-4)
    args = parser.parse_args()
    thetaPath = args.theta_path
    thetaName = args.name
    threshold = args.threshold

    saveDir = f"eval/turbulent/{thetaName}_{threshold}"

    os.makedirs(saveDir, exist_ok=True)
    for s in range(SC):
        if rho[s] > 0:
            print(s // cols, s % cols)
    # sys.exit()
    theta = np.load(thetaPath)
    M = theta.shape[1]
    assert theta.shape == (2, M, 4 * M)
    pi = softmax(theta, axis = 2)
    print("PI to be evaluated: ", pi, flush= True)
    results = np.zeros(traj)
    print("Inizio: ", time.ctime(), flush=True)
    starts = np.random.choice(range(SC), size=traj, replace=True, p = rho)

    N = traj // procNumber
    remainder = traj % procNumber
    print("prima: ", pi[0, 0].shape)
    args = [(starts[i * procNumber: (i+1)* procNumber], maxStep, pi, M) for i in range(N)]
    with mp.Pool(procNumber) as p:
        rewardList = p.starmap(getTrajectories, args)
    print(time.ctime())
    for i, rl in enumerate(rewardList):
        results[procNumber*i:procNumber*(i+1)] += rl
    np.save(f"{saveDir}/res.npy", results)
    n, b, patch = plt.hist(results, 50, range = (0, maxStep))
    patch[-1].set_facecolor('red')
    plt.ylim(0, 5000)
    finished = results[results != maxStep]
    plt.yticks([i*500 for i in range(0, 11)] + [np.count_nonzero(results == maxStep)])
    plt.title(thetaName)

    plt.savefig(f"{saveDir}/res.png")
    print("Mean Reached: ", np.mean(finished )," STD Reached: ", np.std(finished ), " Finished", np.count_nonzero(results != maxStep) / traj * 100, "%")
    print("Mean Overall: ", np.mean(results )," STD Overall: ", np.std(results) , "Not finished", np.count_nonzero(results == maxStep) / traj * 100, "%")

    