import numpy as np
from scipy.special import softmax
import sys
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import argparse as ap
import os

cSource = 45.5
rSource = 91
find_range = 1.1
gamma = 0.99975
cols = 92
rows = 131
traj = 10000
cumProbs = None
obsProb = None
maxStep = 10000
dataC = None
SC = 92 * 131 # Number of States
ActionDict = np.asarray([
            [-1,  0], # North
            [ 0,  1], # East
            [ 1,  0], # South
            [ 0, -1]  # West
        ])
reward = -(1 -gamma)
procNumber = 50

def chooseActionsVect(obs, curMems):
    actions = np.arange(4*M)
    CDFs = cumProbs[obs*M+curMems]
    u = np.random.random(obs.shape)[:, None]
    idx = np.argmax(u < CDFs, axis = 1)
    return actions[idx]

def takeActionVect(curStates, actionsMem, unbounded):
    #TODO: count how many times the agent hit the boundaries?
    if unbounded:
        newStates = curStates + ActionDict[actionsMem % 4]
    else:
        newStates = np.clip(curStates + ActionDict[actionsMem % 4], [0,0], [rows-1, cols-1])
    newMem = actionsMem // 4
    return newStates, newMem

def getObsVect(states):
    cdf = np.ones((starts.shape[0], 2))
    rr = states[:, 0]
    cc = states[:, 1]
    valid = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
    u = np.random.random(states.shape[0])[:, None]
    cdf[valid] = obsProb[rr[valid]*cols+cc[valid]]
    
    return np.argmax(u < cdf, axis = 1)

def isEndVect(curStates, finished=False):
    return finished | (np.linalg.norm(curStates-[rSource, cSource], axis=1) < find_range)

def getManyTraj(starts, unbounded):
    minDist = np.linalg.norm(starts-[rSource, cSource], ord=1, axis=1)
    curStates = starts
    curMem = np.zeros(starts.shape[0], dtype=int)
    stepsDone = np.zeros(starts.shape[0], dtype=int)
    Gs = np.zeros(starts.shape[0])
    # Checking if any starting states is an ending states. It will never happen as of now, but maybe a change in the starring condition will make it possible
    done = isEndVect(starts)
    t = 0
    while np.any(~done) and t < maxStep:
        obs = getObsVect(curStates)
        actions = chooseActionsVect(obs, curMem)
        curStates, curMem = takeActionVect(curStates, actions, unbounded)
        stepsDone[~done] += 1
        Gs[~done] += reward * gamma**t
        # curFrame += 1
        t += 1
        done = isEndVect(curStates, done)
    return minDist / stepsDone, Gs, done, stepsDone


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("name", help="the name of the file to save into")
    parser.add_argument("obs_path",  help="the path to the obsservation probability file")
    parser.add_argument("theta_path", help="the path to the theta values")
    parser.add_argument("-f", "--subfolder", help="save the results into another folder")
    parser.add_argument("-u", "--unbounded", help="whether to use an infinite domain", action="store_true", default=False)
    args = parser.parse_args()
    thetaPath = args.theta_path
    dataFile = args.obs_path
    thetaName = args.name
    unbounded = args.unbounded
    sb = args.subfolder


    if sb is None:
        saveDir = f"eval/likelihood/{thetaName}{"_unbounded" if unbounded else ""}"
    else:
        saveDir = f"eval/likelihood/{sb}/{thetaName}{"_unbounded" if unbounded else ""}"

    os.makedirs(saveDir, exist_ok=True)
    theta = np.load(thetaPath)
    M = theta.shape[1]
    assert theta.shape == (2, M, 4 * M)
    pi = softmax(theta, axis = 2)
    cumProbs = np.cumsum(pi.reshape(-1, 4*M), axis = 1)

    print("PI to be evaluated: ", pi, flush= True)
    dataC = np.load(dataFile)
    obsProb = np.cumsum(dataC.T, axis = 1)
    rho = np.zeros(SC * M)
    rho[:cols] = (1-dataC[0,:cols])/np.sum((1-dataC[0,:cols]))
    results = np.zeros(traj)
    print("Inizio: ", time.ctime(), flush=True)

    startsCols = np.random.choice(range(cols), size=traj, replace=True, p = rho[:cols])
    starts = np.array([(0, cols) for cols in startsCols])
    timeToReach, empiricalG, successes, steps = getManyTraj(starts, unbounded)
    results = np.stack((timeToReach, empiricalG, successes, steps))
    np.save(f"{saveDir}/res.npy", results)

    plt.boxplot((timeToReach[successes],empiricalG), tick_labels = ["t", r"$\hat{G}$"])
    plt.title(thetaName)

    plt.savefig(f"{saveDir}/res.svg")
    plt.close()

    n, b, patch = plt.hist(steps, 50, range = (0, maxStep))
    patch[-1].set_facecolor('red')
    plt.ylim(0, 5000)
    finished = steps[steps != maxStep]
    plt.yticks([i*500 for i in range(0, 11)] + [np.count_nonzero(steps == maxStep)])
    plt.title(thetaName)
    plt.savefig(f"{saveDir}/hist.svg")
    print("Mean Reached: ", np.mean(finished )," STD Reached: ", np.std(finished ), " Finished", np.count_nonzero(steps != maxStep) / traj * 100, "%")
    print("Mean Overall: ", np.mean(steps )," STD Overall: ", np.std(steps) , "Not finished", np.count_nonzero(steps == maxStep) / traj * 100, "%")
