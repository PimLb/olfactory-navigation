import numpy as np
from scipy.special import softmax
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import argparse as ap
import h5py
import os

ActionDict = np.asarray([
            [-1,  0], # Downwind
            [ 0,  1], # Crosswind
            [ 1,  0], # Upwind
            [ 0, -1]  # Crosswind
        ])
# I leave these here as Global vars so when I change the values in the main, the change is reflected in the various functions
# Ugly, but better (IMO) than having to init a list of the same parameter when using map or to define the functions inside getTraj
rSource = 0
cSource = 0
rows = 0
cMax = 0
odor = None
threshold = 1e-4
maxFrames = 0
pi = None
cumProbs = None
M = 0
maxSteps = 0

def chooseActionsVect(obs, curMems):
    actions = np.arange(4*M)
    CDFs = cumProbs[obs*M+curMems]
    u = np.random.random(obs.shape)[:, None]
    idx = np.argmax(u < CDFs, axis = 1)
    return actions[idx]

def takeActionVect(curStates, actionsMem):
    #TODO: count how many times the agent hit the boundaries? 
    newStates = np.clip(curStates + ActionDict[actionsMem % 4], [0,0], [rows-1, cols-1])
    newMem = actionsMem // 4
    return newStates, newMem

def getObsVect(states, t):
    curFrame = t % maxFrames
    return odor[f'odor/{curFrame}'][:][states[:, 0], states[:, 1]] >= threshold

def isEndVect(curStates, finished=False):
    return finished | (np.linalg.norm(curStates-[rSource, cSource], axis=1) < find_range)

def getManyTraj(starts):
    minDist = np.linalg.norm(starts-[rSource, cSource], ord=1, axis=1)
    curStates = starts
    curMem = np.zeros(starts.shape[0], dtype=int)
    stepsDone = np.zeros(starts.shape[0], dtype=int)
    Gs = np.zeros(starts.shape[0])
    # Checking if any starting states is an ending states. It will never happen as of now, but maybe a change in the starring condition will make it possible
    done = isEndVect(starts)
    curFrame = np.random.randint(maxFrames) # The same for all trajectories. Maybe change, but like this it's much more efficient
    t = 0
    while np.any(~done) and t < maxSteps:
        obs = getObsVect(curStates, curFrame)
        actions = chooseActionsVect(obs, curMem)
        curStates, curMem = takeActionVect(curStates, actions)
        stepsDone[~done] += 1
        Gs[~done] += reward * gamma**t
        curFrame += 1
        t += 1
        done = isEndVect(curStates, done)
    return minDist / stepsDone, Gs, done, stepsDone

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("name", help="the name of the file to save into")
    parser.add_argument("dataPath", help="the path to a h5 file that contains the data")
    parser.add_argument("theta_path", help="the path to the theta values")
    parser.add_argument("--threshold", help="The threshold abov which the agent will receive a positve observation. Default 1e-4", default=1e-4)
    args = parser.parse_args()
    thetaName = args.name
    dataPath = args.dataPath
    thetaPath = args.theta_path
    threshold = args.threshold

    odor = h5py.File(dataPath)
    rows, cols = odor['odor/0'].shape
    cMax = cols
    SC = rows * cols
    rSource, cSource = odor['source']
    find_range = 1.1 # Source radius
    maxFrames = odor['frames'][()] # Syntax to get a scalar value from h5py
    maxSteps = 10000

    gamma = 0.99975
    reward = -(1 -gamma)

    rho = np.zeros(cols) # For now I assume to start always from the most downwind row
    if "rho" in odor:
        rho = np.array(odor["rho"])
    else:
        rho = odor['odor/0'][0] / np.sum(odor['odor/0'][0])

    procNumber = 50
    traj = 10000

    saveDir = f"eval/turbulent/{thetaName}"
    if threshold != 1e-4:
       saveDir+= f"_{threshold}"

    os.makedirs(saveDir, exist_ok=True)
    theta = np.load(thetaPath)
    M = theta.shape[1]
    assert theta.shape == (2, M, 4 * M)
    pi = softmax(theta, axis = 2)
    cumProbs = np.cumsum(pi.reshape(-1, 4*M), axis = 1)
    print("PI to be evaluated: ", pi, flush= True)
    steps = np.zeros(traj)
    timeToReach = np.zeros(traj)
    empiricalG = np.zeros(traj)
    successes = np.zeros(traj, dtype=bool)
    print("Inizio: ", time.ctime(), flush=True)
    startsCols = np.random.choice(range(cols), size=traj, replace=True, p = rho)
    starts = np.array([(0, cols) for cols in startsCols])
    timeToReach, empiricalG, successes, steps = getManyTraj(starts)
    np.save(f"{saveDir}/res.npy", np.stack((timeToReach, empiricalG, successes, steps)))
    plt.boxplot((timeToReach[successes],empiricalG), tick_labels = ["t", r"$\hat{G}$"])
    plt.title(thetaName)

    plt.savefig(f"{saveDir}/res.svg")
    plt.close()
    n, b, patch = plt.hist(steps, 50, range = (0, maxSteps))
    patch[-1].set_facecolor('red')
    plt.ylim(0, 5000)
    plt.yticks([i*500 for i in range(0, 11)] + [np.count_nonzero(~successes)])
    plt.savefig(f"{saveDir}/hist.svg")
    print(f"{np.count_nonzero(successes) / len(successes):.2%} success rate")
    # print("Mean Reached: ", np.mean(finished )," STD Reached: ", np.std(finished ), " Finished", np.count_nonzero(results != maxSteps) / traj * 100, "%")
    # print("Mean Overall: ", np.mean(results )," STD Overall: ", np.std(results) , "Not finished", np.count_nonzero(results == maxSteps) / traj * 100, "%")

    