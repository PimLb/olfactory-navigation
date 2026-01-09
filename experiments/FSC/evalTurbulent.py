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
rMax = 0
cMax = 0
odor = None
threshold = 1e-4
rowNoPad = 0
maxFrames = 0
pi = None
M = 0
maxSteps = 0

def isEnd(state, find_range = 1.1):
    r, c = state
    return (r - rSource) ** 2 + (c -cSource) **2 < find_range**2

def takeAction(state, actionMem):
    #TODO: count how many times the agent hit the boundaries? 
    actionIdx = actionMem % 4
    newM = actionMem // 4
    r, c = state
    action = ActionDict[actionIdx]
    r = np.clip(r + action[0], 0, rMax -1)
    c = np.clip(c + action[1], 0, cMax -1)
    return (r,c),newM

def getObservation(state, time):
    r,c = state
    if r >= rowNoPad:
        return 0
    t = time % maxFrames
    try:
        return int(odor[f"odor/{str(t)}"][r,c] >= threshold)
    except IndexError as e:
        print(state, odor[f"odor/{str(t)}"].shape, rowNoPad)
        raise e
    

def choose_action(o, curMem):
    return np.random.choice(4 * M, p = pi[o, curMem])

def getTrajectories(start ):
    # num = start.shape[0]
    num = len(start)
    # curStates = start.astype(int)
    curStates = start
    curMem = np.zeros(num,int)
    done = np.array([a for a in map(isEnd, curStates)])
    # obs = np.array([a for a in map(get_observation, step, [0 for i in range(num)])])
    t = 0
    stepsDone = np.zeros(num)
    while np.any(~done) and t < maxSteps:
        obs = np.array([a for a in map(getObservation, curStates, [t for i in range(num)])])
        for i in range(num): 
            if not done[i]:
                a = choose_action(obs[i], curMem[i])
                curStates[i], curMem[i] = takeAction(curStates[i], a)
        t+= 1
        stepsDone[~done] += 1
        done = np.array([a for a in map(isEnd, curStates)])
    return stepsDone

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("name", help="the name of the file to save into")
    parser.add_argument("dataPath", help="the path to a h5 file that contains the data")
    parser.add_argument("theta_path", help="the path to the theta values")
    parser.add_argument("--threshold", help="The threshold abov which the agent will receive a positve observation. Default 1e-4", default=1e-4)
    parser.add_argument("--padRows", type=int, default=0, help="how many rows add behind the last row of the data. In these rows, the observation is always 0")
    args = parser.parse_args()
    thetaName = args.name
    dataPath = args.dataPath
    thetaPath = args.theta_path
    threshold = args.threshold
    padRows = args.padRows

    odor = h5py.File(dataPath)
    rowNoPad, cols = odor['odor/0'].shape
    rMax = rowNoPad + padRows
    cMax = cols
    SC = rMax * cols
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

    saveDir = f"eval/turbulent/{thetaName}_{threshold}"

    os.makedirs(saveDir, exist_ok=True)
    # for s in range(SC):
    #     if rho[s] > 0:
    #         print(s // cols, s % cols)
    # sys.exit()
    theta = np.load(thetaPath)
    M = theta.shape[1]
    assert theta.shape == (2, M, 4 * M)
    pi = softmax(theta, axis = 2)
    print("PI to be evaluated: ", pi, flush= True)
    results = np.zeros(traj)
    print("Inizio: ", time.ctime(), flush=True)
    startsCols = np.random.choice(range(cols), size=traj, replace=True, p = rho)
    starts = [(0, cols) for cols in startsCols]

    N = traj // procNumber
    remainder = traj % procNumber
    print("prima: ", pi[0, 0].shape)
    args = [starts[i * procNumber: (i+1)* procNumber] for i in range(N)]
    with mp.Pool(procNumber) as p:
        rewardList = p.map(getTrajectories, args)
    print(time.ctime())
    for i, rl in enumerate(rewardList):
        results[procNumber*i:procNumber*(i+1)] += rl
    np.save(f"{saveDir}/res.npy", results)
    n, b, patch = plt.hist(results, 50, range = (0, maxSteps))
    patch[-1].set_facecolor('red')
    plt.ylim(0, 5000)
    finished = results[results != maxSteps]
    plt.yticks([i*500 for i in range(0, 11)] + [np.count_nonzero(results == maxSteps)])
    plt.title(thetaName)

    plt.savefig(f"{saveDir}/res.png")
    print("Mean Reached: ", np.mean(finished )," STD Reached: ", np.std(finished ), " Finished", np.count_nonzero(results != maxSteps) / traj * 100, "%")
    print("Mean Overall: ", np.mean(results )," STD Overall: ", np.std(results) , "Not finished", np.count_nonzero(results == maxSteps) / traj * 100, "%")

    