import numpy as np
from scipy.special import softmax
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import argparse as ap
import h5py
import os
import utils

ActionDict = np.asarray([
            [-1,  0], # Downwind
            [ 0,  1], # Crosswind
            [ 1,  0], # Upwind
            [ 0, -1]  # Crosswind
        ])

#TODO:  Maybe change that this function samples from rho inside the h5 file odor, maybe providing also different starting distributions
#       Thus changing start to simply the number of trajectories instead of all the straing states
def getManyTraj(pi, odor, starts, maxSteps, unbounded=False, gamma = 0.99975, reward = None,threshold = 1e-4, src=None, shape =None, maxFrames = None):
    if src is None:
        src = np.array(odor['source'])
    else:
        src = np.array(src)
    if shape is None:
        rows, cols = odor['odor/0'].shape
    else:
        rows, cols = shape
    if maxFrames is None:
        maxFrames = odor['frames'][()]
    if reward is None:
        reward = -(1-gamma)
    M = pi.shape[1]
    assert pi.shape == (2, M, 4*M)
    cumProbs = np.cumsum(pi.reshape(-1, 4*M), axis = 1)
    ends = utils.getEndingStates(rows, cols, src)
    minDist = utils.getMinDist(starts, ends)
    curStates = starts
    curMem = np.zeros(starts.shape[0], dtype=int)
    stepsDone = np.zeros(starts.shape[0], dtype=int)
    Gs = np.zeros(starts.shape[0])
    # Checking if any starting states is an ending states. It will never happen as of now, but maybe a change in the starring condition will make it possible
    done = utils.isEnd(curStates, src)
    curFrame = np.random.randint(maxFrames) # The same for all trajectories. Maybe change, but like this it's much more efficient
    t = 0
    while np.any(~done) and t < maxSteps:
        obs = utils.getObsTurb(curStates,t,odor, rows, cols, maxFrames)
        actions = utils.chooseActions(obs, curMem, cumProbs=cumProbs)
        curStates, curMem = utils.takeActionVect(curStates, actions, rows, cols, unbounded)
        stepsDone[~done] += 1
        Gs[~done] += reward * gamma**t
        curFrame += 1
        t += 1
        done = utils.isEnd(curStates, src, done)
    Gs[~done] = -1
    return minDist / stepsDone, Gs, done, stepsDone

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("name", help="the name of the file to save into")
    parser.add_argument("dataPath", help="the path to a h5 file that contains the data")
    parser.add_argument("theta_path", help="the path to the theta values")
    parser.add_argument("-f", "--subfolder", help="save the results into another folder")
    parser.add_argument("-u", "--unbounded", help="whether to use an infinite domain", action="store_true", default=False)
    parser.add_argument("--threshold", help="The threshold abov which the agent will receive a positve observation. Default 1e-4", default=1e-4)
    args = parser.parse_args()
    thetaName = args.name
    dataPath = args.dataPath
    thetaPath = args.theta_path
    threshold = args.threshold
    unbounded = args.unbounded
    sb = args.subfolder

    odor = h5py.File(dataPath)
    rows, cols = odor['odor/0'].shape
    cMax = cols
    maxSteps = 10000

    rho = np.zeros(cols) # For now I assume to start always from the most downwind row
    if "rho" in odor:
        rho = np.array(odor["rho"])
    else:
        rho = odor['odor/0'][0] / np.sum(odor['odor/0'][0])

    traj = 10000
    if sb is None:
        saveDir = f"eval/turbulent/{thetaName}{"_unbounded" if unbounded else ""}"
    else:
        saveDir = f"eval/turbulent/{sb}/{thetaName}{"_unbounded" if unbounded else ""}"
    if threshold != 1e-4:
       saveDir+= f"_{threshold}"

    os.makedirs(saveDir, exist_ok=True)
    theta = np.load(thetaPath)
    M = theta.shape[1]
    assert theta.shape == (2, M, 4 * M)
    pi = softmax(theta, axis = 2)
    print("PI to be evaluated: ", pi, flush= True)
    steps = np.zeros(traj)
    timeToReach = np.zeros(traj)
    empiricalG = np.zeros(traj)
    successes = np.zeros(traj, dtype=bool)
    print("Inizio: ", time.ctime(), flush=True)
    startsCols = np.random.choice(range(cols), size=traj, replace=True, p = rho)
    starts = np.array([(0, cols) for cols in startsCols])
    s = time.perf_counter()


    timeToReach, empiricalG, successes, steps = getManyTraj(pi, odor, starts, maxSteps)
    e = time.perf_counter()
    print(f"{traj} trajectories done in {e-s}s")
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

    