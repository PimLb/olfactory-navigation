import numpy as np
from scipy.special import softmax as softmax
import time
import os
import sys
import argparse as ap
import h5py
import signal
import utils

ActionDict = np.array([
            (-1,  0), # Downwind
            ( 0,  1), # Crosswind (Left when facing upwind)
            ( 1,  0), # Upwind
            ( 0, -1)  # Crosswind (Right when facing upwind)
        ])

def natGrad(pi, obs, curMem, action):
    ret = np.zeros_like(pi)
    ret[obs, curMem, action] = 1 / pi[obs, curMem, action]
    return ret

def vanillaGrad(pi, obs, curMem, action):
    ret = np.zeros_like(pi)
    ret[obs, curMem, action] = 1
    for b in range(pi.shape[2]):
        ret[obs,curMem, b] -= pi[obs, curMem, b]
    return ret

def isEnd(state, rSource, cSource, find_range = 1.1):
    r, c = state
    return (r - rSource) ** 2 + (c -cSource) **2 < find_range**2

def takeAction(state, actionMem, rMax, cMax):
    #TODO: count how many times the agent hit the boundaries? 
    actionIdx = actionMem % 4
    newM = actionMem // 4
    r, c = state
    action = ActionDict[actionIdx]
    r = np.clip(r + action[0], 0, rMax -1)
    c = np.clip(c + action[1], 0, cMax -1)
    return (r,c),newM

def getObservation(state, odor, time, threshold, maxFrames):
    r,c = state
    t = time % maxFrames
    return int(odor[f"odor/{str(t)}"][r,c] >= threshold)
    
def totalTime(end, start, file = None):
    tot = end - start
    seconds = int(tot % 60)
    minutes = int((tot // 60) % 60)
    hours = int(tot // 3600)
    print(f"End time: {time.ctime()}", file=file)
    if hours >= 24:
        days = hours // 24
        hours = hours % 24
        print(f"Total time: {days}d {hours}:{minutes}:{seconds}", file=file)
        return
    if hours == 0:
        print(f"Total time: {minutes}m:{seconds}s", file=file)
        return
    print(f"Total time: {hours}h:{minutes}m:{seconds}s", file=file)
#TODO: won't work anymore
def handleTERM(sig, frame): # Maybe not best practice, but seems to work
    e = time.perf_counter()
    print("Terminated:", end="", file=avgOutput)
    totalTime(e, s, avgOutput)
    exit()

parser = ap.ArgumentParser()
parser.add_argument("name", help="name the folder in which to save the result")
parser.add_argument("dataPath", help="the path to a h5 file that contains the data")
parser.add_argument("lr", help="the starting learning rate", type=float)
parser.add_argument("memories", type=int, help="The memories of the FSC")
parser.add_argument("episodes", type=int, help="The final episode number. ")
parser.add_argument("-c", "--clip", type=float, help="TODO")
parser.add_argument("-p", "--parallel", type=int, help="TODO", default=1)
parser.add_argument("--threshold", default=1e-4, help="The concentration of odor above which the agent receives a positive observation", type=float)
parser.add_argument("-t","--thetaStart", help="the path to a .npy file containing the starting values of theta")
parser.add_argument("--schedule", help="wheter or not to decrease the learning rate", action="store_true")
parser.add_argument("--vanilla", help="if specified uses vanilla gradient instead on the natural", action="store_true")
parser.add_argument("--logEvery", help="how many episodes between checkpoint and print", type=int, default=1000)
args = parser.parse_args()
name = args.name
dataPath = args.dataPath
lr = args.lr
M = args.memories
episodes = args.episodes
thetaPath = args.thetaStart
schedule = args.schedule
vanilla = args.vanilla
threshold = args.threshold
c = args.clip
parallelRuns = args.parallel
logEvery = args.logEvery

odor = h5py.File(dataPath)
rows, cols = odor['odor/0'].shape
SC = rows * cols
src = odor['source']
maxFrames = odor['frames'][()] # Syntax to get a scalar value from h5py
find_range = 1.1 # Source radius

gamma = 0.99975
reward = -(1 -gamma)
maxSteps = 10000

rho = np.zeros(cols) # For now I assume to start always from the most downwind row
if "rho" in odor:
    rho = np.array(odor["rho"])
else:
    rho = odor['odor/0'][0] / np.sum(odor['odor/0'][0])

assert M >= 1
outputs = []
if thetaPath is not None:
    thetas = np.tile(np.load(thetaPath),(parallelRuns,1,1,1))
else:
    thetas = (np.random.rand(parallelRuns, 2, M, 4*M) -0.5) * 0.5
if parallelRuns == 1:
    saveDirs = [f"storage/reinforce/turbulent/thresh_{threshold}/{"vanilla" if vanilla else "natural"}/M{M}/lr_{lr}{"_scheduled" if schedule else ""}/{name}_episodes{episodes}/"]
    thetaDirs = [saveDirs[0]+"/thetas"]
else:
    parentDir = f"storage/reinforce/turbulent/thresh_{threshold}/{"vanilla" if vanilla else "natural"}/M{M}/lr_{lr}{"_scheduled" if schedule else ""}/{name}"
    saveDirs = [f"{parentDir}/run_{n}_episodes{episodes}/" for n in range(parallelRuns)]
    thetaDirs = [sd+"/thetas" for sd in saveDirs]
for sd in thetaDirs:
    os.makedirs(sd)
for sd in saveDirs:
    outputs.append(open(os.path.join(sd, "_results.out"), "w"))
if parallelRuns > 1:
    # TODO: make a file that tracks the averages of all runs? FOr now should work
    avgOutput = open(f"{parentDir}/average.out", "w")
calcGrad = utils.vanillaGradVect if vanilla else utils.natGradVect
pis = softmax(thetas, axis = 3)

# Given that the reward is constant, the cumulative reward depends only on time to reach the source and we can precompute them
cumulativeRewards = np.cumsum([gamma**a for a in range(maxSteps)]+ [-1])[::-1]* reward

# Given a reward list rl, where rewards are arbitrary
# G = 0
# for r in reversed(rl):
#   G = r + gamma * G
# produces the correct G. If at each step they're prependend to a list, it computes the correct in O(n)
if parallelRuns > 1:
    print(f" Starting {parallelRuns} parallel run of {episodes} episodes at {time.ctime()}",file=avgOutput, flush=True)
for i in range(parallelRuns):
    print(f" Starting {parallelRuns} parallel run of {episodes} episodes at {time.ctime()}",file=outputs[i], flush=True)
    np.save(thetaDirs[i]+"/thetaStart.npy", thetas[i])

i = 0
s = time.perf_counter()
signal.signal(signal.SIGTERM, handleTERM)
reached = np.zeros(parallelRuns, dtype=int)
stepsDone = np.zeros(parallelRuns, dtype=int)
empiricalTimeNormalized = np.zeros(parallelRuns)
empiricalG = np.zeros(parallelRuns)
curLr = lr
unbounded = False #TODO: add parameter and make it works
dead = np.zeros(parallelRuns, dtype=bool)
while i < episodes:
    if schedule:
        curLr = lr * 1000 / (1000 + i)
    startCols = np.random.choice(range(cols), p = rho, size=parallelRuns)
    curStates = np.stack((np.zeros_like(startCols), startCols)).T # (row,col) -> starts from most downwind row
    curMems = np.zeros_like(startCols) # starts from first memory

    ends = utils.getEndingStates(rows, cols, src)
    minDist = utils.getMinDist(curStates, ends)

    step = 0
    stepsDone[:] = 0
    t = np.random.randint(maxFrames) # Randomize on the starting time of the plume. Will make it harder to overfit the plume dynamic

    # history is initialized like this because if there's a bug that makes the gradient look at wrong steps, it will raise an OutOfBound Exception
    history = np.full((maxSteps, 3, parallelRuns), np.iinfo(np.uint8).max, dtype=np.uint8) # Observation 0; Memory 1; Actions 2
    done = utils.isEnd(curStates, src, dead) # Won't count training that arrived to determinism
    while np.any(~done) and step < maxSteps:
        obs = utils.getObsTurb(curStates, t, odor, rows, cols, maxFrames)
        actions = utils.chooseActions(obs, curMems, pis=pis)
        history[step, 0, ~done] = obs[~done]
        history[step, 1, ~done] = curMems[~done]
        history[step, 2, ~done] = actions[~done]
        curStates, curMems = utils.takeActionVect(curStates, actions, rows, cols, unbounded)
        stepsDone[~done] += 1
        done = utils.isEnd(curStates, src, done)
        step += 1
        t+=1
    reached += done & ~dead
    empiricalTimeNormalized[done & ~dead] += minDist[done & ~dead] / stepsDone[done & ~dead]
    empiricalG[done & ~dead] += cumulativeRewards[-stepsDone[done & ~dead]]
    empiricalG[~done & ~dead] += -1# Failed episode treated as if it would never reach the source. Uncomparable with Model Based G, though.
    for j in range(step):
        mask = -stepsDone + j < 0
        grad = calcGrad(pis[mask], history[j,0,mask], history[j,1,mask], history[j,2,mask])
        if c is not None:
            utils.clipMultiGrad(grad, c) # Operation with side effect. Will compute the clipping in-place. TODO: maybe eventually check if different norms are better
        thetas[mask] += curLr * gamma ** j * cumulativeRewards[-stepsDone[mask]+j, None, None, None] * grad
    thetas -= np.max(thetas, axis =3 , keepdims=True)
    pis = softmax(thetas, axis = 3)
    if (i+1) % logEvery == 0:
        for k in range(parallelRuns):
            if not dead[k]:
                print(f"Episode {i+1} done at {time.ctime()}; In the last {logEvery} episodes: {reached[k]/logEvery:.1%}",
                    f"Time to reach normalized (only succesful): {empiricalTimeNormalized[k] / reached[k] if reached[k] != 0 else 0.0}",
                    f"Empirical G: {empiricalG[k] / logEvery}", file=outputs[k], flush=True)
                np.save(os.path.join(thetaDirs[k] , f"theta{i+1}.npy"), thetas[k])
        if parallelRuns > 1:  
            mask = reached != 0
            #TODO: aggiungere standard error
            #TODO: usare la maschera ~dead fa in modo che i training morti tra un checkpoint e l'altro non vengano conteggiati affatto, nemmeno per la parte "ancora buona"
            print(f"Episode {i+1} done at {time.ctime()}; In the last {logEvery} episodes on average of {parallelRuns} agents: {np.mean(reached[~dead])/logEvery:.1%} success",
                f"Time to reach normalized (only succesful): {np.mean(empiricalTimeNormalized[mask] / reached[mask])}",
                f"Empirical G: {np.mean(empiricalG[~dead]) / logEvery}",
                f"{np.count_nonzero(dead)/parallelRuns:.1%} trainings dead", file=avgOutput, flush=True)
            
        reached[:] = 0
        empiricalTimeNormalized[:] = 0
        empiricalG[:] = 0
    if np.any(np.isclose(pis[:, 0], 1)):
        for k in range(parallelRuns):
            if not dead[k] and np.any(np.isclose(pis[k, 0], 1)):
                e = time.perf_counter()
                np.save(os.path.join(thetaDirs[k] , f"thetaErr_{i+1}.npy"), thetas[k])
                print(f"Error iteration {i+1}: reached determinism at {time.ctime()} ", file=outputs[k])
                totalTime(e, s, outputs[k])
                dead[k] = True
                if np.count_nonzero(dead) >= parallelRuns:
                    print(f"All Dead at {time.ctime()} ", file=avgOutput)
                    totalTime(e, s, avgOutput)
                    print("All dead___", dead)
                    sys.exit()
    # print(file=ouput, flush=True)
    i+=1
e = time.perf_counter()
for k in range(parallelRuns):
    if not dead[k]:
        np.save(os.path.join(thetaDirs[k],"thetaFinal.npy"), thetas[k])
        totalTime(e, s, outputs[k])
    # totalTime(e, s)
if parallelRuns > 1:
    totalTime(e, s, avgOutput)