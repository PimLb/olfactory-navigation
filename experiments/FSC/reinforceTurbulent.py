import numpy as np
from scipy.special import softmax as softmax
import time
import os
import sys
import argparse as ap
import h5py
import signal

ActionDict = [
            (-1,  0), # North
            ( 0,  1), # East
            ( 1,  0), # South
            ( 0, -1)  # West
        ]

def natGrad(pi, obs, curMem, action):
    ret = np.zeros_like(pi)
    ret[obs, curMem, action] = 1 / pi[obs, curMem, action]
    # print(1 / pi[obs, curMem, action])
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

def handleTERM(sig, frame): # Maybe not best practice, but seems to work
    e = time.perf_counter()
    print("Terminated:", end="", file=output)
    totalTime(e, s, output)
    exit()

parser = ap.ArgumentParser()
parser.add_argument("name", help="name the folder in which to save the result")
parser.add_argument("dataPath", help="the path to a h5 file that contains the data")
parser.add_argument("lr", help="the starting learning rate", type=float)
parser.add_argument("memories", type=int, help="The memories of the FSC")
parser.add_argument("episodes", type=int, help="The final episode number. ")
parser.add_argument("--threshold", default=1e-4, help="The concentration of odor above which the agent receives a positive observation", type=float)
parser.add_argument("-t","--thetaStart", help="the path to a .npy file containing the starting values of theta")
parser.add_argument("--schedule", help="wheter or not to decrease the learning rate", action="store_true")
parser.add_argument("--vanilla", help="if specified uses vanilla gradient instead on the natural", action="store_true")
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

odor = h5py.File(dataPath)
rows, cols = odor['odor/0'].shape
SC = rows * cols
rSource, cSource = odor['source']
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
if thetaPath is not None:
    theta = np.load(thetaPath)
else:
    theta = (np.random.rand(2, M, 4*M) -0.5) * 0.5

saveDir = f"storage/reinforce/turbulent/thresh_{threshold}/{"vanilla" if vanilla else "natural"}/M{M}/lr_{lr}{"_scheduled" if schedule else ""}/{name}_episodes{episodes}/"
thetaDir = saveDir+"/thetas"
os.makedirs(saveDir)
os.makedirs(thetaDir)
output = open(os.path.join(saveDir, "_results.out"), "w")

grad = vanillaGrad if vanilla else natGrad
pi = softmax(theta, axis = 2)

# Given that the reward is constant, the cumulative reward depends only on time to reach the source and we can precompute them
cumulativeRewards = np.cumsum([gamma**a for a in range(maxSteps)] )[::-1]* reward

print(f" Starting {episodes} episodes at {time.ctime()}",file=output, flush=True)
np.save(thetaDir+"/thetaStart.npy", theta)


i = 0
s = time.perf_counter()
signal.signal(signal.SIGTERM, handleTERM)
reached = 0
empiricalTimeNormalized = 0
empiricalG = 0
while i < episodes:
    s1 = time.perf_counter()
    curLr = lr * 1000 / (1000 + i)
    startCol = np.random.choice(range(cols), p = rho)
    curState = (0, startCol) # (row,col) -> starts from most downwind row
    curMem = 0 # starts from first memory
    step = 0
    t = np.random.randint(maxFrames) # Randomize on the starting time of the plume. Will make it harder to overfit the plume dynamic
    history = np.zeros((maxSteps, 3), dtype=np.uint8) # Observation 0; Memory 1; Actions 2
    while not isEnd(curState, rSource, cSource) and step < maxSteps:

        obs = getObservation(curState, odor, t, threshold, maxFrames)
        action = np.random.choice(4 * M, p= pi[obs, curMem])
        history[step, 0] = obs
        history[step, 1] = curMem
        history[step, 2] = action
        curState, curMem = takeAction(curState, action, rows, cols)
        step += 1
        t+=1
    if step < maxSteps:
        reached += 1
        manhattanSource = np.abs(startCol - cSource) + rSource # Since we always start from row 0, startRow would be 0
        empiricalTimeNormalized += manhattanSource / step
        empiricalG += cumulativeRewards[-step]
    else:
        empiricalG += -1 # Failed episode treated as if it would never reach the source. Uncomparable with Model Based G, though.
    for j in range(step):
        theta += curLr * gamma ** j * cumulativeRewards[-step+j] * grad(pi, history[j,0], history[j,1], history[j,2])
    theta -= np.max(theta, axis =2 , keepdims=True)
    pi = softmax(theta, axis = 2)
    if np.any(np.isclose(pi[0], 1)):
        e = time.perf_counter()
        np.save(os.path.join(thetaDir , f"thetaErr_{i+1}.npy"), theta)
        print(f"Error iteration {i+1}: reached determinism at {time.ctime()} ", file=output)
        totalTime(e, s, output)
        sys.exit()
    # print(file=ouput, flush=True)
    if (i+1) % 1000 == 0:
        print(f"Episode {i+1} done at {time.ctime()}; In the last 1000 episodes: {reached/1000:.1%}",
              f"Time to reach normalized (only succesful): {empiricalTimeNormalized / reached if reached != 0 else 0.0}",
              f"Empirical G: {empiricalG / 1000}", file=output, flush=True)
        reached = 0
        empiricalTimeNormalized = 0
        empiricalG = 0
        np.save(os.path.join(thetaDir , f"theta{i+1}.npy"), theta)
    i+=1
    e1 = time.perf_counter()
    # print("Total Iteration", step, e1-s1, flush = True)
e = time.perf_counter()

np.save(os.path.join(thetaDir,"thetaFinal.npy"), theta)
totalTime(e, s, output)
totalTime(e, s)