import numpy as np
from scipy.special import softmax as softmax
import time
import os
import sys
import argparse as ap

# Setting Parameters
SC = 92 * 131 # Number of States
cSource = 45.5 # Source coordinates
rSource = 91
cols = 92
rows = 131
find_range = 1.1 # Source radius
gamma = 0.99975
reward = -(1 -gamma)
maxSteps = 10000

ActionDict = np.asarray([
            [-1,  0], # North
            [ 0,  1], # East
            [ 1,  0], # South
            [ 0, -1]  # West
        ])
dataC = np.load("celaniData/fine5.npy")
rho = np.zeros(SC)
rho[:cols] = (1-dataC[0,:cols])/np.sum((1-dataC[0,:cols]))

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

def isEnd(sm):
    s = sm % SC
    r, c = s // cols, s % cols
    return (r - rSource) ** 2 + (c -cSource) **2 < find_range**2

def takeAction(sm, am):
    s = sm % SC
    a = am % 4
    newM = am // 4
    r, c = s // cols, s % cols
    action = ActionDict[a]
    rNew = r + action[0]
    cNew = c + action[1]
    r = rNew if rNew >= 0 and rNew < 131 else r
    c = cNew if cNew >= 0 and cNew < 92 else c
    return r * 92 + c + newM * SC

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


parser = ap.ArgumentParser()
parser.add_argument("name", help="name the folder in which to save the result")
parser.add_argument("lr", help="the starting learning rate", type=float)
parser.add_argument("memories", type=int, help="The memories of the FSC")
parser.add_argument("episodes", type=int, help="The final episode number. ")
parser.add_argument("-t","--thetaStart", help="the path to a .npy file containing the starting values of theta")
parser.add_argument("--schedule", help="wheter or not to decrease the learning rate", action="store_true")
parser.add_argument("--subMax", help="if specified, every iteration from each row of theta will be subtracted its maximum", action="store_true")
parser.add_argument("--toClip", help="if specified, theta's entries will be clipped in the [-20, 0] interval", action="store_true")
parser.add_argument("--vanilla", help="whether to use vanilla gradient instead on the natural", action="store_true")
args = parser.parse_args()
name = args.name
lr = args.lr
M = args.memories
episodes = args.episodes
thetaPath = args.thetaStart
schedule = args.schedule
subMax = args.subMax
toClip = args.toClip
vanilla = args.vanilla

assert M >= 1
if thetaPath is not None:
    theta = np.load(thetaPath)
else:
    theta = (np.random.rand(2, M, 4*M) -0.5) * 0.5

saveDir = f"results/reinforce{"_Vanilla" if vanilla else ""}/M{M}/lr_{lr}{"_scheduled" if schedule else ""}/{name}_episodes{episodes}"
thetaDir = saveDir+"/thetas"
os.makedirs(saveDir)
os.makedirs(thetaDir)
ouput = open(os.path.join(saveDir, "_results.out"), "w")

grad = vanillaGrad if vanilla else natGrad
pi = softmax(theta, axis = 2)

s = time.perf_counter()
print(f" Startinng {episodes} episodes at {time.ctime()}",file=ouput)
print("Starting pi:", pi,file=ouput, flush=True)
np.save(thetaDir+"/thetaStart.npy", theta)


i = 0
s = time.perf_counter()
while i < episodes:
    s1 = time.perf_counter()
    curLr = lr * 1000 / (1000 + i)
    curState = np.random.choice(range(SC), p = rho)
    curMem = 0
    step = 0
    history = np.zeros((maxSteps, 3), dtype=np.uint8) # Observation 0; Memory 1; Actions 2
    while not isEnd(curState) and step < maxSteps:
        obs = np.random.choice(2, p = dataC[:, curState % SC])
        action = np.random.choice(4 * M, p= pi[obs, curMem])
        history[step, 0] = obs
        history[step, 1] = curMem
        history[step, 2] = action
        curState = takeAction(curState, action)
        curMem = curState // SC
        step += 1
    hasFoundSource = isEnd(curState)
    for j in range(step):
        G = 0
        for k in range(j, step-1): # Should have the correct reward both when it found the source and when it didn't
            G += gamma ** (k-j)*reward
        if not hasFoundSource:
            G += gamma ** (step-1-j) * reward
            
        # It could likely be made more efficient, but for now I implemented exactly as in the sutton
        theta += curLr * gamma ** j * G * grad(pi, history[j,0], history[j,1], history[j,2])
    if subMax:
        theta -= np.max(theta, axis =2 , keepdims=True)
    if toClip:
            theta = np.clip(theta, -20, 0)
    e1 = time.perf_counter()
    # print(i, e1-s1, flush=True, file = ouput)
    pi = softmax(theta, axis = 2)
    if np.any(np.isclose(pi[0], 1)):
        np.save(os.path.join(thetaDir , f"thetaErr_{i+1}.npy"), theta)
        print(f"Error iteration {i+1}: reached determinism at {time.ctime()} ", file=ouput)
        sys.exit()
    # print(file=ouput, flush=True)
    if (i+1) % 1000 == 0:
        print(f"Episode {i+1} done at {time.ctime()}",file=ouput, flush=True)
        np.save(os.path.join(thetaDir , f"theta{i+1}.npy"), theta)
    i+=1
e = time.perf_counter()

print("Learned pi:", pi,file=ouput)
np.save(os.path.join(thetaDir,"thetaFinal.npy"), theta)
totalTime(e, s, ouput)