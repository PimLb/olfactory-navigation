import numpy as np
import cupy as cp
from modelBasedTrain import ggTrasfer, sparse_T_CPU
import sys
from scipy.special import softmax
import scipy.sparse as sparse
import argparse as ap
import matplotlib.pyplot as plt
import re
import glob
import time
import os

SC = 131*92
gamma = 0.99975
cSource = 45.5
rSource = 91
find_range = 1.1
dataC = np.load("celaniData/fine5.npy")
cols = 92
rows = 131

reward = -(1 -gamma)

def manhattan(pos):
    source = np.array([91, 45.5])
    manhattan_distance = np.sum(np.abs( pos - source))    
    distance_to_border = max(0, manhattan_distance - 1.1)
    return np.ceil(distance_to_border)


def phi(s):
    r,c = np.unravel_index(s % SC, (131,92))
    ret = 1 -manhattan([r,c])/136
    return  ret

def F(s, sNext, d0 = None):
    return gamma *phi(sNext) - phi(s)

def getReachable(s, M):
    r, c, m = (s % SC) // 92, s % 92 , s // SC
    stateInMem0 = r * 92 + c
    ret = np.zeros(4 * M, dtype = int)
    for i in range(M):
        ret[0 + i *4] = stateInMem0 - 92 + SC * i if r - 1 >= 0 else  stateInMem0 + SC * i
        ret[1 + i *4] = stateInMem0 + 1  + SC * i if c + 1 < 92 else  stateInMem0 + SC * i
        ret[2 + i *4] = stateInMem0 + 92 + SC * i if r + 1 < 131 else stateInMem0 + SC * i
        ret[3 + i *4] = stateInMem0  -1  + SC * i if c - 1 >= 0 else  stateInMem0 + SC * i
    return ret

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

def getTraj(pi, pObs, rho):
    start = np.random.choice(range(131*92), size=1, p = rho[:SC]).astype(int)[0]
    d0 = manhattan(np.unravel_index(start, (131,92)))
    dBox = 131 + 92 -2
    obs = get_observation(start, pObs)
    curState = start
    curM = 0
    Rewards = np.zeros((4, 10000))
    history = np.zeros((3, 10001)).astype(int)
    history[0, 0] = start
    history[1, 0] = obs
    t = 0
    while not isEnd(curState) and t < 10000:
        if curState == 8417 or curState == 8418 or curState == 20469 or curState == 20470:
            print("AAAAAA")
        curD = -manhattan(np.unravel_index(curState % SC, (131,92)))
        action = choose_action(pi, obs, M, curM)
        history[2, t] = action
        newState, curM = takeAction(curState, action)
        nextD = -manhattan(np.unravel_index(newState % SC, (131,92)))
        Rewards[0, t] = gamma**t * (reward +gamma * nextD/d0 -curD/d0)
        Rewards[1, t] = gamma**t * (reward +gamma * nextD/dBox -curD/dBox)
        Rewards[2, t] = gamma**t * (reward + gamma * phi(newState) - phi(curState))
        Rewards[3, t] = gamma**t * reward
        obs = get_observation(newState, pObs)
        t += 1
        history[0, t] = newState
        history[1, t] = obs
        curState = newState
    return history, t, Rewards

def prettyTraj():


    hst, t, rs = getTraj(softmax(th, axis = 2), dataC, rho)
    print(t)
    x, y = np.unravel_index(hst[0, :t] % SC, (131, 92))
    pippo = np.linspace(0,1, len(x))
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    fig, ax = plt.subplots(figsize=(20,10))
    lc = LineCollection(segments, cmap="viridis")
    lc.set_linewidth(2)
    lc.set_array(pippo)
    line = ax.add_collection(lc)
    ax.set_xlim([0,131])
    ax.set_ylim([0,92])

    ax.matshow(dataC.reshape((2,131,92))[1].T, cmap = "binary")
    ax.add_patch(plt.Circle((91,45.5), 1.1, color="r"))
    ax.add_patch(plt.Circle((x[0], y[0]), 0.9, color="b"))
    ax.add_patch(plt.Circle((x[-1], y[-1]), 0.7, color="g"))
    xObs = x[np.where(hst[1, :t])]
    yObs = y[np.where(hst[1, :t])]
    ax.scatter(xObs, yObs, c="k")

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

def plot_and_save(totIter, thetas, obj, normDiff, diffFromOpt, diffPrev, paramas, name, M, sb, vanilla, close = False):
    color = "orange"
    style = "--"
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"{name}\n Actor Lambda {paramas[1]}; Lr {paramas[3]}\nCritic Lambda {paramas[2]}; Lr {paramas[4]}; M {paramas[0]}" + ("\n Vanilla" if vanilla else ""))
    if vanilla:
        name += "_vanilla"
    plt.subplot(2,2, 1)
    plt.plot(range(totIter +1), thetas, label = "Theta Norm")
    plt.legend()
    plt.subplot(2,2,2)
    ticks = [0, -0.1, -0.3, -0.4,-0.485, -0.6, -0.7, -0.8, -0.9, -1]
    if M == 3:
        plt.hlines(-0.138, 0,totIter, "r", label = f"Optimal M3")
        ticks += [-0.138]
    if M >= 2:
        ticks += [-0.197]
        plt.hlines(-0.197, 0,totIter, "y", label = f"Optimal M2")
    else:
        ticks += [-0.2]
    plt.hlines(-0.485, 0,totIter, "g", label = f"Optimal M1")
    plt.yticks(ticks)
    plt.plot(range(totIter+1), obj,marker=None if obj.shape[0] > 5 else "x",markersize = 8, label = "Objective")
    plt.grid()
    # plt.ylim(-1, -0.475)
    plt.legend()
    plt.subplot(2,2,3)
    plt.plot(range(1,totIter+1), normDiff, label = "Diff from True")
    plt.plot(range(1,totIter+1), diffFromOpt, label = "Diff from Optimal")
    plt.hlines(0, 0,totIter, "k", label= "0")
    plt.legend()
    plt.subplot(2,2,4)
    plt.plot(range(1, totIter), diffPrev[1:], label = "Diff from Prev")
    plt.hlines(0, 0,totIter, "k", label= "0")
    plt.legend()
    if sb is not None:
        os.makedirs(f"objOut/png/{sb}", exist_ok=True)
        plt.savefig(f"objOut/png/{sb}/{name}_{paramas}.png")
    else:
        plt.savefig(f"objOut/png/{name}_{paramas}.png")
    if not close:
        plt.close()

parser = ap.ArgumentParser()
parser.add_argument("path", help="The path of the directory with the policies to plot")
parser.add_argument("--subFolder")
parser.add_argument("--GPU", help="Which GPU to use, if not specified will use CPU", type = int)
args = parser.parse_args()
parentDir = args.path
GPU = args.GPU
subFolder = args.subFolder
reg = re.compile("results/TD_Lambda(_Vanilla)?/M([0-9])/lambda_actor([01]\\.[0-9]*)/lambda_critic([01]\\.[0-9]*)/alphaActor_([0-9]+\\.[0-9]*)(?:_Scheduled)?_alphaCritic_([0-9]+\\.[0-9]*)(?:_Scheduled)?/(.*)/")
gr = reg.match(parentDir).groups()
M = int(gr[1])

rho = np.zeros(SC * M)
rho[:cols] = (1-dataC[0,:cols])/np.sum((1-dataC[0,:cols]))
R = np.ones(SC*M) * -(1 - gamma)
for s in range(SC):
    r, c = s // 92, s % 92 
    if (r - rSource) ** 2 + (c -cSource) **2 < find_range**2:
        R[s::SC] = 0
calc_V_eta = sparse_T_CPU
xp = np
if GPU is not None:
    cp.cuda.Device(GPU).use()
    R = cp.asarray(R)
    rho = cp.asarray(rho)
    calc_V_eta = ggTrasfer
    xp = cp

Vopt = xp.load(f"celaniData/V{M}_opt.npy")

ls = glob.glob(parentDir+"Actors/theta*")
totIter = len(ls) - (2 if parentDir + "Actors/thetaActorCriticFInale.npy" in ls else 1)
minTh = int(re.search("theta([0-9]+).npy", min(ls)).group(1))
# print(f"{gr}")
vanilla = gr[0] is not None
print(f"Actor Lambda {gr[2]}; M {gr[1]}; Lr {gr[4]}\nCritic Lambda {gr[3]}; Lr {gr[5]}; {gr[6]}\nVanilla: {vanilla}")

start = 0
if os.path.exists(parentDir + "Obj"):
    normDiff = np.load(parentDir + "Obj/normDiff.npy")
    start = normDiff.shape[0]
    obj = np.load(parentDir + "Obj/obj.npy")
    obj.resize(totIter+1)
    thetas = np.load(parentDir + "Obj/thetas.npy")
    thetas.resize(totIter+1)
    normDiff.resize(totIter)
    diffFromOpt = np.load(parentDir + "Obj/diffFromOpt.npy")
    diffFromOpt.resize(totIter)
    diffPrev = np.load(parentDir + "Obj/diffPrev.npy")
    diffPrev.resize(totIter)
else:
    thetas = np.zeros(totIter+1)
    normDiff = np.zeros(totIter)
    diffFromOpt = np.zeros(totIter)
    obj = np.zeros(totIter +1)
    diffPrev = np.zeros(totIter)
    th = np.load(parentDir + f"Actors/thetaSTART.npy")
    thetas[0] = np.linalg.norm(th)
    # T = mb.prova(softmax(th, axis = 2), dataC, rSource, cSource, find_range, M)
    # AV = sparse.eye(SC * M, format="csr") - gamma * T
    # trueV = sparse.linalg.spsolve(AV, R)
    trueV, _ = calc_V_eta(softmax(th, axis = 2), dataC, rSource, cSource, find_range, R, rho, M)
    obj[0] = xp.dot(trueV, rho)

print(f"Starting from {start} To do {totIter - start}")
s = time.perf_counter()
prevTime = s

for i in range(start, totIter):
    th = np.load(parentDir + f"Actors/theta{minTh + i*1000}.npy")
    thetas[i+1] = np.linalg.norm(th)
    # T = mb.prova(softmax(th, axis = 2), dataC, rSource, cSource, find_range, M)
    # AV = sparse.eye(SC * M, format="csr") - gamma * T
    # trueV = sparse.linalg.spsolve(AV, R)
    trueV, _ = calc_V_eta(softmax(th, axis = 2), dataC, rSource, cSource, find_range, R, rho, M)
    lambdaV = xp.load(parentDir + f"Critics/critic{minTh + i*1000}.npy")
    obj[i+1] = xp.dot(trueV, rho)
    normDiff[i] = xp.linalg.norm(trueV - lambdaV, 2)
    diffFromOpt[i] = xp.linalg.norm(lambdaV - Vopt, 2)
    if i > start:
        diffPrev[i] = xp.linalg.norm(lambdaV - prev)
    elif start > 0:
        diffPrev[i] = xp.linalg.norm(lambdaV - xp.load(parentDir + f"Critics/critic{minTh + (i-1)*1000}.npy"))

    prev = lambdaV
    if i % 10 == 0:
        t = time.perf_counter()
        print(i, "done", t-prevTime, flush=True)
        prevTime = t
e = time.perf_counter()
totalTime(e, s)
plot_and_save(totIter, thetas, obj, normDiff, diffFromOpt,diffPrev, gr[1:6],gr[6],M, subFolder,vanilla, close=True )    

os.makedirs(parentDir +"Obj", exist_ok=True)
np.save(parentDir+"Obj/obj.npy",obj)
np.save(parentDir+"Obj/normDiff.npy",normDiff)
np.save(parentDir+"Obj/diffFromOpt.npy",diffFromOpt)
np.save(parentDir+"Obj/diffPrev.npy",diffPrev)
np.save(parentDir+"Obj/thetas.npy",thetas)