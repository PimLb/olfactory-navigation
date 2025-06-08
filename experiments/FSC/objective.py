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

def plot_and_save(totIter, thetas, obj, normDiff, diffFromOpt, diffPrev, paramas, name, M, sb, close = False):
    color = "orange"
    style = "--"
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"{name}\n Actor Lambda {paramas[1]}; Lr {paramas[3]}\nCritic Lambda {paramas[2]}; Lr {paramas[4]}; M {paramas[0]}")
    plt.subplot(2,2, 1)
    plt.plot(range(totIter), thetas, label = "Theta Norm")
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
    plt.plot(range(totIter), obj,marker=None if obj.shape[0] > 5 else "x",markersize = 8, label = "Objective")
    plt.grid()
    # plt.ylim(-1, -0.475)
    plt.legend()
    plt.subplot(2,2,3)
    plt.plot(range(totIter), normDiff, label = "Diff from True")
    plt.plot(range(totIter), diffFromOpt, label = "Diff from Optimal")
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
parser.add_argument("M", help="The memories of the FSC", type = int)
parser.add_argument("--subFolder")
parser.add_argument("--GPU", help="Which GPU to use, if not specified will use CPU", type = int)
args = parser.parse_args()
parentDir = args.path
M = args.M
GPU = args.GPU
subFolder = args.subFolder

rho = np.zeros(SC * M)
rho[:cols] = (1-dataC[0,:cols])/np.sum((1-dataC[0,:cols]))
R = np.ones(SC*M) * -(1 - gamma)
for s in range(SC):
    r, c = s // 92, s % 92 
    if (r - rSource) ** 2 + (c -cSource) **2 < find_range**2:
        R[s::SC] = 0
calc_V_eta = sparse_T_CPU
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
reg = re.compile("results/TD_Lambda/M([0-9])/lambda_actor([01]\\.[0-9]*)/lambda_critic([01]\\.[0-9]*)/alphaActor_([0-9]+\\.[0-9]*)(?:_Scheduled)?_alphaCritic_([0-9]+\\.[0-9]*)(?:_Scheduled)?/(.*)/")
gr = reg.match(parentDir).groups()
# print(f"{gr}")
print(f"Actor Lambda {gr[1]}; M {gr[0]}; Lr {gr[3]}\nCritic Lambda {gr[2]}; Lr {gr[4]}; {gr[5]}")

start = 0
if os.path.exists(parentDir + "Obj"):
    obj = np.load(parentDir + "Obj/obj.npy")
    start = obj.shape[0]
    obj.resize(totIter)
    thetas = np.load(parentDir + "Obj/thetas.npy")
    thetas.resize(totIter)
    normDiff = np.load(parentDir + "Obj/normDiff.npy")
    normDiff.resize(totIter)
    diffFromOpt = np.load(parentDir + "Obj/diffFromOpt.npy")
    diffFromOpt.resize(totIter)
    diffPrev = np.load(parentDir + "Obj/diffPrev.npy")
    diffPrev.resize(totIter)
else:
    thetas = np.zeros(totIter)
    normDiff = np.zeros(totIter)
    diffFromOpt = np.zeros(totIter)
    obj = np.zeros(totIter)
    diffPrev = np.zeros(totIter)

print(f"Starting from {start} To do {totIter - start}")
s = time.perf_counter()
prevTime = s
for i in range(start, totIter):
    th = np.load(parentDir + f"Actors/theta{minTh + i*1000}.npy")
    thetas[i] = np.linalg.norm(th)
    # T = mb.prova(softmax(th, axis = 2), dataC, rSource, cSource, find_range, M)
    # AV = sparse.eye(SC * M, format="csr") - gamma * T
    # trueV = sparse.linalg.spsolve(AV, R)
    trueV, _ = calc_V_eta(softmax(th, axis = 2), dataC, rSource, cSource, find_range, R, rho, M)
    lambdaV = xp.load(parentDir + f"Critics/critic{minTh + i*1000}.npy")
    obj[i] = xp.dot(trueV, rho)
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
plot_and_save(totIter, thetas, obj, normDiff, diffFromOpt,diffPrev, gr[:5],gr[5],M, subFolder, close=True )    

os.makedirs(parentDir +"Obj", exist_ok=True)
np.save(parentDir+"Obj/obj.npy",obj)
np.save(parentDir+"Obj/normDiff.npy",normDiff)
np.save(parentDir+"Obj/diffFromOpt.npy",diffFromOpt)
np.save(parentDir+"Obj/diffPrev.npy",diffPrev)
np.save(parentDir+"Obj/thetas.npy",thetas)