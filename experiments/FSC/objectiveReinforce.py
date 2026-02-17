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

def plot_and_save(totIter, obj, paramas, sb, vanilla):
    color = "orange"
    style = "--"
    plt.figure(figsize=(15, 10))
    M, lr, tmp = paramas
    l = tmp.split("_")
    n, episodes = l[:-1], l[-1]
    name = f"{n[0]}"
    for i in range(1, len(n)):
        name += f"_{n[i]}"
    M = int(M)
    plt.suptitle(f"{name}\n Lr {lr} M {M}" + ("\n Vanilla" if vanilla else ""))
    name += f"_{lr}" + "_vanilla" if vanilla else "_natural"
    ticks = [0, -0.1, -0.3, -0.4,-0.485, -0.6, -0.7, -0.8, -0.9, -1]
    if M >= 4:
        plt.hlines(-0.098, 0,totIter, "orange", label = f"Optimal M4")
        ticks += [-0.098]
        ticks.remove(-0.1)
    if M >= 3:
        plt.hlines(-0.13895278486341234, 0,totIter, "r", label = f"Optimal M3")
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
    if sb is not None:
        os.makedirs(f"objOut/reinforce/{sb}", exist_ok=True)
        plt.savefig(f"objOut/reinforce/{sb}/{name}_{lr}_{episodes}.png")
    else:
        plt.savefig(f"objOut/reinforce/{name}.png")

parser = ap.ArgumentParser()
parser.add_argument("path", help="The path of the directory with the policies to plot")
parser.add_argument("--subFolder")
parser.add_argument("--GPU", help="Which GPU to use, if not specified will use CPU", type = int)
args = parser.parse_args()
parentDir = args.path
GPU = args.GPU
subFolder = args.subFolder
reg = re.compile(".*/(vanilla|natural)?/M([0-9])/lr_([0-9]+\\.[0-9]*|[0-9]e-[0-9]+)(?:_scheduled)?/(.*)/")
gr = reg.match(parentDir).groups()
M = int(gr[1])

ls = glob.glob(parentDir+"thetas/theta*")
totIter = len(ls) - (2 if parentDir + "thetas/thetaFinal.npy" in ls else 1)
if totIter <= 0:
    print("Nothing to do")
    sys.exit()
minTh = int(re.search("theta([0-9]+).npy", min(ls)).group(1))
# print(f"{gr}")
vanilla = gr[0] == "vanilla"
print(f"M {gr[1]}; Lr {gr[2]} {gr[3]}\nVanilla: {vanilla}")

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

start = 0
if os.path.exists(parentDir + "Obj"):
    obj = np.load(parentDir + "Obj/obj.npy")
    start = obj.shape[0] -1
    obj.resize(totIter+1)
else:
    obj = np.zeros(totIter +1)
    th = np.load(parentDir + f"thetas/thetaStart.npy")
    # T = mb.prova(softmax(th, axis = 2), dataC, rSource, cSource, find_range, M)
    # AV = sparse.eye(SC * M, format="csr") - gamma * T
    # trueV = sparse.linalg.spsolve(AV, R)
    trueV, _ = calc_V_eta(softmax(th, axis = 2), dataC, rSource, cSource, find_range, R, rho, M)
    obj[0] = xp.dot(trueV, rho)

# if totIter -start <= 0:
#     print("Nothing to do")
#     sys.exit()


print(f"Starting from {start} To do {totIter - start}")
s = time.perf_counter()
prevTime = s

for i in range(start, totIter):
    try:
        th = np.load(parentDir + f"thetas/theta{minTh + i*1000}.npy")
        trueV, _ = calc_V_eta(softmax(th, axis = 2), dataC, rSource, cSource, find_range, R, rho, M)
        obj[i+1] = xp.dot(trueV, rho)
    except FileNotFoundError:
        obj[i+1] = -1
    if i % 10 == 0:
        t = time.perf_counter()
        print(i, "done", t-prevTime, flush=True)
        prevTime = t
e = time.perf_counter()
totalTime(e, s)

os.makedirs(parentDir +"Obj", exist_ok=True)
np.save(parentDir+"Obj/obj.npy",obj)

plot_and_save(totIter, obj, gr[1:], subFolder,vanilla)    
