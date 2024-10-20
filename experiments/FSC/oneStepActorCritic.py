import numpy as np
from matplotlib import pyplot as plt
import cupy as cp
from scipy.special import softmax as softmax
import multiprocessing as mp
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
ActionDict = np.asarray([
            [-1,  0], # North
            [ 0,  1], # East
            [ 1,  0], # South
            [ 0, -1]  # West
        ])
dataC = np.load("celaniData/fine5.npy")
rho = np.zeros(SC)
rho[:cols] = (1-dataC[0,:cols])/np.sum((1-dataC[0,:cols]))

# Algo Parameters
numberEpisodes = 100000
maxStepsPerEpisode = 10000
actor_lr = 0.001
critic_lr = 0.001

def isEnd(s):
    r, c = s // cols, s % cols
    return (r - rSource) ** 2 + (c -cSource) **2 < find_range**2

def takeAction(s, a):
    r, c = s // cols, s % cols
    action = ActionDict[a]
    rNew = r + action[0]
    cNew = c + action[1]
    r = rNew if rNew >= 0 and rNew < 131 else r
    c = cNew if cNew >= 0 and cNew < 92 else c
    return r * 92 + c


parser = ap.ArgumentParser()
parser.add_argument("actor_lr", type=float, help="the learning rate for the actor")
parser.add_argument("critic_lr", type=float, help="the learning rate for the critic")
parser.add_argument("episodes", type=int, help="How many episodes to run")
parser.add_argument("-n", "--name", help="subfolder name in which to save the results")
parser.add_argument("-t","--thetaStart", help="the path to a .npy file containing the starting values of theta")
parser.add_argument("-v","--vStart", help="the path to a .npy file containing the starting values of V")
args = parser.parse_args()
print(args)
actor_lr = args.actor_lr
critic_lr = args.critic_lr
numberEpisodes = args.episodes

if args.name is not None:
    saveDir = os.path.join(f"results/oneStepAC/M1/alphaCritic_{critic_lr}_alphaActor_{actor_lr}", f"{args.name}_episodes_{numberEpisodes}")
else:
    saveDir = os.path.join(f"results/oneStepAC/M1/alphaCritic_{critic_lr}_alphaActor_{actor_lr}", f"episodes_{numberEpisodes}")

if args.thetaStart is not None:
    theta = np.load(args.thetaStart)
else:
    theta = (np.random.rand(2, 1, 4) -0.5) * 0.5
    theta[1, :, 0] += 0.5
    theta[1, :, 2] += 0.5

if args.vStart is not None:
    V = np.load(args.vStart)
else:
    V = np.zeros(SC)
    # V = np.ones(SC) * -1
    # mask = np.array([isEnd(s) for s in range(SC)])
    # V[mask] = 0
    # print(np.argwhere(V == 0), flush=True)
    # V = np.load("results/modelBased/M1/celani/fine5/alpha1e-2_Rescaled_Subtract/V_Conv7000.npy")


pi = softmax(theta, axis=2)
os.makedirs(saveDir)
ouput = open(os.path.join(saveDir, "results.out"), "w")
s = time.perf_counter()
print(f" Startinng {numberEpisodes} episodes at {time.ctime()}",file=ouput)
print("Starting pi:", pi,file=ouput, flush=True)
np.save(os.path.join(saveDir, "thetaSTART.npy"), theta)
for i in range(numberEpisodes):
    start = np.random.choice(range(SC), p = rho)
    eligibility = 1
    curState = start
    curStep = 0
    while( not isEnd(curState) and curStep < maxStepsPerEpisode):
        obs = np.random.choice(2, p = dataC[:, curState])
        action = np.random.choice(4, p= pi[obs, 0])
        newState = takeAction(curState, action)
        reward = -(1 - gamma) if not isEnd(newState) else 0
        # print(curState, obs, action, newState, reward)
        tdError = reward + gamma * V[newState] - V[curState]
        #print(f"Ep {i} state {curState} O: {obs}, A:{action}, TD error: {tdError}", flush=True)
        # Caso speciale per Natural Gradient e softmax. Credo sia giusto
        theta[obs, 0, action] = theta[obs, 0, action] + actor_lr * tdError * eligibility / pi[obs, 0, action] 
        V[curState] += critic_lr * tdError # Caso speciale per V tabulare. Credo sia giusto
        eligibility *= gamma
        pi = softmax(theta, axis = 2)
        curState = newState
        curStep += 1
        # print(f"Step {curStep}/{maxStepsPerEpisode} of episode {i}/{numberEpisodes} took {e-s} seconds")
    if (i +1) % 1000 == 0:
        print(f"Episode {i+1} done at {time.ctime()}",file=ouput)
        print(f"PI at episode {i+1}: {pi}", flush=True,file=ouput)
        np.save(os.path.join(saveDir , f"theta{i+1}.npy"), theta)
        np.save(os.path.join(saveDir , f"critic{i+1}.npy"), V)
    # if(isEnd(curState)):
    #     print(f"Episode {i} has reached the source in {curStep} steps", file=ouput, flush=True)
    #     print(f"Episode {i} has reached the source in {curStep} steps", flush=True)

e = time.perf_counter()
print(f"{numberEpisodes} episodes done in {e -s } seconds, at {time.ctime()}",file=ouput)
print("Learned pi:", pi,file=ouput)
np.save(os.path.join(saveDir,"thetaActorCriticFInale.npy"), theta)
