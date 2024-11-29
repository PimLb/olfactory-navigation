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
parser.add_argument("actor_lr", type=float, help="the learning rate for the actor")
parser.add_argument("critic_lr", type=float, help="the learning rate for the critic")
parser.add_argument("lambda_actor", type=float, help="the trace decay parameter for the actor")
parser.add_argument("lambda_critic", type=float, help="the trace decay parameter for the critic")
parser.add_argument("episodes", type=int, help="The final episode number. If --iterationStart is specified, the number of episode ran will be episodes - iterationStart")
parser.add_argument("-n", "--name", help="subfolder name in which to save the results")
parser.add_argument("-t","--thetaStart", help="the path to a .npy file containing the starting values of theta")
parser.add_argument("-v","--vStart", help="the path to a .npy file containing the starting values of V")
parser.add_argument("--scheduleActorLR", help="wheter or not to decrease the actor learning rate", action="store_true")
parser.add_argument("--scheduleCriticLR", help="wheter or not to decrease the critic learning rate", action="store_true")
parser.add_argument("--iterationStart", help="if specified, is the iteration to start from, useful to continue stopped run. If any scheduling is on, it will restart from that too.", type=int)
parser.add_argument("--subMax", help="if specified, every iteration from each row of theta will be subtracted its maximum", action="store_true")
parser.add_argument("--toClip", help="if specified, theta's entries will be clipped in the [-20, 0] interval", action="store_true")
args = parser.parse_args()
actor_lr = args.actor_lr
critic_lr = args.critic_lr
lambda_actor = args.lambda_actor
lambda_critic = args.lambda_critic
numberEpisodes = args.episodes
scheduleActor = args.scheduleActorLR
scheduleCritic = args.scheduleCriticLR
subMax = args.subMax
toClip = args.toClip
itStart = args.iterationStart if args.iterationStart else 0


saveDir = f"results/TD_Lambda/M1/lambda_actor{lambda_actor}/lambda_critic{lambda_critic}/alphaActor_{actor_lr}_"
if scheduleActor:
    saveDir += "Scheduled_"
saveDir += f"alphaCritic_{critic_lr}"
if scheduleCritic:
    saveDir += "_Scheduled"
saveDir += "/"
if args.name is not None:
    saveDir += f"{args.name}_"
saveDir += f"episodes_{numberEpisodes}"


critDir = os.path.join(saveDir, "Critics")
actDir = os.path.join(saveDir, "Actors")
if args.thetaStart is not None:
    theta = np.load(args.thetaStart)
else:
    theta = (np.random.rand(2, 1, 4) -0.5) * 0.5
    # theta[1, :, 0] += 0.5
    # theta[1, :, 2] += 0.5

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
os.makedirs(critDir)
os.makedirs(actDir)
ouput = open(os.path.join(saveDir, "_results.out"), "w")
s = time.perf_counter()
print(f" Startinng {numberEpisodes} episodes at {time.ctime()}",file=ouput)
print("Starting pi:", pi,file=ouput, flush=True)
np.save(os.path.join(actDir, "thetaSTART.npy"), theta)
t = 1
for i in range(itStart, numberEpisodes):
    start = np.random.choice(range(SC), p = rho)
    discount = 1
    curState = start
    curStep = 0
    zCritic = np.zeros_like(V)
    zActor = np.zeros_like(theta)

    cur_actor_lr = actor_lr / np.sqrt(i+1) if scheduleActor else actor_lr
    cur_critic_lr = critic_lr / np.sqrt(i+1) if scheduleCritic else critic_lr
    while( not isEnd(curState) and curStep < maxStepsPerEpisode):


        obs = np.random.choice(2, p = dataC[:, curState])
        action = np.random.choice(4, p= pi[obs, 0])
        newState = takeAction(curState, action)
        reward = -(1 - gamma) if not isEnd(newState) else 0
        # print(curState, obs, action, newState, reward)
        tdError = reward + gamma * V[newState] - V[curState]
        zCritic = gamma * lambda_critic * zCritic 
        zCritic[curState] += 1 # Caso speciale per V tabulare. Credo sia giusto
        # Caso speciale per Natural Gradient e softmax. Credo sia giusto
        zActor = gamma * lambda_actor * zActor
        zActor[obs, 0, action] += discount / pi[obs, 0, action]

        theta += cur_actor_lr * tdError * zActor
        if subMax:
            theta -= np.max(theta, axis =2 , keepdims=True)
        if toClip:
            theta = np.clip(theta, -20, 0)
        V += cur_critic_lr * tdError * zCritic 
        discount *= gamma
        pi = softmax(theta, axis = 2)
        curState = newState
        curStep += 1
        t += 1
        # print(f"Step {curStep}/{maxStepsPerEpisode} of episode {i}/{numberEpisodes} took {e-s} seconds")
    if (i +1) % 1000 == 0:
        print(f"Episode {i+1} done at {time.ctime()}",file=ouput)
        print(f"PI at episode {i+1}: {pi}", flush=True,file=ouput)
        np.save(os.path.join(actDir , f"theta{i+1}.npy"), theta)
        np.save(os.path.join(critDir , f"critic{i+1}.npy"), V)
        if np.any(np.isclose(pi[0,0], 1)):
            print("Terminated", file=ouput)
            sys.exit()
    # if(isEnd(curState)):
    #     print(f"Episode {i} has reached the source in {curStep} steps", file=ouput, flush=True)
    #     print(f"Episode {i} has reached the source in {curStep} steps", flush=True)

e = time.perf_counter()
print("Learned pi:", pi,file=ouput)
np.save(os.path.join(actDir,"thetaActorCriticFInale.npy"), theta)
totalTime(e, s, ouput)
