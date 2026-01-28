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

def getObservation(state, odor, time, threshold,  maxFrames):
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
parser.add_argument("actor_lr", type=float, help="the learning rate for the actor")
parser.add_argument("critic_lr", type=float, help="the learning rate for the critic")
parser.add_argument("memories", type=int, help="The memories of the FSC")
parser.add_argument("lambda_actor", type=float, help="the trace decay parameter for the actor")
parser.add_argument("lambda_critic", type=float, help="the trace decay parameter for the critic")
parser.add_argument("episodes", type=int, help="The final episode number. If --iterationStart is specified, the number of episode ran will be episodes - iterationStart")
parser.add_argument("-t","--thetaStart", help="the path to a .npy file containing the starting values of theta")
parser.add_argument("-v","--vStart", help="the path to a .npy file containing the starting values of V")
parser.add_argument("--scheduleActorLR", help="wheter or not to decrease the actor learning rate", action="store_true", default=True)
parser.add_argument("--scheduleCriticLR", help="wheter or not to decrease the critic learning rate", action="store_true", default=True)
parser.add_argument("--iterationStart", help="if specified, is the iteration to start from, useful to continue stopped run. If any scheduling is on, it will restart from that too.", type=int)
parser.add_argument("--vanilla", help="whether to use vanilla gradient instead on the natural", action="store_true")
parser.add_argument("--threshold", default=1e-4, help="The concentration of odor above which the agent receives a positive observation", type=float)

args = parser.parse_args()
name = args.name
episodes = args.episodes
dataPath = args.dataPath
actor_lr = args.actor_lr
critic_lr = args.critic_lr
M = args.memories
assert M >= 1
lambda_actor = args.lambda_actor
lambda_critic = args.lambda_critic
scheduleActor = args.scheduleActorLR
scheduleCritic = args.scheduleCriticLR
itStart = args.iterationStart if args.iterationStart else 0
vanilla = args.vanilla
threshold = args.threshold
thetaPath = args.thetaStart

saveDir = f"storage/AC_turbulent/{"vanilla" if vanilla else "natural"}/M{M}/lambda_actor{lambda_actor}/lambda_critic{lambda_critic}/alphaActor_{actor_lr}_"
if scheduleActor:
    saveDir += "Scheduled_"
saveDir += f"alphaCritic_{critic_lr}"
if scheduleCritic:
    saveDir += "_Scheduled"
saveDir += f"/{name}_episodes{episodes}"
critDir = os.path.join(saveDir, "Critics")
actDir = os.path.join(saveDir, "Actors")
os.makedirs(saveDir)
os.makedirs(actDir)
os.makedirs(critDir)
output = open(os.path.join(saveDir, "_results.out"), "w")

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

if thetaPath is not None:
    theta = np.load(thetaPath)
    assert theta.shape == (2,M, 4*M)
else:
    theta = (np.random.rand(2, M, 4*M) -0.5) * 0.5
if args.vStart is not None:
    V = np.load(args.vStart)
    # Should be compatible with V previously saved as a 1-dimensional array
    if V.shape == rows * cols* M:
        V = V.reshape((rows, cols, M))
    assert V.shape == (rows, cols, M)
else:
    V = np.zeros((rows, cols, M))

grad = vanillaGrad if vanilla else natGrad
pi = softmax(theta, axis = 2)
s = time.perf_counter()
print(f" Starting {episodes} episodes at {time.ctime()}",file=output, flush=True)
np.save(os.path.join(actDir, "thetaSTART.npy"), theta)
signal.signal(signal.SIGTERM, handleTERM)
thPrev = theta.copy()
Vprev = V.copy()
errors = []
i = itStart
reached = 0
avgSteps = 0
while i < episodes:
    startCol = np.random.choice(range(cols), p = rho)
    curState = (0, startCol)
    curStateMem = (0, startCol, 0) # Always start from most downwind col and from memory 0
    curMem = 0
    discount = 1
    step = 0
    t = np.random.randint(maxFrames) # Randomize on the starting time of the plume. Will make it harder to overfit the plume dynamic
    zCritic = np.zeros_like(V)
    zActor = np.zeros_like(theta)

    cur_actor_lr = actor_lr * 1000 / (1000 + i) if scheduleActor else actor_lr
    cur_critic_lr = critic_lr * 1000 / (1000 + i**(2/3)) if scheduleCritic else critic_lr

    while( not isEnd(curState, rSource, cSource) and step < maxSteps):

        obs = getObservation(curState, odor, t, threshold, maxFrames)
        action = np.random.choice(4 * M, p= pi[obs, curMem])
        newState, newMem = takeAction(curState, action, rows, cols)
        newStateMem = newState + (newMem,) # Concatenation of tuples. Used to index V 
        reward = -(1 - gamma) if not isEnd(newState, rSource, cSource) else 0
        tdError = reward + gamma * V[newStateMem] - V[curStateMem]
        zCritic = gamma * lambda_critic * zCritic 
        zCritic[curState] += 1 # Caso speciale per V tabulare. Credo sia giusto
        # Caso speciale per Natural Gradient e softmax. Credo sia giusto
        zActor = gamma * lambda_actor * zActor
        zActor += discount * grad(pi, obs, curMem, action)

        theta += cur_actor_lr * tdError * zActor
        theta -= np.max(theta, axis =2 , keepdims=True)
        
        V += cur_critic_lr * tdError * zCritic 
        discount *= gamma
        pi = softmax(theta, axis = 2)
        curState = newState
        step += 1
        t += 1
        curMem = newMem
        curStateMem = newStateMem
    if step < maxSteps:
        reached += 1
        avgSteps += step
    if (i +1) % 1000 == 0:
        np.save(os.path.join(actDir , f"theta{i+1}.npy"), theta)
        np.save(os.path.join(critDir , f"critic{i+1}.npy"), V)
        if np.any(np.isclose(pi[0], 1)):
            errors.append(i)
            print(f"rollback {len(errors)}", file=output, flush=True)
            theta = thPrev.copy()
            V = Vprev.copy()
            i -= 1000
            if len(errors) == 3:
                e = time.perf_counter()
                print(f"Episode {i+1} done at {time.ctime()}; In the last 1000 episodes: {reached/1000:.1%} converged with {avgSteps / reached if reached != 0 else 0.0} avg steps",file=output, flush=True)
                print("Determinism", file=output)
                totalTime(e,s,output)
                sys.exit()
        else:
            print(f"Episode {i+1} done at {time.ctime()}; In the last 1000 episodes: {reached/1000:.1%} converged with {avgSteps / reached if reached != 0 else 0.0} avg steps",file=output, flush=True)
        thPrev = theta.copy()
        Vprev = V.copy()
        reached = 0
        avgSteps = 0
    i += 1

e = time.perf_counter()
np.save(os.path.join(actDir,"thetaActorCriticFInale.npy"), theta)
totalTime(e, s, output)