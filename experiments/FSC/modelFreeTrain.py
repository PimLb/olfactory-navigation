import numpy as np
from matplotlib import pyplot as plt
import cupy as cp
from scipy.special import softmax as softmax
import multiprocessing as mp
import time
import os
import sys

SC = 92 * 131 # Number of States
cSource = 45.5 # Source coordinates
rSource = 91
cols = 92
rows = 131
find_range = 1.1 # Source radius
maxIt = 50000
gamma = 0.99975
lr = 0.01
tol = 1e-8
reward = -(1 -gamma)
MC_samples = 10
h = 1e-3
ActionDict = np.asarray([
            [-1,  0], # North
            [ 0,  1], # East
            [ 1,  0], # South
            [ 0, -1]  # West
        ])

def manhattan(pos):
    source = np.array([91, 45.5])
    manhattan_distance = np.sum(np.abs( pos - source))    
    distance_to_border = max(0, manhattan_distance - 1.1)
    return np.ceil(distance_to_border)

def isEnd(s):
    return (s[0] - rSource) ** 2 + (s[1] -cSource) **2 < find_range**2

def get_observation(s, pObs, gen):
    singleD = s[0] * cols + s[1]
    return gen.choice(2, p=pObs[:, singleD])

def choose_action(pi, o, gen):
    return gen.choice(4, p = pi[o, 0])

def move(states, actions):
    ret = states + actions
    ret[:, 0] = np.clip(ret[:, 0], 0, rows -1)
    ret[:, 1] = np.clip(ret[:, 1], 0, cols -1)
    return ret

def getTrajectories(start, pObs, max_MC_steps, pi, rng):
    num = start.shape[0]
    mapObs = [pObs for i in range(num)]
    step = start.astype(int)
    done = np.array([a for a in map(isEnd, step)])
    obs = np.array([a for a in map(get_observation, step, mapObs, [rng for i in range(num)])]) # FOrse cambiare generatori anche qui
    t = 0
    actions = np.zeros((num, 2), dtype=int)
    rewards = np.zeros(num)
    d0s = np.zeros(num)
    curD = np.zeros(num)
    nextD = np.zeros(num)
    for i in range(num):
        d0s[i] = manhattan(start[i])
    while np.any(~done) and t < max_MC_steps:
        for i in range(num): # TODO: sequenziale, sarebbe da parallelizare
            if not done[i]:
                a = choose_action(pi, obs[i], rng)
                actions[i] = ActionDict[a]
            else:
                actions[i] = np.array([0, 0])
            curD[i] = -manhattan(step[i]) / d0s[i]
        t+= 1
        step = move(step, actions)
        for i in range(num):
            nextD[i] = -manhattan(step[i]) / d0s[i]
        obs = np.array([a for a in map(get_observation, step, mapObs, [rng for i in range(num)])]) # Forse anche qui
        done = np.array([a for a in map(isEnd, step)])
        # There is a missing gamma on this line. Does it matter?
        # rewards[~done] += reward
        rewards[~done] += gamma**t * (reward +gamma * nextD[~done] - curD[~done])
    return np.mean(rewards)

if __name__ == "__main__":
    MC_max_steps = int(sys.argv[2])
    try:
        startingRow = int(sys.argv[3])
    except IndexError:
        startingRow = 0
    baseDir = f"results/modelFree/M1/row{startingRow}/maxIt_{maxIt}/{MC_max_steps}/{sys.argv[1]}"
    os.makedirs(baseDir, exist_ok=True)
    output = open(os.path.join(baseDir, "output.out"), "w")
    print("Inizio: ", time.ctime(), flush=True, file=output)
    dataC = np.load("celaniData/fine5.npy")
    theta = (np.random.rand(2, 1, 4) -0.5) * 0.5
    theta[1, :, 0] += 0.5
    theta[1, :, 2] += 0.5 # Bias on upwind and downwind directions
    theta -= np.max(theta, axis=2, keepdims=True)
    # theta = np.ones((2,1,4))
    # theta[:, :, 2] = 10
    print("PI iniziale: ", softmax(theta, axis = 2), file=output)
    grad = np.ones_like(theta)
    rho = np.zeros(SC)
    rho[startingRow * cols:cols *(startingRow+1)] = (1-dataC[0,startingRow * cols:cols *(startingRow+1)])/np.sum((1-dataC[0,startingRow * cols:cols *(startingRow+1)])) # Copiato dal loro
    np.save(f"results/modelFree/M1/row{startingRow}/maxIt_{maxIt}/{MC_max_steps}/{sys.argv[1]}/theta_START", theta)
    i = 0
    gens = [np.random.default_rng() for a in range(MC_samples)]
    while i < maxIt:
        s = time.perf_counter()
        pi = softmax(theta, axis = 2)
        tmp = np.random.choice(range(SC), size=MC_samples, replace=True, p = rho)
        # tmp[0] = 45
        starts = np.array(np.unravel_index(tmp, (131, 92))).T
        if i == 0:
            print("Starting point at iteration 0", starts, flush=True, file=output)
        grad = np.zeros_like(theta)
        piCopies = [pi]
        for o in range(2):
            for a in range(4):
                th = theta.copy()
                th[o, 0, a] += h # TODO: Troppo piccolo per fare differenza sulla policy spesso
                piCopies.append(softmax(th, axis=2))
        args = [(starts, dataC, MC_max_steps, p, gens[id]) for id,p in enumerate(piCopies)]
        with mp.Pool(len(piCopies)) as p:
            rewardList = p.starmap(getTrajectories, args)
        for o in range(2):
            for a in range(4):
                grad[o, 0, a] = (rewardList[a + o * 4 +1] - rewardList[0]) / h

        theta += lr * grad
        theta -= np.max(theta, axis=2, keepdims=True)
        pi = softmax(theta, axis = 2)

        # if np.any(grad != 0):
        #     print("Grad not zero at iteration ", i, flush=True, file=output)
        i+=1
        if (i +1) % 1000 == 0:
            print(f"Episode {i+1}:", pi, file=output)
            np.save(f"results/modelFree/M1/row{startingRow}/maxIt_{maxIt}/{MC_max_steps}/{sys.argv[1]}/theta_{i+1}", theta)
    print(pi, file=output)
    print("Fine: ", time.ctime(), file=output)
        
