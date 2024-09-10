import numpy as np
from matplotlib import pyplot as plt
import cupy as cp
from scipy.special import softmax as softmax
import time
import sys
import os
# from cupyx.scipy.special import softmax as softmax
from utils import get_value

#Azioni sono [North, East, South, West]
SC = 92 * 131 # Number of States

def getReachable(s):
    r, c = s // 92, s % 92 
    ret = np.zeros(4, dtype = int)
    ret[0] = s - 92 if r - 1 >= 0 else s
    ret[1] = s + 1  if c + 1 < 92 else s
    ret[2] = s + 92 if r + 1  < 131  else s
    ret[3] = s  -1 if c - 1 >= 0 else s
    return ret

def get_Transition_Matrix_vect(pi, pObs, rSource, cSource, find_range):
    T = np.zeros((SC, SC))
    assert np.allclose(np.sum(pi, axis = 2), 1), "Pi is not a probability"
    #For each state create a list of reachable states
    reachable = np.array([a for a in map(getReachable, np.arange(SC))])
    #We multiply the pi of observation with the probability of observation
    # None is there to make the pi broadcastable on the states. Is first because is transposed
    # It's transposed because the observation are on the first axis of pi
    # Then we sum on the observation axis (1), since the 0 is now the action (plus memory but we don't have it yet), which were last
    # The result has shape (4, number of states), each row is for an action and columns are for states
    # Each entry is then the probability of doing an action in a state
    toSum = np.sum(pi[None, :, 0, :].T * pObs[:, :], axis = 1)
    for a in range(4):
        # Here I sum the probability of doing 
        np.add.at(T, (np.arange(SC, dtype = int), reachable[:, a]), toSum[a, :])
    mask = np.array([(s // 92 - rSource) ** 2 + (s % 92 -cSource) **2 < find_range**2 for s in range(SC)])
    T[mask] = 0
    T[mask, mask] = 1
    return T 


def calc_Q(V, R):
    Q = np.zeros((SC, 4))
    for s in range(SC):
        if (s // 92 - rSource) ** 2 + (s % 92 -cSource) **2 < find_range**2:
            continue
        reach = getReachable(s)
        for a in range(4):
            r = reach[a]
            Q[s, a ] += (R[r] + gamma * V[r])
    return Q

def find_grad(Q, eta, pObs):
    grad = np.zeros((2, 1, 4))
    O, M, AM = grad.shape
    for o in range(O):
        for a in range(4):
            for s in range(SC):
                grad[o, 0, a] += eta[s] * pObs[o, s] * Q[s, a]
    return grad

if __name__ == "__main__":
    cSource = 45.5
    rSource = 91
    find_range = 1.1
    gamma = 0.99975
    cols = 92
    rows = 131
    maxIt = 70000
    lr = 0.01
    tol = 1e-8
    print("Inizio: ", time.ctime(), flush=True)
    dataFile = sys.argv[1]
    folder = sys.argv[2]
    os.makedirs(f"results/modelBased/M1/celani/{dataFile}/{folder}", exist_ok=True)
    with cp.cuda.Device(3):
        dataC = np.load(f"celaniData/{dataFile}.npy")
        theta = (np.random.rand(2, 1, 4) -0.5) * 0.5
        theta[1, :, 0] += 0.5
        theta[1, :, 2] += 0.5 # Bias on upwind and downwind directions
        np.save(f"results/modelBased/M1/celani/{dataFile}/{folder}/theta_START", theta)
        pi = softmax(theta, axis = 2)
        print("THETA_START", theta)
        print("PISTART", pi)
        print(flush=True)

        R = np.ones(SC) * -(1 - gamma)
        for s in range(SC):
            r, c = s // 92, s % 92 
            if (r - rSource) ** 2 + (c -cSource) **2 < find_range**2:
                R[s] = 0
        # RGPU = np.asarray(R)
        rho = np.zeros(SC)
        rho[:cols] = (1-dataC[0,:cols])/np.sum((1-dataC[0,:cols])) # Copiato dal loro
        T = get_Transition_Matrix_vect(pi, dataC,rSource, cSource, find_range)
        Tinv = cp.linalg.inv( cp.eye(SC) - gamma * cp.asarray(T)).get()
        Vold = np.matmul(Tinv, R)
        Vconv = Vold.copy()
#        oldObjective = np.matmul(Vold, rho)
        eta = np.matmul(rho, Tinv)
        Q = calc_Q(Vold, R)
#        oldValueLoro = get_value(Q.reshape(-1), pi, dataC, 131*92, rho)
        # np.save("V", Vold)
        # np.save("eta", eta)
        # np.save("Q", Q)
        converged = False
        for i in range(maxIt):
            s = time.perf_counter()
            grad = find_grad(Q, eta, dataC)
            # grad -= np.mean(grad, axis = 2, keepdims=True)
            # print(f"GRAD {i} ", grad)
            # print(f"Meano GRAD {i} ", grad - np.mean(grad, axis = 2, keepdims=True))
            theta += lr * grad
            theta -= np.mean(theta, axis = 2, keepdims=True)
            # print("THETA dopo mean", theta)
            pi = softmax(theta, axis = 2)
            # print("PI", pi)
            T = get_Transition_Matrix_vect(pi, dataC, rSource, cSource, find_range)
            Tinv = cp.linalg.inv( cp.eye(SC) - gamma * cp.asarray(T)).get()
            V = np.matmul(Tinv, R)
            Vold = V
            eta = np.matmul(rho, Tinv)
            Q = calc_Q(V, R)
            e = time.perf_counter()
            if (i+1) % 1000 == 0:
#                objective = np.matmul(V, rho)
#                valueLoro = get_value(Q.reshape(-1), pi, dataC, 131*92, rho)
                delta = np.max(np.abs(V - Vconv))
#                delta = np.abs((objective - oldObjective) / objective)
#                delta = np.abs((valueLoro - oldValueLoro) / valueLoro)
                print(i+1, time.ctime(), "Delta :", delta)
                if delta < tol:
                    print(f"Converged in {i+1} iterations")
                    np.save(f"results/modelBased/M1/celani/{dataFile}/{folder}/theta_Conv{i+1}", theta)
                    print("Theta END: ", theta, flush=True)
                    print("PI END: ", pi, flush=True)
                    converged = True
                    break
                Vconv = V
#                oldObjective = objective
#                oldValueLoro = valueLoro
                np.save(f"results/modelBased/M1/celani/{dataFile}/{folder}/theta_{i+1}", theta)
                print("last iteration took ", e-s, " seconds", flush=True)
        if not converged:
            np.save(f"results/modelBased/M1/celani/{dataFile}/{folder}/theta_{maxIt}", theta)
            print("Theta END not conv :", theta)
            print("PI END not COnv", pi, flush = True)
