import numpy as np
from matplotlib import pyplot as plt
import cupy as cp
from scipy.special import softmax as softmax
import scipy.sparse as sparse
import cupyx.scipy.sparse as cSparse
from itertools import chain
from itertools import repeat as itReapeat
import time
import sys
import os
# from cupyx.scipy.special import softmax as softmax
# from utils import get_value

#Azioni sono [North, East, South, West]
SC = 92 * 131 # Number of States

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


def invT(pi, dataC, rSource, cSource, find_range, R, rho):
    T = get_Transition_Matrix_vect(pi, dataC,rSource, cSource, find_range)
    Tinv = cp.linalg.inv( cp.eye(SC) - gamma * cp.asarray(T)).get()
    Vold = np.matmul(Tinv, R)
    eta = np.matmul(rho, Tinv)
    return Vold, eta

def sparse_T_GPU(pi, dataC, rSource, cSource, find_range, R, rho, M):
    T = get_Transition_Matrix_sparse_GPU(pi, dataC, rSource, cSource, find_range, M)
    AV = cSparse.eye(SC *M, format="csr") - gamma * T
    Aeta = cSparse.eye(SC *M, format="csr") - gamma * T.T
    V = cSparse.linalg.spsolve(AV, R)
    eta = cSparse.linalg.spsolve(Aeta, rho)
    return V, eta

# Se ho fatto bene i test, questo dovrebbe essere il più veloce di tutti. Continuo a non capire perchè usare la GPU non mi aiuta
def sparse_T_CPU(pi, dataC, rSource, cSource, find_range, R, rho, M):
    T = get_Transition_Matrix_sparse_CPU(pi, dataC, rSource, cSource, find_range, M)
    AV = sparse.eye(SC * M, format="csr") - gamma * T
    Aeta = sparse.eye(SC * M, format="csr") - gamma * T.T
    V = sparse.linalg.spsolve(AV, R)
    eta = sparse.linalg.spsolve(Aeta, rho)
    return V, eta

def get_Transition_Matrix_sparse_GPU(pi, pObs, rSource, cSource, find_range, M):
    rowIdx = cp.array(range(SC*M)).repeat(4*M) # Ogni riga ha 4 azioni possibili per ogni memoria
    colIdx = cp.array(list(chain.from_iterable(map(getReachable, range(SC*M), itReapeat(M))))) # Per ogni stato prendo gli stati raggiungibili e li metto in una lista
    toSum = cp.sum(pi[None, :, :, :].T * pObs[:, :], axis = 2).T.reshape(-1)
    # print(rowIdx.shape, colIdx.shape, toSum.shape, pObs.shape, pi.shape, pObs.shape)
    tempFinal = [s for s in range(SC) if (s // 92 - rSource) ** 2 + (s % 92 -cSource) **2 < find_range**2 ]
    final = tempFinal.copy()
    for i in range(1, M):
        final += [f + i * SC for f in tempFinal]
    # print(final)
    # Sono gli stati finali. Ad ognuno di essi cambio gli stati raggiungibili
    for i in range(len(final)):
        toSum[int(final[i]*4*M):int(final[i]+1) * 4*M] = 0.25 / M # Le coordinate doppie vengono sommate tra loro
        colIdx[final[i]*4*M:(final[i]+1) * 4*M] = final[i]
    return cSparse.csr_matrix((toSum, (rowIdx, colIdx)))

def get_Transition_Matrix_sparse_CPU(pi, pObs, rSource, cSource, find_range, M):
    rowIdx = np.array(range(SC*M)).repeat(4*M) # Ogni riga ha 4 azioni possibili per ogni memoria
    colIdx = np.array(list(chain.from_iterable(map(getReachable, range(SC*M), itReapeat(M))))) # Per ogni stato prendo gli stati raggiungibili e li metto in una lista
    toSum = np.sum(pi[None, :, :, :].T * pObs[:, :], axis = 2).T.reshape(-1)
    # print(rowIdx.shape, colIdx.shape, toSum.shape, pObs.shape, pi.shape, pObs.shape)
    tempFinal = [s for s in range(SC) if (s // 92 - rSource) ** 2 + (s % 92 -cSource) **2 < find_range**2 ]
    final = tempFinal.copy()
    for i in range(1, M):
        final += [f + i * SC for f in tempFinal]
    # print(final)
    # Sono gli stati finali. Ad ognuno di essi cambio gli stati raggiungibili
    for i in range(len(final)):
        toSum[int(final[i]*4*M):int(final[i]+1) * 4*M] = 0.25 / M # Le coordinate doppie vengono sommate tra loro
        colIdx[final[i]*4*M:(final[i]+1) * 4*M] = final[i]
    return sparse.csr_matrix((toSum, (rowIdx, colIdx)))


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


def calc_Q(V, R, M):
    Q = np.zeros((SC, 4))
    RV = R + gamma * V
    toSum = np.array([RV[getReachable(s, M)] for s in range(SC)])
    for a in range(4):
        np.add.at(Q, (np.arange(SC, dtype=int), a), toSum[:, a])
    mask = np.array([(s // 92 - rSource) ** 2 + (s % 92 -cSource) **2 < find_range**2 for s in range(SC)])
    Q[mask] = 0
    return Q

def find_grad(Q, eta, pObs):
    grad = np.zeros((2, 1, 4))
    O, M, AM = grad.shape
    for a in range(4):
        grad[:, 0, a] = np.sum(pObs * eta * Q[:, a], axis=1)
    return grad

if __name__ == "__main__":
    cSource = 45.5
    rSource = 91
    find_range = 1.1
    gamma = 0.99975
    cols = 92
    rows = 131
    maxIt = 35000
    lr = 0.01
    tol = 1e-8
    print("Inizio: ", time.ctime(), flush=True)
    dataFile = sys.argv[1]
    folder = sys.argv[2]
    M = int(sys.argv[3])
    os.makedirs(f"results/modelBased/M{M}/celani/{dataFile}/{folder}", exist_ok=True)
    with cp.cuda.Device(3):
        dataC = np.load(f"celaniData/{dataFile}.npy")
        theta = (np.random.rand(2, M, 4*M) -0.5) * 0.5
        theta[1, :, 0::4] += 0.5
        theta[1, :, 2::4] += 0.5 # Bias on upwind and downwind directions
        # theta = np.load("results/modelBased/M1/celani/fine5/alpha1e-3_noRescaled_noSubtract/theta_START.npy")
        np.save(f"results/modelBased/M{M}/celani/{dataFile}/{folder}/theta_START", theta)
        pi = softmax(theta, axis = 2)
        print("THETA_START", theta)
        print("PISTART", pi)
        print(flush=True)

        R = np.ones(SC*M) * -(1 - gamma)
        for s in range(SC):
            r, c = s // 92, s % 92 
            if (r - rSource) ** 2 + (c -cSource) **2 < find_range**2:
                R[s::SC] = 0
        # RGPU = np.asarray(R)
        rho = np.zeros(SC*M)
        rho[:cols] = (1-dataC[0,:cols])/np.sum((1-dataC[0,:cols])) # Copiato dal loro
#         T = get_Transition_Matrix_vect(pi, dataC,rSource, cSource, find_range)
#         Tinv = cp.linalg.inv( cp.eye(SC) - gamma * cp.asarray(T)).get()
#         Vold = np.matmul(Tinv, R)
# #        oldObjective = np.matmul(Vold, rho)
#         eta = np.matmul(rho, Tinv)
        Vold, eta = sparse_T_CPU(pi, dataC, rSource, cSource, find_range, R, rho, M)
        Vconv = Vold.copy()
        Q = calc_Q(Vold, R)
#        oldValueLoro = get_value(Q.reshape(-1), pi, dataC, 131*92, rho)
        # np.save("V", Vold)
        # np.save("eta", eta)
        # np.save("Q", Q)
        converged = False
        for i in range(maxIt):
            s = time.perf_counter()
            grad = find_grad(Q, eta, dataC)
            grad -= np.max(grad, axis = 2, keepdims=True)
            # print(f"GRAD {i} ", grad)
            # print(f"Meano GRAD {i} ", grad - np.mean(grad, axis = 2, keepdims=True))
            theta += lr / (np.max(np.abs(grad))) * grad
            # theta += lr * grad
            print(f"Grad {i}: ", grad)
            # theta -= np.mean(theta, axis = 2, keepdims=True)
            # print("THETA dopo mean", theta)
            pi = softmax(theta, axis = 2)
            # print("PI", pi)
            # T = get_Transition_Matrix_vect(pi, dataC, rSource, cSource, find_range)
            # Tinv = cp.linalg.inv( cp.eye(SC) - gamma * cp.asarray(T)).get()
            # V = np.matmul(Tinv, R)
            # Vold = V
            # eta = np.matmul(rho, Tinv)
            V, eta = sparse_T_CPU(pi, dataC, rSource, cSource, find_range, R, rho, M)
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
                    np.save(f"results/modelBased/M1/celani/{dataFile}/{folder}/V_Conv{i+1}", V)
                    print("Theta END: ", theta, flush=True)
                    print("PI END: ", pi, flush=True)
                    converged = True
                    break
                Vconv = V
#                oldObjective = objective
#                oldValueLoro = valueLoro
                np.save(f"results/modelBased/M1/celani/{dataFile}/{folder}/theta_{i+1}", theta)
                np.save(f"results/modelBased/M1/celani/{dataFile}/{folder}/V_{i+1}", V)
                print("last iteration took ", e-s, " seconds", flush=True)
        if not converged:
            np.save(f"results/modelBased/M1/celani/{dataFile}/{folder}/theta_{maxIt}", theta)
            np.save(f"results/modelBased/M1/celani/{dataFile}/{folder}/V_{maxIt}", V)
            print("Theta END not conv :", theta)
            print("PI END not COnv", pi, flush = True)
