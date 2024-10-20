import numpy as np
from matplotlib import pyplot as plt
import cupy as cp
from scipy.special import softmax as softmax
from cupyx.scipy.special import softmax as softmaxGPU
import scipy.sparse as sparse
import cupyx.scipy.sparse as cSparse
import cupyx.scipy.sparse.linalg as cSparseLA
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


def sparse_T_GPU(pi, dataC, rSource, cSource, find_range, R, rho, M):
    T = get_Transition_Matrix_sparse_GPU(pi, dataC, rSource, cSource, find_range, M)
    # Da qui alla fine della funzione ci mette di più che tutta un iterazione sulla CPU
    AV = cSparse.eye(SC *M, format="csr") - gamma * T
    Aeta = cSparse.eye(SC *M, format="csr") - gamma * T.T
    V = cSparseLA.spsolve(AV, R)
    eta = cSparseLA.spsolve(Aeta, rho)
    return V, eta

# Se ho fatto bene i test, questo dovrebbe essere il più veloce di tutti.
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
    
    # Per ogni stato prendo gli stati raggiungibili e li metto in una lista
    colIdx = np.array(list(chain.from_iterable(map(getReachable, range(SC*M), itReapeat(M))))) 
    
    toSum = np.sum(pi[None, :, :, :].T * pObs[:, :], axis = 2).T.reshape(-1)
    
    tempFinal = [s for s in range(SC) if (s // 92 - rSource) ** 2 + (s % 92 -cSource) **2 < find_range**2 ]
    final = tempFinal.copy()
    for i in range(1, M):
        final += [f + i * SC for f in tempFinal]
    # Ad ognuno degli stati finali cambio gli stati raggiungibili.
    for i in range(len(final)):
        toSum[int(final[i]*4*M):int(final[i]+1) * 4*M] = 0.25 / M # Le coordinate doppie vengono sommate tra loro
        colIdx[final[i]*4*M:(final[i]+1) * 4*M] = final[i] # Da uno stato finale non posso andarmene
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
    Q = np.zeros((SC * M, 4*M))
    RV = R + gamma * V
    toSum = np.array([RV[getReachable(s, M)] for s in range(SC*M)])
    for a in range(4*M):
        np.add.at(Q, (np.arange(SC * M, dtype=int), a), toSum[:, a])
    mask = np.array([((s % SC) // 92 - rSource) ** 2 + (s % 92 -cSource) **2 < find_range**2 for s in range(SC * M)])
    Q[mask] = 0
    return Q

def find_grad(Q, eta, pObs, M):
    grad = np.zeros((2, M, 4 *M))
    tmp = np.tile(pObs, M)
    for a in range(4):
        grad[:, 0, a] = np.sum(tmp * eta * Q[:, a], axis=1)
    return grad

#Risolvere il sistema lineare con CUPY ci mette di più che un'intera iterazione di Gradient Ascent con scipy

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
    dataFile = sys.argv[1]
    folder = sys.argv[2]
    M = int(sys.argv[3])

    # TODO: permettere di cambiarli alla chiamata da linea di comando
    rescale = True
    subtract = True
    saveDir = f"results/modelBased/M{M}/celani/{dataFile}/alpha{lr}"
    if rescale:
        saveDir += "_Rescale"
    else:
        saveDir += "_noRescale"
    if subtract:
        saveDir += "_Subtract"
    else:
        saveDir += "_noSubtract"
    saveDir = os.path.join(saveDir, folder)

    os.makedirs(saveDir, exist_ok=True)
    output = open(os.path.join(saveDir, "output.out"), "w")
    print("Inizio: ", time.ctime(), flush=True, file=output)

    dataC = np.load(f"celaniData/{dataFile}.npy")
    theta = (np.random.rand(2, M, 4*M) -0.5) * 0.5
    theta[1, :, 0::4] += 0.5
    theta[1, :, 2::4] += 0.5 # Bias on upwind and downwind directions
    # theta = np.load("results/modelBased/M1/celani/fine5/alpha1e-3_noRescaled_noSubtract/theta_START.npy")
    np.save(os.path.join(saveDir,"theta_START"), theta)
    pi = softmax(theta, axis = 2)
    print("THETA_START", theta, file=output)
    print("PISTART", pi, file=output)
    print(flush=True, file=output)

    R = np.ones(SC*M) * -(1 - gamma)
    for s in range(SC):
        r, c = s // 92, s % 92 
        if (r - rSource) ** 2 + (c -cSource) **2 < find_range**2:
            R[s::SC] = 0
    rho = np.zeros(SC*M)
    rho[:cols] = (1-dataC[0,:cols])/np.sum((1-dataC[0,:cols])) # Copiato dal loro
    Vold, eta = sparse_T_CPU(pi, dataC, rSource, cSource, find_range, R, rho, M)
    Vconv = Vold.copy()
    Q = calc_Q(Vold, R, M)
    converged = False
    for i in range(maxIt):
        s = time.perf_counter()
        
        grad = find_grad(Q, eta, dataC, M)
        if subtract:
            grad -= np.max(grad, axis = 2, keepdims=True)
        if rescale:  
            theta += lr / (np.max(np.abs(grad))) * grad
        else:
            theta += lr * grad

        # print(f"Grad {i}: ", grad)
        pi = softmax(theta, axis = 2)
        V, eta = sparse_T_CPU(pi, dataC, rSource, cSource, find_range, R, rho, M)
        Q = calc_Q(V, R, M)
        e = time.perf_counter()
        if (i+1) % 1000 == 0:
            delta = np.max(np.abs(V - Vconv))
            print(i+1, time.ctime(), "Delta :", delta, file=output)
            if delta < tol:
                print(f"Converged in {i+1} iterations", file=output)
                np.save(os.path.join(saveDir,f"theta_Conv{i+1}"), theta)
                np.save(os.path.join(saveDir,f"V_Conv{i+1}"), V)
                print("Theta END: ", theta, file=output)
                print("PI END: ", pi, flush=True, file=output)
                converged = True
                break
            Vconv = V
            np.save(os.path.join(saveDir,f"theta_{i+1}"), theta)
            np.save(os.path.join(saveDir,f"V_{i+1}"), V)
            print("last iteration took ", e-s, " seconds", flush=True, file=output)
    if not converged:
        np.save(os.path.join(saveDir,f"theta_{maxIt}"), theta)
        np.save(os.path.join(saveDir,f"V_{maxIt}"), V)
        print("Theta END not conv :", theta, file=output)
        print("PI END not COnv", pi, flush = True, file=output)
