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
import argparse as ap
# from cupyx.scipy.special import softmax as softmax
# from utils import get_value

#Azioni sono [North, East, South, West]
SC = 92 * 131 # Number of States
gamma = 0.99975
cSource = 45.5
rSource = 91
find_range = 1.1
cols = 92
rows = 131
maxIt = 35000
lr = 0.01

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


#Dovrebbe essere meglio che farlo solo su GPU o solo su CPU
def ggTrasfer(pi, pObs, rSource, cSource, find_range, RGPU, rhoGPU, M):
    toSum, rowIdx, colIdx = get_Transition_Matrix_sparse_CPU(pi, pObs, rSource, cSource, find_range, M, create_matrix = False)
    toSumGPU = cp.asarray(toSum)
    rowIdxGPU = cp.asarray(rowIdx)
    colIdxGPU = cp.asarray(colIdx)
    T = cSparse.csr_matrix((toSumGPU, (rowIdxGPU, colIdxGPU)))
    AV = cSparse.eye(SC *M, format="csr") - gamma * T
    Aeta = cSparse.eye(SC *M, format="csr") - gamma * T.T
    V = cSparseLA.spsolve(AV, RGPU)
    eta = cSparseLA.spsolve(Aeta, rhoGPU)
    # T = cSparse.csr_matrix((toSumGPU, (rowIdxGPU, colIdxGPU))).todense()
    # Tinv = cp.linalg.inv(cp.eye(SC *M) - gamma * T)
    # V = Tinv @ RGPU
    return V, eta

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
    s = time.perf_counter()
    T = get_Transition_Matrix_sparse_CPU(pi, dataC, rSource, cSource, find_range, M)
    AV = sparse.eye(SC * M, format="csr") - gamma * T
    Aeta = sparse.eye(SC * M, format="csr") - gamma * T.T
    V = sparse.linalg.spsolve(AV, R)
    eta = sparse.linalg.spsolve(Aeta, rho)
    e = time.perf_counter()
    print("TOT", e-s)
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

def isEnd(sm, rSource, cSource, find_range):
    s = sm % SC
    r, c = s // cols, s % cols
    return (r - rSource) ** 2 + (c -cSource) **2 < find_range**2


# Dovrebbe essere giusto anche per M > 1; Quasi certo per M = 2, da provare per 3
def get_Transition_Matrix_sparse_CPU(pi, pObs, rSource, cSource, find_range, M, create_matrix = True):
    # Creo un array in cui ogni valore rappresenta lo stato di partenza
    # ogni stato fisico è seguito dallo stesso stato in altre memorie
    rowIdx = np.fromiter((i + j * SC for i in range(SC) for j in range(M)), int)
    
    # Per ogni stato prendo gli stati raggiungibili e li metto in un array
    colIdx = np.fromiter((chain.from_iterable(map(getReachable, rowIdx, itReapeat(M)))) , int) 
    
    #Da ogni stato posso fare 4*M azioni, quindi replico in modo che siano coerenti con colIdx
    rowIdx = rowIdx.repeat(4*M)

    # Moltiplico la probabilità di fare l'azione data l'osservazione e moltipico per la probabilità d'osservazione
    # e sommo le probabilità dagli stessi stati
    toSum = np.sum(pi[None, :, :, :].T * pObs[:, :], axis = 2).T.reshape(-1)

    final = [s for s in range(SC * M) if isEnd(s, rSource, cSource, find_range)]
    

    # Ad ognuno degli stati finali cambio gli stati raggiungibili.
    v = 1 / (4*M)
    for f in final:
        idxs = rowIdx == f
        colIdx[idxs] = f
        toSum[idxs] = v
    if create_matrix:
        return sparse.csr_matrix((toSum, (rowIdx, colIdx)))
    return toSum, rowIdx, colIdx # Se usato con ggTransfer

def prova(pi, pObs, rSource, cSource, find_range, M):
    T = np.zeros((SC*M, SC*M))
    for i in range(SC * M):
        for a, j in enumerate(getReachable(i, M)):
            for o in range(2):
                T[i,j] += pi[o, i // SC, a] * pObs[o, i % SC]
    final = [s for s in range(SC) if (s // 92 - rSource) ** 2 + (s % 92 -cSource) **2 < find_range**2 ]
    for i in final:
        for m in range(M):
            T[i + m *SC] = 0
            T[i + m * SC, i + m * SC] = 1
    return T


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


def calc_Q_GPU(V, R, M):
    Q = cp.zeros((SC * M, 4*M))
    RV = R + gamma * V
    toSum = cp.array([RV[getReachable(s, M)] for s in range(SC*M)])
    for a in range(4*M):
        cp.add.at(Q, (cp.arange(SC * M, dtype=int), a), toSum[:, a])
    mask = cp.array([((s % SC) // 92 - rSource) ** 2 + (s % 92 -cSource) **2 < find_range**2 for s in range(SC * M)])
    Q[mask] = 0
    return Q.get()

def calc_Q(V, R, M):
    Q = np.zeros((SC * M, 4*M))
    RV = (R + gamma * V).get()
    # print(type(Q), type(V), type(R))
    toSum = np.array([RV[getReachable(s, M)] for s in range(SC*M)])
    for a in range(4*M):
        np.add.at(Q, (np.arange(SC * M, dtype=int), a), toSum[:, a])
    mask = np.array([((s % SC) // 92 - rSource) ** 2 + (s % 92 -cSource) **2 < find_range**2 for s in range(SC * M)])
    Q[mask] = 0
    return Q

def find_grad(Q, eta, pObs, M):
    grad = np.zeros((2, M, 4 *M))
    # tmp = np.tile(pObs, M)
    # for a in range(4):
    #     grad[:, 0, a] = np.sum(tmp * eta * Q[:, a], axis=1)
    # print(type(Q), type(eta), type(pObs), type(M))
    for obs in range(2):
        for m in range(M):
            for am in range(4*M):
                grad[obs, m, am] = np.sum(eta[SC*m:SC*(m+1)] * pObs[obs, :] * Q[SC*m:SC*(m+1), am])
    return grad

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

#Risolvere il sistema lineare con CUPY ci mette di più che un'intera iterazione di Gradient Ascent con scipy

if __name__ == "__main__":

    parser = ap.ArgumentParser(formatter_class=ap.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_file", help="the data to use")
    parser.add_argument("name", help="subfolder name in which to save the results")
    parser.add_argument("learning_rate", type=float, help="the learning rate")
    parser.add_argument("memories", type=int, help="How many memories to use")
    parser.add_argument("-t","--thetaStart", help="the path to a .npy file containing the starting values of theta")
    parser.add_argument("-r","--rescale", help="Wheter to rescale the gradient or not", action="store_true")
    parser.add_argument("-s","--subtract", help="Wheter to subtract the maximum of the gradient", action="store_true")
    parser.add_argument("-g","--gamma", type = float, help="the value of gamma", default=0.99975)
    parser.add_argument("--tolerance", type = float, help="the minimum difference under which assume convergence", default=1e-8)
    parser.add_argument("--GPU", help="Which GPU to use, if not specified will use CPU", type = int)
    args = parser.parse_args()

    dataFile = args.data_file
    folder = args.name
    lr = args.learning_rate
    M = args.memories
    rescale = args.rescale
    subtract = args.subtract
    gamma = args.gamma
    ts = args.thetaStart
    tol = args.tolerance
    GPU = args.GPU
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
    os.makedirs(saveDir)
    output = open(os.path.join(saveDir, "output.out"), "w")
    print("Inizio: ", time.ctime(), flush=True, file=output)

    dataC = np.load(f"celaniData/{dataFile}.npy")
    if ts is None:
        theta = (np.random.rand(2, M, 4*M) -0.5) * 0.5
        theta[1, :, 0::4] += 0.5
        theta[1, :, 2::4] += 0.5 # Bias on upwind and downwind directions
    else:
        theta = np.load(ts)
    np.save(os.path.join(saveDir,"theta_START"), theta)
    pi = softmax(theta, axis = 2)
    print(f"Learning Rate {lr}; Memories {M}; gamma {gamma}; tolerance {tol}", file=output)
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
    calc_V_Eta = sparse_T_CPU
    if GPU is not None:
        cp.cuda.Device(GPU).use()
        R = cp.asarray(R)
        rho = cp.asarray(rho)
        calc_V_Eta = ggTrasfer
        # calc_Q = calc_Q_GPU

    Vold, eta = calc_V_Eta(pi, dataC, rSource, cSource, find_range, R, rho, M)
    Vconv = Vold.copy()
    Q = calc_Q(Vold, R, M)
    converged = False
    s = time.perf_counter()
    objs = [(Vconv @ rho).get()] if GPU is not None else [Vconv @ rho]
    for i in range(maxIt):
        # print(i, flush = True)
        itS = time.perf_counter()
        
        grad = find_grad(Q, eta.get(), dataC, M)
        if subtract:
            grad -= np.max(grad, axis = 2, keepdims=True)
        if rescale:  
            theta += lr / (np.max(np.abs(grad))) * grad
        else:
            theta += lr * grad

        # print(f"Grad {i}: ", grad, file = output, flush = True)
        
        pi = softmax(theta, axis = 2)
        V, eta = calc_V_Eta(pi, dataC, rSource, cSource, find_range, R, rho, M)
        Q = calc_Q(V, R, M)
        e = time.perf_counter()
        if (i+1) % 1000 == 0:
            delta = np.max(np.abs(V - Vconv)) # Use difference between objectives instead?
            print(i+1, time.ctime(), "Delta :", delta, "\tObj: ", V @ rho, file=output)
            objs.append(V @ rho if GPU is None else (V @ rho).get())
            print(f"PI {i+1}: ", pi, flush=True, file=output)
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
            print("last iteration took ", e-itS, " seconds", flush=True, file=output)
    if not converged:
        np.save(os.path.join(saveDir,f"theta_{maxIt}"), theta)
        np.save(os.path.join(saveDir,f"V_{maxIt}"), V)
        print("Theta END not conv :", theta, file=output)
        print("PI END not COnv", pi, flush = True, file=output)
    totalTime(e, s, output)
    ticks = [0, -0.1, -0.3, -0.4,-0.485, -0.6, -0.7, -0.8, -0.9, -1]
    if M == 3:
        plt.hlines(-0.138, 0,maxIt, "r", label = f"Optimal M3")
        ticks += [-0.138]
    if M >= 2:
        ticks += [-0.197]
        plt.hlines(-0.197, 0,maxIt, "y", label = f"Optimal M2")
    else:
        ticks += [-0.2]
    plt.hlines(-0.485, 0,maxIt, "g", label = f"Optimal M1")
    plt.yticks(ticks)
    plt.plot(range(0, maxIt+1, 1000), objs, label = "Objective")
    plt.grid()
    plt.legend()
    imgName = folder + f"_{lr}"
    if rescale:
        imgName += "_r"
    if subtract:
        imgName += "_s"
    plt.savefig(f"objOut/png/modelBased/{imgName}.png")