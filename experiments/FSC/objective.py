import numpy as np
from modelBasedTrain import get_Transition_Matrix_sparse_CPU
import sys
from scipy.special import softmax
import scipy.sparse as sparse

SC = 131*92
gamma = 0.99975
cSource = 45.5
rSource = 91
find_range = 1.1
cols = 92
rows = 131

dataC = np.load(f"celaniData/fine5.npy")

R = np.ones(SC) * -(1 - gamma)
for s in range(SC):
    r, c = s // 92, s % 92 
    if (r - rSource) ** 2 + (c -cSource) **2 < find_range**2:
        R[s::SC] = 0
rho = np.zeros(SC)
rho[:cols] = (1-dataC[0,:cols])/np.sum((1-dataC[0,:cols])) # Copiato dal loro

maxKey = None
maxValue = -10

for line in sys.stdin:
    th = np.load(line.rstrip())
    pi = softmax(th, axis = 2)
    T = get_Transition_Matrix_sparse_CPU(pi, dataC, rSource, cSource, find_range, 1)
    AV = sparse.eye(SC , format="csr") - gamma * T
    V = sparse.linalg.spsolve(AV, R)

    key = line.rstrip()
    val = np.dot(V, rho)
    print(key, val)
    if maxValue < val:
        maxKey = key
        maxValue = val
print("MAX: ", maxKey, maxValue)
