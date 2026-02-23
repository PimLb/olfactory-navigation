import numpy as np
from scipy.special import softmax as softmax
import time
import os
import h5py


ActionDict = [
            (-1,  0), # North
            ( 0,  1), # East
            ( 1,  0), # South
            ( 0, -1)  # West
        ]

def natGrad(pi, obs, curMem, action, out = None):
    if out is None:
        ret = np.zeros_like(pi)
        ret[obs, curMem, action] = 1 / pi[obs, curMem, action]
        return ret
    out[:] = 0
    out[obs, curMem, action] = 1 / pi[obs, curMem, action]

def vanillaGrad(pi, obs, curMem, action, out = None):
    if out is None:
        ret = np.zeros_like(pi)
        ret[obs, curMem, action] = 1
        ret[obs,curMem] -= pi[obs, curMem]
        return ret
    out[:] = 0
    out[obs, curMem, action] = 1
    out[obs,curMem] -= pi[obs, curMem]

def isEnd(states, src, finished=False, find_range = 1.1 ):
    return finished | (np.linalg.norm(states-src, axis=1) < find_range)

def getEndingStates(rMax, cMax, src, find_range=1.1):
    allStates = np.stack(np.unravel_index(np.arange(rMax*cMax), (rMax,cMax))).T
    return allStates[isEnd(allStates, src, find_range=find_range)]

# Starts and Ends should have a dimension (n,2) and (m, 2) respectively
# The None in the middle of ends[:, None, :] makes it boradcastable, thus subtracting each of the n starts by each of the m ends
# Get the minimum manhattan distance (L1 norm) between an array of states starts and an array of states ends
# np.floor might be superfluous
def getMinDist(starts, ends):
    return np.min(np.floor(np.linalg.norm(starts-ends[:, None, :], ord = 1, axis = 2)), axis = 0)

   
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

