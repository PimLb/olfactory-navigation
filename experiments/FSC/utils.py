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


def vanillaGradVect(pis, obs, curMems, actions, out = None):
    tmp = np.arange(pis.shape[0]).astype(int)
    if out is None:
        ret = np.zeros_like(pis)
        ret[tmp, obs, curMems, actions] = 1
        ret[tmp, obs,curMems] -= pis[tmp, obs, curMems]
        return ret
    out[:] = 0
    out[tmp, obs, curMems, actions] = 1
    out[tmp, obs, curMems] -= pis[tmp, obs, curMems]

def natGradVect(pis, obs, curMems, actions, out = None):
    tmp = np.arange(pis.shape[0]).astype(int)
    if out is None:
        ret = np.zeros_like(pis)
        ret[tmp, obs, curMems, actions] = 1 / pis[tmp, obs, curMems, actions]
        return ret
    out[:] = 0
    out[tmp, obs, curMems, actions] = 1 / pis[tmp, obs, curMems, actions]

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

# Assumes CDF is obtained as np.cumsum(p, axis = 1), where p is an array (s, e). Each entry of the e axis represent the probability to sample the
# event in such position. Makes no check wheter this assumption is respected
# If event is provided, is assumed to be a numpy array of correct dimension and return the samples from such array
# If event is not provided, return the index of choice done, should be equivalent to provide np.arange(cdf.shape[0]) as events
def multiChoice(CDF, events = None):
    u = np.random.random(CDF.shape[0])[:, None]
    idx = np.argmax(u < CDF, axis = 1)
    if events is None:
        return idx
    return events[idx]

# Only 1 between cumProbs and pi can be provided.
# cumProbs is assumed to be derived as np.cumSum(pi.reshape(-1, 4*M), axis = 1)
# if pi is provided, cumProbs will be constructed as such
# then, the function will return the actions sampled from each pair of obs and curMem
def chooseActionsVect(obs, curMems, cumProbs = None, pi = None):
    if cumProbs is not None and pi is not None:
        raise ValueError("Only one of cumProbs and pi can be provided")
    if cumProbs is None and pi is None:
        raise ValueError("Either cumProbs or pi must be provided")
    if pi is not None:
        M = pi.shape[1]
        cumProbs = np.cumsum(pi.reshape(-1, 4*M), axis = 1)
    else:
        M = cumProbs.shape[1]
    assert np.all((obs == 0) | (obs == 1)) , "Invalid Observation"
    assert np.all((0 <= curMems) & (curMems < M)), "Invalid Memories"
    CDFs = cumProbs[obs*M+curMems]
    return multiChoice(CDFs)


def getObsTurb(states, t, odor, rows, cols, maxFrames, threshold = 1e-4):
    curFrame = t % maxFrames
    rr = states[:, 0]
    cc = states[:, 1]
    valid = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
    obs = odor[f'odor/{curFrame}'][:] >= threshold
    res = np.zeros(states.shape[0], dtype=bool)
    res[valid] = obs[rr[valid], cc[valid]]
    return res

def getObsLikelihood(states, obsProb, rows, cols):
    cdf = np.ones((states.shape[0], 2))
    rr = states[:, 0]
    cc = states[:, 1]
    valid = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
    u = np.random.random(states.shape[0])[:, None]
    cdf[valid] = obsProb[rr[valid]*cols+cc[valid]]
    
    return np.argmax(u < cdf, axis = 1)


# Maybe let having only part unbounded
def takeActionVect(curStates, actionsMem, rMax = None, cMax = None, unbounded = False):
    if not unbounded and (rMax is None or cMax is None):
        raise ValueError("When not unbounded the limit rMax and cMax must be specified")
    if unbounded:
        newStates = curStates + ActionDict[actionsMem % 4]
    else:
        newStates = np.clip(curStates + ActionDict[actionsMem % 4], [0,0], [rMax-1, cMax-1])
    newMem = actionsMem // 4
    return newStates, newMem

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

