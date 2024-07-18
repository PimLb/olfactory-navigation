from ..environment import Environment
from ..agent import Agent
from tqdm import tqdm

import numpy as np
gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')

if gpu_support:
    from cupyx.scipy.special import softmax as softmax
else:
    from scipy.special import softmax as softmax


class FSC_model_free(Agent):
    def __init__(self, environment: Environment, 
                 treshold: float  = 3e-6, 
                 memory_states: int | None = 1,
                 lr : float | None = 1e-2,
                 gamma: float | None = 0.99975,
                 maxIt: int | None = 12000,
                 tolerance: float | None = 1e-3,
                 MC_samples: int | None = 50,
                 max_MC_steps: int | None = 10000,
                 h: float | None = 1e-2,
                 #TODO: aggiungere la possibilità di cambiare le direzioni 
                 #TODO: aggiungere la possibilità di scegliere il theta iniziale
                 name = "FSC_modelFree"
                 ) -> None :
        
        super().__init__(environment, treshold, name)
        self.M = memory_states
        self.lr = lr
        self.gamma = gamma
        self.maxIt = maxIt
        self.tolerance = tolerance
        self.MC_samples = MC_samples
        self.h = h
        self.S = np.prod(environment.shape)
        self.SM = np.prod(environment.shape) * self.M
        self.max_MC_steps = max_MC_steps

        self.end_states = np.argwhere(np.fromfunction(lambda x,y: ((x-environment.source_position[0])**2 + (y-environment.source_position[1])**2) <= environment.source_radius**2,
                                                 shape=environment.shape).ravel())[:,0].tolist()

        self.xp = cp if gpu_support else np

        self.theta = self.xp.ones((3, self.M, 4 * self.M))
        self.pi = softmax(self.theta, axis = 2)
        self.reward = -(1- gamma)

        self.R = self.xp.ones(self.S) * self.reward
        self.R[self.end_states] = 0
        self.R = self.xp.tile(self.R, self.M)

        self.ActionDict = self.xp.asarray([
            [-1,  0], # North
            [ 0,  1], # East
            [ 1,  0], # South
            [ 0, -1]  # West
        ])

        self.environment = self.environment.to_gpu()
    
    #Returns the chosen action and memory as integers
    def choose_action(self, obs, curM):
        am = self.xp.random.choice(4*self.M, size = 1, p= self.pi[obs, curM])
        return am % 4, am // 4
    
    def getTrajectories(self, num = None, start = None):
        if start is None and num is None:
            start = self.environment.random_start_points(self.MC_samples)
            num = start.shape[0]
        elif start is None:
            start = self.environment.random_start_points(num)
        if num is None:
            num = start.shape[0]
        assert start.shape[0] == num, f"Shape mismatch: {num} != {start.shape[0]}"
        step = start.astype(int)
        done = self.environment.source_reached(step)
        t = 0
        obs = self.environment.get_observation(start, t).astype(int) 
        curMs = self.xp.zeros(num, dtype=int)
        actions = self.xp.zeros((num, 2), dtype=int)
        rewards = self.xp.zeros(num)
        while self.xp.any(~done) and t < self.max_MC_steps:
            for i in range(num): # TODO: sequenziale, sarebbe da parallelizare
                if not done[i]:
                    a, m = self.choose_action(obs[i], curMs[i])
                    actions[i] = self.ActionDict[a]
                    curMs[i] = m
                else:
                    actions[i] = cp.zeros(2)
            t+= 1
            step = self.environment.move(step, actions)
            obs = self.environment.get_observation(step, t).astype(int, copy=False) # TODO: Ad una certa finiscono i dati, per ora ricomincia
            done = self.environment.source_reached(step)
            rewards[~done] += self.reward
        return rewards


    def train(self):
        grad = self.xp.ones_like(self.theta)
        i = 0
        while self.xp.linalg.norm(grad) > self.tolerance and i < self.maxIt:
            self.pi = softmax(self.theta, axis = 2)
            starts = self.environment.random_start_points(self.MC_samples)
            rewards = self.xp.mean(self.getTrajectories(start=starts))
            #TODO: mancano le direzioni. Per ora dovrebbe essere la base canonica
            #TODO: assolutamente da parallelizare
            for o in range(3):
                for m in range(self.M):
                    for am in range(4 * self.M):
                        th2 = self.theta.copy()
                        th2[o, m, am] +=  self.h
                        self.pi = softmax(th2, axis= 2)
                        rewards2 = self.xp.mean(self.getTrajectories(start=starts))
                        grad[o, m, am] = self.xp.sum(rewards2 - rewards) / self.h
            self.theta += self.lr * grad
            self.theta -= self.xp.mean(self.theta, axis = 2, keepdims=True) # Should improve numerical stability
            self.pi = softmax(self.theta, axis= 2) # Softmax doesn't change if we add the same number to each operand


            i += 1