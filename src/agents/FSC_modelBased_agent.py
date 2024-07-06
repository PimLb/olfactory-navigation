from ..environment import Environment
from ..agent import Agent
from .model_based_util.pomdp import Model
from scipy.special import softmax as softmax
from tqdm import tqdm

import numpy as np
gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')

class FSC_model_based(Agent):
    def __init__(self, environment: Environment, 
                 treshold: float | None = 0.000003, 
                 name: str | None = "FSC_modelBased",
                 memory_states : int | None = 3,
                 tol_eta : float | None = 1e-7, # eta and Q will be calculated iteratively
                 #tol_Q :float | None = 1e-7,
                 tol_V : float | None = 1e-7,
                 lr : float | None = 1e-3, # learning rate
                 tol_convergence : float | None = 1e-10,
                 gamma: float | None = 0.99975,
                 maxIt: int | None = 1200000,
                 etaMaxIt: int | None = 1000,
                 #QMaxIt: int | None = 1000,
                 VMaxIt: int | None = 1000
                 ) -> None:
        super().__init__(environment, treshold, name)
        self.model = Model.from_environment(environment, treshold)

        self.M = memory_states
        # self.cur_M = 0
        self.tol_eta = tol_eta
        # self.tol_Q = tol_Q
        self.tol_V = tol_V
        self.lr = lr
        self.tol_convergence = tol_convergence
        self.gamma = gamma
        self.maxIt = maxIt
        self.etaMaxIt = etaMaxIt
        # self.QMaxIt = QMaxIt
        self.VMaxIt = VMaxIt
        # The policy is parametrized as #(Obs * Memories) softmaxes. Each of them represent the probability of doing an action and transitioning into a memory
        # Theta are the parameter of the softmax. The first dimension represent the observations, the second which memory the agent is, the third which action and memory have been chosen
        
        #self.theta = np.ones((2, self.M, self.M * 4)) # TODO? make Observation and Action parametric

        self.theta = np.ones((3, self.M, self.M * 4)) # TODO: this one has 3 observation, but I don't need them

        #The policy itself is the softmax on the third axis
        self.pi = softmax(self.theta, axis = 2)

        self.reward = -(1- gamma) # TODO: making it a function of state, action and memory; And not harcoded (?)
        self.xp = cp if gpu_support else np


        # I create the Reward vector, it is self.reward everywhere except in the ends states
        self.R = self.xp.ones(self.model.state_count) * self.reward
        self.R[self.model.end_states] = 0
        #but replicated for each memory
        self.R = self.xp.tile(self.R, self.M)

    def _get_Transition_Matrix_(self):
        sc = self.model.state_count
        T = self.xp.zeros((sc * self.M, sc * self.M))
        assert np.allclose(np.sum(self.pi, axis = 2), 1), "Pi is not a probability"
        # print(T.shape)
        for s in self.model.states:
            reachable = self.model.reachable_states[s]
            pReachable = self.model.reachable_probabilities[s]
            # print(self.model.reachable_probabilities.shape, "<- total", reachable.shape)
            for m in range(self.M):
                if s in self.model.end_states:
                    T[ m * sc + s, m * sc + s] = 1
                    continue
                for a in self.model.actions:
                    for r_idx, r in enumerate(reachable[a]):
                        for nextM in range(self.M):
                            #TODO: the Model.from_env method creates a model with 3 observation, but I need only 2
                            for o in self.model.observations: 
                                # in the paper, they use the p(o | s) while this is p(o | s'). Does it changes anything?
                                # Is it, tho? If I put r instead of s the matrix won't sum to 1.
                                # Also, for every s, obs_table[s, 0] == obs_table[s, 1] == obs_table[s, 2], which seems strange to me

                                pObs = self.model.observation_table[s, a, o]
                                T[ m * sc + s, nextM * sc + r] += self.pi[o, m, nextM * 4 + a] * pReachable[a, r_idx] * pObs
            # if s % 100:
            #     print(f"{s} out of {T.shape[0]}")
        return T 
        # It's probably not consistent with the order I used for all other array
        # return T.reshape(T.shape[0] * T.shape[1], -1)

    def _solve_eta_(self, eta, T = None):
        # print("eta")
        sp = self.environment.start_probabilities.ravel()
        rho = self.xp.tile(sp, self.M) / self.M # unfiform in all memories
        # rho = self.xp.append(sp, [0] * (self.M -1) * sp.shape[0] ) # start only on the first memory
        assert self.xp.sum(rho), "Rho is not normalized"
        if T is None:
            T = self._get_Transition_Matrix_()
        for i in range(self.etaMaxIt):
            #print(rho.shape, T.shape, eta.shape)
            # The matmul is transposed in respect to the paper one Because we use a different convention for T
            new_eta = rho + self.gamma * self.xp.matmul(eta, T) # TODO: is this efficent enough? Does it uses GPU if enabled?
            if self.xp.max(self.xp.abs(new_eta - eta) ) < self.tol_eta:
                return new_eta
            eta = new_eta
            if i % 100 == 0:
                print(f"Eta: {i} out of maximum {self.etaMaxIt} iterations")
        print(f"eta not converged with {self.etaMaxIt} iteration and {self.tol_eta} tolerance")
        return eta

    # def _solve_Q_(self, Q):
    #     print("Q")
    #     R = self.xp.ones(np.prod(self.environment.shape) * self.M) * self.reward
    #     end_states = np.argwhere(np.fromfunction(lambda x,y: ((x-self.environment.source_position[0])**2 + (y-self.environment.source_position[1])**2) <= self.environment.source_radius**2,
    #                                              shape=self.environment.shape).ravel())[:,0].tolist() * self.M
    #     R[end_states] = 0 # the final reward is 0, while every step has self.reward
    #     T = self._get_Transition_Matrix_()
    #     V = 
    #     for i in range(self.QMaxIt):
    #         new_Q = R + self.gamma * self.xp.matmul(T, V)
    
    def _calc_Q_(self, V):
        sc = self.model.state_count
        ac = self.model.action_count
        Q = self.xp.zeros((sc * self.M, ac * self.M))
        for s in self.model.states:
            for m in range(self.M):
                for a in self.model.actions:
                    for nextM in range(self.M):
                        for r_idx, r in enumerate(self.model.reachable_states[s, a]):
                            Q[s + sc * m, a + ac * nextM ] += self.model.reachable_probabilities[s, a, r_idx] * (self.R[sc*nextM + r] + self.gamma * V[sc*nextM + r])
        return Q

    def _solve_V_(self, V, T = None):
        # print(f"calcultaing V with {self.VMaxIt} iterations and {self.tol_V} tolerance")
        if T is None:
            T = self._get_Transition_Matrix_()
        # V = V.T
        for i in range(self.VMaxIt):
            new_V = self.R + self.gamma * self.xp.matmul(T, V) # TODO: is this efficent enough? Does it uses GPU if enabled?
            if( self.xp.max(self.xp.abs(new_V - V)) < self.tol_V):
                return new_V
            V = new_V
        print(f"V not converged with {self.VMaxIt} iteration and {self.tol_V} tolerance")
        return V

    def _find_grad_(self, Q, eta):
        grad = np.zeros_like(self.theta)
        sc = self.model.state_count
        O, M, AM = self.theta.shape
        for o in range(O):
            for m in range(M):
                for am in range(AM):
                    a = am % self.model.action_count
                    for s in self.model.states:
                        grad[o, m, am] += eta[sc * m + s] * self.model.observation_table[s, a, o] * Q[sc * m + s, am]
        return grad
            

    def train(self) -> None:
        sc = self.model.state_count
        SM = sc * self.M
        eta = self.xp.zeros(SM)
        oldV = self.xp.zeros(SM)
        self.pi = softmax(self.theta, axis = 2)
        eta = self._solve_eta_(eta)
        oldV = self._solve_V_(oldV)
        Q = self._calc_Q_(oldV)

        iterator = tqdm(range(self.maxIt))
        for i in iterator:
            grad = self._find_grad_(Q, eta)
            #grad -= np.max(grad, axis=2, keepdims=True) # I don't know why they do this, so it's commented. probably same reason as below
            self.theta += self.lr * grad
            self.theta -= np.mean(self.theta, axis = 2, keepdims=True) # Should improve numerical stability
            self.pi = softmax(self.theta, axis= 2) # Softmax doesn't change if we add the same number to each operand
            T = self._get_Transition_Matrix_()
            V = self._solve_V_(oldV)
            if self.xp.max(self.xp.abs(V - oldV)) < self.tol_convergence:
            V = self._solve_V_(oldV, T)
            delta = self.xp.max(self.xp.abs(V - oldV))
            if delta < self.tol_convergence:
                print(f"Converged in {i} steps")
                return
            oldV = V
            eta = self._solve_eta_(eta, T)
            Q = self._calc_Q_(V)
            iterator.set_postfix({"delta": f"{delta}/{self.tol_convergence}"})
