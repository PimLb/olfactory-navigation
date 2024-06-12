from olfactory_navigation.agents.model_based_util.pomdp import Model
from olfactory_navigation.agents.model_based_util.value_function import ValueFunction
from olfactory_navigation.agents.model_based_util.belief import Belief

import numpy as np
gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')

class BeliefValueMapping:
    '''
    Alternate representation of a value function, particularly for pomdp models.
    It works by adding adding belief and associated value to the object.
    To evaluate this version of the value function the sawtooth algorithm is used (described in Shani G. et al., "A survey of point-based POMDP solvers")
    
    We can also compute the Q value for a particular belief b and action using the qva function.


    Parameters
    ----------
    model : pomdp.Model
        The model on which the value function applies on
    corner_belief_values : ValueFunction
        A general value function to define the value at corner points in belief space (ie: at certainty beliefs, or when beliefs have a probability of 1 for a given state).
        This is usually the solution of the MDP version of the problem.

    Attributes
    ----------
    model : pomdp.Model
    corner_belief_values : ValueFunction
    corner_values : np.ndarray
        Array of |S| shape, having the max value at each state based on the corner_belief_values.
    beliefs : Belief
        Beliefs contained in the belief-value mapping.
    belief_value_mapping : dict[bytes, float]
        Mapping of beliefs points with their associated value.
    
    '''
    def __init__(self,
                 model: Model,
                 corner_belief_values: ValueFunction
                 ) -> None:
        xp = np if not gpu_support else cp.get_array_module(corner_belief_values.alpha_vector_array)

        self.model = model
        self.corner_belief_values = corner_belief_values
        
        self.corner_values = xp.max(corner_belief_values.alpha_vector_array, axis=0)

        self.beliefs = []
        self.belief_value_mapping = {}

        self._belief_array = None
        self._value_array = None

    
    def add(self,
            b: Belief,
            v: float
            ) -> None:
        '''
        Function to a belief point and its associated value to the belief value mappings.

        Parameters
        ----------
        b: Belief
            A belief to add the belief value mappings.
        v: float
            The value associated to the belief to be added to the mappings.
        '''
        if b not in self.beliefs:
            self.beliefs.append(b)
            self.belief_value_mapping[b.bytes_repr] = v


    @property
    def belief_array(self) -> np.ndarray:
        '''
        The beliefs represented in the form of an array.
        '''
        xp = np if not gpu_support else cp.get_array_module(self.beliefs[0].values)

        if self._belief_array is None:
            self._belief_array = xp.array([b.values for b in self.beliefs])

        return self._belief_array
    

    @property
    def value_array(self) -> np.ndarray:
        '''
        An array of the values.
        '''
        xp = np if not gpu_support else cp.get_array_module(self.beliefs[0].values)

        if self._value_array is None:
            self._value_array = xp.array(list(self.belief_value_mapping.values()))

        return self._value_array
    

    def update(self) -> None:
        '''
        Function to update the belief and value arrays to speed up computation.
        '''
        xp = np if not gpu_support else cp.get_array_module(self.beliefs[0].values)

        self._belief_array = xp.array([b.values for b in self.beliefs])
        self._value_array = xp.array(list(self.belief_value_mapping.values()))


    def evaluate(self, belief: Belief) -> float:
        '''
        Runs the sawtooth algorithm to find the value at a given belief point.

        Parameters
        ----------
        belief : Belief
        '''
        xp = np if not gpu_support else cp.get_array_module(belief.values)

        # Shortcut if belief already exists in the mapping
        if belief in self.beliefs:
            return self.belief_value_mapping[belief.bytes_repr]

        v0 = xp.dot(belief.values, self.corner_values)

        if len(self.beliefs) == 0:
            return float(v0)

        with np.errstate(divide='ignore', invalid='ignore'):
            vb = v0 + ((self.value_array - xp.dot(self.belief_array, self.corner_values)) * xp.min(belief.values / self.belief_array, axis=1))

        return float(xp.min(xp.append(vb, v0)))