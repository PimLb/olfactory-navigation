from matplotlib import pyplot as plt
from scipy.stats import entropy

from olfactory_navigation.agents.model_based_util.pomdp import Model

import numpy as np
gpu_support = False
try:
    import cupy as cp
    from cupyx.scipy.stats import entropy as cupy_entropy
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')


class Belief:
    '''
    A class representing a belief in the space of a given model. It is the belief to be in any combination of states:
    eg:
        - In a 2 state POMDP: a belief of (0.5, 0.5) represent the complete ignorance of which state we are in. Where a (1.0, 0.0) belief is the certainty to be in state 0.

    The belief update function has been implemented based on the belief update define in the paper of J. Pineau, G. Gordon, and S. Thrun, 'Point-based approximations for fast POMDP solving'


    Parameters
    ----------
    model : pomdp.Model
        The model on which the belief applies on.
    values : np.ndarray, optional
        A vector of the probabilities to be in each state of the model. The sum of the probabilities must sum to 1.
        If not specified, it will be set as the start probabilities of the model.

    Attributes
    ----------
    model : pomdp.Model
    values : np.ndarray
    bytes_repr : bytes
        A representation in bytes of the value of the belief
    '''
    def __init__(self,
                 model: Model,
                 values: np.ndarray | None = None
                 ) -> None:
        assert model is not None
        self.model = model

        if values is not None:
            assert values.shape[0] == model.state_count, "Belief must contain be of dimension |S|"

            xp = np if not gpu_support else cp.get_array_module(values)

            prob_sum = xp.sum(values)
            rounded_sum = xp.round(prob_sum, decimals=3)
            assert rounded_sum == 1.0, f"States probabilities in belief must sum to 1 (found: {prob_sum}; rounded {rounded_sum})"

            self._values = values
        else:
            self._values = model.start_probabilities


    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)

        instance._bytes_repr = None
        instance._successors = {}
        
        return instance


    @property
    def bytes_repr(self) -> bytes:
        '''
        A representation as bytes of a belief.
        '''
        if self._bytes_repr is None:
            self._bytes_repr = self.values.tobytes()
        return self._bytes_repr


    def __eq__(self, other: object) -> bool:
        '''
        A way to check the equality between two belief points.
        The byte representation of each belief point is compared.
        '''
        return self.bytes_repr == other.bytes_repr

    
    @property
    def values(self) -> np.ndarray:
        '''
        An array of the probability distribution to be in each state.
        '''
        return self._values
    

    def update(self,
               a: int,
               o: int,
               throw_error: bool = True
               ) -> 'Belief':
        '''
        Returns a new belief based on this current belief, the most recent action (a) and the most recent observation (o).

        Parameters
        ----------
        a : int
            The most recent action.
        o : int
            The most recent observation.
        throw_error : bool, default=True
            Whether the creation of an impossible belief (sum of probabilities of 0.0) will throw an error or not.

        Returns
        -------
        new_belief : Belief
            An updated belief
        '''
        xp = np if not gpu_support else cp.get_array_module(self._values)

        # Check if successor exists
        succ_id = f'{a}_{o}'
        succ = self._successors.get(succ_id)
        if succ is not None:
            return succ

        # Computing new probabilities
        reachable_state_probabilities = self.model.reachable_transitional_observation_table[:,a,o,:] * self.values[:,None]
        new_state_probabilities = xp.bincount(self.model.reachable_states[:,a,:].flatten(), weights=reachable_state_probabilities.flatten(), minlength=self.model.state_count)
        
        # Normalization
        probability_sum = xp.sum(new_state_probabilities)
        if probability_sum == 0:
            if throw_error:
                raise ValueError("Impossible belief: the sum of probabilities is 0...")
        else:
            new_state_probabilities /= probability_sum

        # Generation of new belief from new state probabilities
        new_belief = self.__new__(self.__class__)
        new_belief.model = self.model
        new_belief._values = new_state_probabilities

        # Remember generated successor
        self._successors[succ_id] = new_belief

        return new_belief
    

    def generate_successors(self) -> list['Belief']:
        '''
        Function to generate a set of belief that can be reached for each actions and observations available in the model.

        Returns
        -------
        successor_beliefs : list[Belief]
            The successor beliefs.
        '''
        successor_beliefs = []
        for a in self.model.actions:
            for o in self.model.observations:
                b_ao = self.update(a,o)
                successor_beliefs.append(b_ao)

        return successor_beliefs


    def random_state(self) -> int:
        '''
        Returns a random state of the model weighted by the belief probabily.

        Returns
        -------
        rand_s : int
            A random state.
        '''
        xp = np if not gpu_support else cp.get_array_module(self._values)

        rand_s = int(self.model.rnd_state.choice(a=self.model.states, size=1, p=self._values)[0])
        return rand_s
    

    @property
    def entropy(self) -> float:
        '''
        The entropy of the belief point
        '''
        xp = np if not gpu_support else cp.get_array_module(self._values)

        return float(entropy(self._values) if xp == np else cupy_entropy(self._values))
    

    def plot(self, size: int = 5) -> None:
        '''
        Function to plot a heatmap of the belief distribution if the belief is of a grid model.

        Parameters
        ----------
        size : int, default=5
            The scale of the plot.
        '''
        # Plot setup
        plt.figure(figsize=(size*1.2,size))

        model = self.model.cpu_model

        # Ticks
        dimensions = model.state_grid.shape
        x_ticks = np.arange(0, dimensions[1], (1 if dimensions[1] < 10 else int(dimensions[1] / 10)))
        y_ticks = np.arange(0, dimensions[0], (1 if dimensions[0] < 5 else int(dimensions[0] / 5)))

        plt.xticks(x_ticks)
        plt.yticks(y_ticks)

        # Title
        plt.title(f'Belief (probability distribution over states)')

        # Actual plot
        belief_values = self._values if (not gpu_support) or (cp.get_array_module(self._values) == np) else cp.asnumpy(self._values)
        grid_values = belief_values[model.state_grid]
        plt.imshow(grid_values,cmap='Blues')
        plt.colorbar()
        plt.show()


class BeliefSet:
    '''
    Class to represent a set of beliefs with regard to a POMDP model.
    It has the purpose to store the beliefs in a numpy array format and be able to conver it to a list of Belief class objects.
    

    Parameters
    ----------
    model : pomdp.Model
        The model on which the beliefs apply.
    beliefs : list[Belief] or np.ndarray
        The actual set of beliefs.

    Attributes
    ----------
    model : pomdp.Model
    belief_array : np.ndarray
        A 2D array of shape N x S of N belief vectors.
    belief_list : list[Belief]
        A list of N Belief object.
    '''
    def __init__(self,
                 model: Model,
                 beliefs: list[Belief] | np.ndarray
                 ) -> None:
        self.model = model

        self._belief_list = None
        self._belief_array = None
        self._uniqueness_dict = None

        self.is_on_gpu = False

        if isinstance(beliefs, list):
            assert all(len(b.values) == model.state_count for b in beliefs), f"Beliefs in belief list provided dont all have shape ({model.state_count},)"
            self._belief_list = beliefs

            # Check if on gpu and make sure all beliefs are also on the gpu
            if (len(beliefs) > 0) and gpu_support and cp.get_array_module(beliefs[0].values) == cp:
                assert all(cp.get_array_module(b.values) == cp for b in beliefs), "Either all or none of the alpha vectors should be on the GPU, not just some."
                self.is_on_gpu = True
        else:
            assert beliefs.shape[1] == model.state_count, f"Belief array provided doesnt have the right shape (expected (-,{model.state_count}), received {beliefs.shape})"

            self._belief_array = beliefs

            # Check if array is on gpu
            if gpu_support and cp.get_array_module(beliefs) == cp:
                self.is_on_gpu = True


    @property
    def belief_array(self) -> np.ndarray:
        '''
        A matrix of size N x S containing N belief vectors. If belief set is stored as a list of Belief objects, the matrix of beliefs will be generated from them.
        '''
        xp = cp if (gpu_support and self.is_on_gpu) else np

        if self._belief_array is None:
            self._belief_array = xp.array([b.values for b in self._belief_list])
        return self._belief_array
    

    @property
    def belief_list(self) -> list[Belief]:
        '''
        A list of Belief objects. If the belief set is represented as a matrix of Belief vectors, the list of Belief objects will be generated from it.
        '''
        if self._belief_list is None:
            self._belief_list = [Belief(self.model, belief_vector) for belief_vector in self._belief_array]
        return self._belief_list
    

    def generate_all_successors(self) -> 'BeliefSet':
        '''
        Function to generate the successors beliefs of all the beliefs in the belief set.

        Returns
        -------
        all_successors : BeliefSet
            All successors of all beliefs in the belief set.
        '''
        all_successors = []
        for belief in self.belief_list:
            all_successors.extend(belief.generate_successors())
        return BeliefSet(self.model, all_successors)


    def update(self,
               actions: list | np.ndarray,
               observations: list | np.ndarray,
               throw_error: bool = True
               ) -> 'BeliefSet':
        '''
        Returns a new belief based on this current belief, the most recent action (a) and the most recent observation (o).

        Parameters
        ----------
        actions : list or np.ndarray
            The most recent played actions.
        observations : list or np.ndarray
            The most recent received observations.
        throw_error : bool, default=True
            Whether the throw an error when attempting to generate impossible beliefs.

        Returns
        -------
        new_belief_set : BeliefSet
            An set of updated beliefs.
        '''
        # GPU support check
        xp = cp if (gpu_support and self.is_on_gpu) else np

        # Ensuring we are dealing we are dealing with ndarrays
        observations = xp.array(observations)
        actions = xp.array(actions)

        # Computing reachable probabilities and states
        reachable_probabilities = (self.model.reachable_transitional_observation_table[:, actions, observations, :] * self.belief_array.T[:,:,None])
        reachable_state_per_actions = self.model.reachable_states[:, actions, :]

        # Computing new probabilities
        flatten_offset = xp.arange(len(observations))[:,None] * self.model.state_count
        flat_shape = (len(observations), (self.model.state_count * self.model.reachable_state_count))
        
        a = reachable_state_per_actions.swapaxes(0,1).reshape(flat_shape)
        w = reachable_probabilities.swapaxes(0,1).reshape(flat_shape)

        a_offs = a + flatten_offset
        new_probabilities = xp.bincount(a_offs.ravel(), weights=w.ravel(), minlength=a.shape[0]*self.model.state_count).reshape((-1,self.model.state_count))

        # Normalization
        probability_sum = xp.sum(new_probabilities, axis=1)
        if xp.any(probability_sum == 0.0) and throw_error:
            raise ValueError('One or more belief is impossible, (ie the sum of the probability distribution is 0)')
        non_zero_mask = probability_sum != 0
        new_probabilities[non_zero_mask] /= probability_sum[non_zero_mask,None]

        return BeliefSet(self.model, new_probabilities)


    @property
    def unique_belief_dict(self) -> dict:
        '''
        A dictionary of unique belief points with the keys being the byte representation of these belief points.
        '''
        if self._uniqueness_dict is None:
            self._uniqueness_dict = {belief.bytes_repr: belief for belief in self.belief_list}
        return self._uniqueness_dict


    def union(self, other_belief_set: 'BeliefSet') -> 'BeliefSet':
        '''
        Function to make the union between two belief set objects.

        Parameters
        ----------
        other_belief_set : BeliefSet
            The other belief set to make the union with
        
        Returns
        -------
        new_belief_set : BeliefSet
            A new, combined, belief set
        '''
        # Deduplication
        combined_uniqueness_dict = self.unique_belief_dict | other_belief_set.unique_belief_dict

        # Generation of new set
        new_belief_set = BeliefSet(self.model, list(combined_uniqueness_dict.values()))
        new_belief_set._uniqueness_dict = combined_uniqueness_dict

        return new_belief_set


    def __len__(self) -> int:
        return len(self._belief_list) if self._belief_list is not None else self._belief_array.shape[0]
    

    @property
    def entropies(self) -> np.ndarray:
        '''
        An array of the entropies of the belief points.
        '''
        xp = np if not gpu_support else cp.get_array_module(self.belief_array)

        return entropy(self.belief_array, axis=1) if xp == np else cupy_entropy(self.belief_array, axis=1)


    def to_gpu(self) -> 'BeliefSet':
        '''
        Function returning an equivalent belief set object with the array of values stored on GPU instead of CPU.

        Returns
        -------
        gpu_belief_set : BeliefSet
            A new belief set with array on GPU.
        '''
        assert gpu_support, "GPU support is not enabled, unable to execute this function"

        gpu_model = self.model.gpu_model

        gpu_belief_set = None
        if self._belief_array is not None:
            gpu_belief_array = cp.array(self._belief_array)
            gpu_belief_set = BeliefSet(gpu_model, gpu_belief_array)
        else:
            gpu_belief_list = [Belief(gpu_model, cp.array(b.values)) for b in self._belief_list]
            gpu_belief_set = BeliefSet(gpu_model, gpu_belief_list)

        return gpu_belief_set
    

    def to_cpu(self) -> 'BeliefSet':
        '''
        Function returning an equivalent belief set object with the array of values stored on CPU instead of GPU.

        Returns
        -------
        cpu_belief_set : BeliefSet
            A new belief set with array on CPU.
        '''
        assert gpu_support, "GPU support is not enabled, unable to execute this function"

        cpu_model = self.model.cpu_model

        cpu_belief_set = None
        if self._belief_array is not None:
            cpu_belief_array = cp.asnumpy(self._belief_array)
            cpu_belief_set = BeliefSet(cpu_model, cpu_belief_array)
        
        else:
            cpu_belief_list = [Belief(cpu_model, cp.asnumpy(b.values)) for b in self._belief_list]
            cpu_belief_set = BeliefSet(cpu_model, cpu_belief_list)

        return cpu_belief_set