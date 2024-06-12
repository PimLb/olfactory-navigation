from datetime import datetime
from matplotlib import patches
from matplotlib import pyplot as plt

import os
import pandas as pd
import random

from olfactory_navigation.agents.model_based_util.belief import Belief, BeliefSet

from olfactory_navigation.agents.model_based_util.mdp import Model
from olfactory_navigation.agents.model_based_util.mdp import log
from olfactory_navigation.agents.model_based_util.mdp import COLOR_LIST, COLOR_ARRAY

import numpy as np
gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')

ilp_support = False
try:
    from scipy.optimize import milp, LinearConstraint
    ilp_support = True
except:
    print('[Warning] milp library not available, LP solvers will be disabled.')


class AlphaVector:
    '''
    A class to represent an Alpha Vector, a vector representing a plane in |S| dimension for POMDP models.


    Parameters
    ----------
    values : np.ndarray
        The actual vector with the value for each state.
    action : int
        The action associated with the vector.
    '''
    def __init__(self,
                 values: np.ndarray,
                 action: int
                 ) -> None:
        self.values = values
        self.action = int(action)


class ValueFunction:
    '''
    Class representing a set of AlphaVectors. One such set approximates the value function of the MDP model.


    Parameters
    ----------
    model : mdp.Model
        The model the value function is associated with.
    alpha_vectors : list[AlphaVector] or np.ndarray, optional
        The alpha vectors composing the value function, if none are provided, it will be empty to start with and AlphaVectors can be appended.
    action_list : list[int] or np.ndarray, optional
        The actions associated with alpha vectors in the case the alpha vectors are provided as an numpy array.
    
    Attributes
    ----------
    model : mdp.Model
        The model the value function is associated with.
    alpha_vector_list : list[AlphaVector]
    alpha_vector_array : np.ndarray
    actions : np.ndarray
    '''
    def __init__(self,
                 model: Model,
                 alpha_vectors: list[AlphaVector] | np.ndarray = [],
                 action_list: list[int] | np.ndarray = []
                 ) -> None:
        self.model = model

        self._vector_list = None
        self._vector_array = None
        self._actions = None

        self.is_on_gpu = False

        # List of alpha vectors
        if isinstance(alpha_vectors, list):
            assert all(v.values.shape[0] == model.state_count for v in alpha_vectors), f"Some or all alpha vectors in the list provided dont have the right size, they should be of shape: {model.state_count}"
            self._vector_list = alpha_vectors
            
            # Check if on gpu and make sure all vectors are also on the gpu
            if (len(alpha_vectors) > 0) and gpu_support and cp.get_array_module(alpha_vectors[0].values) == cp:
                assert all(cp.get_array_module(v.values) == cp for v in alpha_vectors), "Either all or none of the alpha vectors should be on the GPU, not just some."
                self.is_on_gpu = True
        
        # As numpy array
        else:
            av_shape = alpha_vectors.shape
            exp_shape = (len(action_list), model.state_count)
            assert av_shape == exp_shape, f"Alpha vector array does not have the right shape (received: {av_shape}; expected: {exp_shape})"

            self._vector_list = []
            for alpha_vect, action in zip(alpha_vectors, action_list):
                self._vector_list.append(AlphaVector(alpha_vect, action))

            # Check if array is on gpu
            if gpu_support and cp.get_array_module(alpha_vectors) == cp:
                self.is_on_gpu = True

        # Deduplication
        self._uniqueness_dict = {alpha_vector.values.tobytes(): alpha_vector for alpha_vector in self._vector_list}
        self._vector_list = list(self._uniqueness_dict.values())

        self._pruning_level = 1


    @property
    def alpha_vector_list(self) -> list[AlphaVector]:
        '''
        A list of AlphaVector objects. If the value function is defined as an matrix of vectors and a list of actions, the list of AlphaVectors will be generated from them.
        '''
        if self._vector_list is None:
            self._vector_list = []
            for alpha_vect, action in zip(self._vector_array, self._actions):
                self._vector_list.append(AlphaVector(alpha_vect, action))
        return self._vector_list
    

    @property
    def alpha_vector_array(self) -> np.ndarray:
        '''
        A matrix of size N x S, containing all the alpha vectors making up the value function. (N is the number of alpha vectors and S the amount of states in the model)
        If the value function is defined as a list of AlphaVector objects, the matrix will the generated from them.
        '''
        xp = cp if (gpu_support and self.is_on_gpu) else np

        if self._vector_array is None:
            self._vector_array = xp.array([v.values for v in self._vector_list])
            self._actions = xp.array([v.action for v in self._vector_list])
        return self._vector_array
    

    @property
    def actions(self) -> np.ndarray:
        '''
        A list of N actions corresponding to the N alpha vectors making up the value function.
        If the value function is defined as a list of AlphaVector objects, the list will the generated from the actions of those alpha vector objects.
        '''
        xp = cp if (gpu_support and self.is_on_gpu) else np

        if self._actions is None:
            self._vector_array = xp.array(self._vector_list)
            self._actions = xp.array([v.action for v in self._vector_list])
        return self._actions
    

    def __len__(self) -> int:
        return len(self._vector_list) if self._vector_list is not None else self._vector_array.shape[0]
    

    def __add__(self, other_value_function: 'Model') -> 'Model':
        # combined_dict = {**self._uniqueness_dict, **other_value_function._uniqueness_dict}
        combined_dict = {}
        combined_dict.update(self._uniqueness_dict)
        combined_dict.update(other_value_function._uniqueness_dict)

        # Instantiation of the new value function
        new_value_function = super().__new__(self.__class__)
        new_value_function.model = self.model
        new_value_function.is_on_gpu = self.is_on_gpu

        new_value_function._vector_list = list(combined_dict.values())
        new_value_function._uniqueness_dict = combined_dict
        new_value_function._pruning_level = 1

        new_value_function._vector_array = None
        new_value_function._actions = None

        return new_value_function


    def append(self,
               alpha_vector: AlphaVector
               ) -> None:
        '''
        Function to add an alpha vector to the value function.

        Parameters
        ----------
        alpha_vector : AlphaVector
            The alpha vector to be added to the value function.
        '''
        # Make sure size is correct
        assert alpha_vector.values.shape[0] == self.model.state_count, f"Vector to add to value function doesn't have the right size (received: {alpha_vector.values.shape[0]}, expected: {self.model.state_count})"
        
        # GPU support check
        xp = cp if (gpu_support and self.is_on_gpu) else np
        assert gpu_support and cp.get_array_module(alpha_vector.values) == xp, f"Vector is{' not' if self.is_on_gpu else ''} on GPU while value function is{'' if self.is_on_gpu else ' not'}."

        if self._vector_array is not None:
            self._vector_array = xp.append(self._vector_array, alpha_vector[None,:], axis=0)
            self._actions = xp.append(self._actions, alpha_vector.action)
        
        if self._vector_list is not None:
            self._vector_list.append(alpha_vector)


    def extend(self,
               other_value_function: 'Model'
               ) -> None:
        '''
        Function to add another value function is place.
        Effectively, it performs the union of the two sets of alpha vectors.

        Parameters
        ----------
        other_value_function : ValueFunction
            The other side of the union.
        '''
        self._uniqueness_dict.update(other_value_function._uniqueness_dict)
        self._vector_list = list(self._uniqueness_dict.values())

        self._vector_array = None
        self._actions = None

        self._pruning_level = 1


    def to_gpu(self) -> 'ValueFunction':
        '''
        Function returning an equivalent value function object with the arrays stored on GPU instead of CPU.

        Returns
        -------
        gpu_value_function : ValueFunction
            A new value function with arrays on GPU.
        '''
        assert gpu_support, "GPU support is not enabled, unable to execute this function"

        gpu_model = self.model.gpu_model

        gpu_value_function = None
        if self._vector_list is not None:
            gpu_alpha_vectors = [AlphaVector(cp.array(av.values), av.action) for av in self._vector_list]
            gpu_value_function = ValueFunction(gpu_model, gpu_alpha_vectors)

        else:
            gpu_vector_array = cp.array(self._vector_array)
            gpu_actions = self._actions if isinstance(self._actions, list) else cp.array(self._actions)
            gpu_value_function = ValueFunction(gpu_model, gpu_vector_array, gpu_actions)

        return gpu_value_function
    

    def to_cpu(self) -> 'ValueFunction':
        '''
        Function returning an equivalent value function object with the arrays stored on CPU instead of GPU.

        Returns
        -------
        cpu_value_function : ValueFunction
            A new value function with arrays on CPU.
        '''
        assert gpu_support, "GPU support is not enabled, unable to execute this function"

        cpu_model = self.model.cpu_model

        cpu_value_function = None
        if self._vector_list is not None:
            cpu_alpha_vectors = [AlphaVector(cp.asnumpy(av.values), av.action) for av in self._vector_list]
            cpu_value_function = ValueFunction(cpu_model, cpu_alpha_vectors)

        else:
            cpu_vector_array = cp.asnumpy(self._vector_array)
            cpu_actions = self._actions if isinstance(self._actions, list) else cp.asnumpy(self._actions)
            cpu_value_function = ValueFunction(cpu_model, cpu_vector_array, cpu_actions)
        
        return cpu_value_function


    def prune(self,
              level: int = 1
              ) -> None:
        '''
        Function pruning the set of alpha vectors composing the value function.
        The pruning is as thorough as the level:
            - 2: 1+ Check of absolute domination (check if dominated at each state).
            - 3: 2+ Solves Linear Programming problem for each alpha vector to see if it is dominated by combinations of other vectors.
        
        Note that the higher the level, the heavier the time impact will be.

        Parameters
        ----------
        level : int, default=1
            Between 0 and 3, how thorough the alpha vector pruning should be.
        '''
        # GPU support check
        xp = cp if (gpu_support and self.is_on_gpu) else np

        # Level 1 or under
        if level < self._pruning_level or level > 3:
            log('Attempting to prune a value function to a level already reached. Returning \'self\'')
            return

        # Level 2 pruning: Check for absolute domination
        if level >= 2 and self._pruning_level < 2:
            non_dominated_vector_indices = []

            for i, v in enumerate(self.alpha_vector_array):
                is_dom_by = xp.all(self.alpha_vector_array >= v, axis=1)
                if len(xp.where(is_dom_by)[0]) == 1:
                    non_dominated_vector_indices.append(i)

            self._vector_array = self._vector_array[non_dominated_vector_indices]
            self._actions = self._actions[non_dominated_vector_indices]

        # Level 3 pruning: LP to check for more complex domination
        if level >= 3:
            assert ilp_support, "ILP support not enabled..."

            pruned_alpha_set = pruned_alpha_set.to_cpu()

            alpha_set = pruned_alpha_set.alpha_vector_array
            non_dominated_vector_indices = []

            for i, alpha_vect in enumerate(alpha_set):
                other_alphas = alpha_set[:i] + alpha_set[(i+1):]

                # Objective function
                c = np.concatenate([np.array([1]), -1*alpha_vect])

                # Alpha vector contraints
                other_count = len(other_alphas)
                A = np.c_[np.ones(other_count), np.multiply(np.array(other_alphas), -1)]
                alpha_constraints = LinearConstraint(A, 0, np.inf)

                # Constraints that sum of beliefs is 1
                belief_constraint = LinearConstraint(np.array([0] + ([1]*self.model.state_count)), 1, 1)

                # Solve problem
                res = milp(c=c, constraints=[alpha_constraints, belief_constraint])

                # Check if dominated
                is_dominated = (res.x[0] - np.dot(res.x[1:], alpha_vect)) >= 0
                if is_dominated:
                    print(alpha_vect)
                    print(' -> Dominated\n')
                else:
                    non_dominated_vector_indices.append(i)

            self._vector_array = self._vector_array[non_dominated_vector_indices]
            self._actions = self._actions[non_dominated_vector_indices]

        # Update the tracked pruned level so far
        self._pruning_level = level


    def evaluate_at(self,
                    belief: Belief | BeliefSet
                    ) -> tuple[float | np.ndarray, int | np.ndarray]:
        '''
        Function to evaluate the value function at a belief point or at a set of belief points.
        It returns a value and the associated action.

        Parameters
        ----------
        belief : Belief or BeliefSet

        Returns
        -------
        value : float or np.ndarray
            The largest value associated with the belief point(s)
        action : int or np.ndarray
            The action(s) associated with the vector having the highest values at the belief point(s).
        '''
        # GPU support check
        xp = cp if (gpu_support and self.is_on_gpu) else np

        best_value = None
        best_action = None

        if isinstance(belief, Belief):
            # Computing values
            belief_values = xp.dot(self.alpha_vector_array, belief.values)

            # Selecting best vectors
            best_vector = xp.argmax(belief_values)

            # From best vector, compute the best value and action
            best_value = float(belief_values[best_vector])
            best_action = int(self.actions[best_vector])
        else:
            # Computing values
            belief_values = xp.matmul(belief.values if isinstance(belief, Belief) else belief.belief_array, self.alpha_vector_array.T)

            # Retrieving the top vectors according to the value function
            best_vectors = xp.argmax(belief_values, axis=1)

            # Retrieving the values and actions associated with the vectors chosen
            best_value = belief_values[xp.arange(belief_values.shape[0]), best_vectors]
            best_action = self.actions[best_vectors]

        return (best_value, best_action)


    def save(self,
             folder: str = './ValueFunctions',
             file_name: str | None = None
             ) -> None:
        '''
        Function to save the value function in a file at a given path. If no path is provided, it will be saved in a subfolder (ValueFunctions) inside the current working directory.
        If no file_name is provided, it be saved as '<current_timestamp>_value_function.csv'.

        Parameters
        ----------
        folder : str, default='./ValueFunctions'
            The path at which the npy file will be saved.
        file_name : str, default='<current_timestamp>_value_function.npy'
            The file name used to save in.
        '''
        if self.is_on_gpu:
            self.to_cpu().save(path=folder, file_name=file_name)
            return

        # Handle file_name
        if file_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = timestamp + '_value_function.npy'

        # Make sure that .csv is in the file name
        if '.npy' not in file_name:
            file_name += '.npy'

        # Getting array
        av_array = np.hstack([self.actions[:,None], self.alpha_vector_array])

        np.save(folder + '/' + file_name, av_array)


    @classmethod
    def load(cls,
             file: str,
             model: Model
             ) -> 'ValueFunction':
        '''
        Function to load the value function from a csv file.

        Parameters
        ----------
        file : str
            The path and file_name of the value function to be loaded.
        model : mdp.Model
            The model the value function is linked to.
            
        Returns
        -------
        loaded_value_function : ValueFunction
            The loaded value function.
        '''
        av_array = np.load(file)

        loaded_value_function = ValueFunction(model=model,
                                              alpha_vectors=av_array[:,1:],
                                              action_list=av_array[:,0].astype(int))

        return loaded_value_function


    def plot(self,
             as_grid: bool = False,
             size: int = 5,
             belief_set: np.ndarray = None
             ) -> None:
        '''
        Function to plot out the value function in 2 or 3 dimensions if possible and the as_grid parameter is kept to false. Else, the value function is plot as a grid.
        If a belief set array is provided and the model is a 2- or 3-model, it will be plot alongside the value function.

        Parameters
        ----------
        as_grid : bool, default=False
            Forces the plot to be plot as a grid.
        size : int, default=5
            The actual plot scale.
        belief_set : np.ndarray, optional
            A set of belief to plot the belief points that were explored.
        '''
        assert len(self) > 0, "Value function is empty, plotting is impossible..."
        
        # If on GPU, convert to CPU and plot that one
        if self.is_on_gpu:
            print('[Warning] Value function on GPU, converting to numpy before plotting...')
            cpu_value_function = self.to_cpu()
            cpu_value_function.plot(as_grid, size, belief_set)
            return

        func = None
        if as_grid:
            func = self._plot_grid
        elif self.model.state_count == 2:
            func = self._plot_2D
        elif self.model.state_count == 3:
            func = self._plot_3D
        else:
            print('[Warning] \'as_grid\' parameter set to False but state count is >3 so it will be plotted as a grid')
            func = self._plot_grid

        func(size, belief_set)


    def _plot_2D(self, size, belief_set=None):
        x = np.linspace(0, 1, 100)

        plt.figure(figsize=(int(size*1.5),size))
        grid_spec = {'height_ratios': ([1] if belief_set is None else [19,1])}
        _, ax = plt.subplots((2 if belief_set is not None else 1),1,sharex=True,gridspec_kw=grid_spec)

        # Vector plotting
        alpha_vects = self.alpha_vector_array

        m = alpha_vects[:,1] - alpha_vects[:,0] # type: ignore
        m = m.reshape(m.shape[0],1)

        x = x.reshape((1,x.shape[0])).repeat(m.shape[0],axis=0)
        y = (m*x) + alpha_vects[:,0].reshape(m.shape[0],1)

        ax0 = ax[0] if belief_set is not None else ax
        for i, alpha in enumerate(self.alpha_vector_list):
            ax0.plot(x[i,:], y[i,:], color=COLOR_LIST[alpha.action]['id']) # type: ignore

        # Set title
        title = 'Value function' + ('' if belief_set is None else ' and explored belief points')
        ax0.set_title(title)

        # X-axis setting
        ticks = [0,0.25,0.5,0.75,1]
        x_ticks = [str(t) for t in ticks]
        x_ticks[0] = self.model.state_labels[0]
        x_ticks[-1] = self.model.state_labels[1]

        ax0.set_xticks(ticks, x_ticks) # type: ignore

        # Action legend
        proxy = [patches.Rectangle((0,0),1,1,fc = COLOR_LIST[a]['id']) for a in self.model.actions]
        ax0.legend(proxy, self.model.action_labels, title='Actions') # type: ignore

        # Belief plotting
        if belief_set is not None:
            beliefs_x = belief_set.belief_array[:,1]
            ax[1].scatter(beliefs_x, np.zeros(beliefs_x.shape[0]), c='red')
            ax[1].get_yaxis().set_visible(False)
            ax[1].axhline(0, color='black')
            ax[1].set_xlabel('Belief space')
        else:
            ax0.set_xlabel('Belief space')
        
        # Axis labels
        ax0.set_ylabel('V(b)')


    def _plot_3D(self, size, belief_set=None):

        def get_alpha_vect_z(xx, yy, alpha_vect):
            x0, y0, z0 = [0, 0, alpha_vect[0]]
            x1, y1, z1 = [1, 0, alpha_vect[1]]
            x2, y2, z2 = [0, 1, alpha_vect[2]]

            ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
            vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]

            u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

            point  = np.array([0, 0, alpha_vect[0]])
            normal = np.array(u_cross_v)

            d = -point.dot(normal)

            z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
            
            return z

        def get_plane_gradient(alpha_vect):
        
            x0, y0, z0 = [0, 0, alpha_vect[0]]
            x1, y1, z1 = [1, 0, alpha_vect[1]]
            x2, y2, z2 = [0, 1, alpha_vect[2]]

            ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
            vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]

            u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]
            
            normal_vector = np.array(u_cross_v)
            normal_vector_norm = float(np.linalg.norm(normal_vector))
            normal_vector = np.divide(normal_vector, normal_vector_norm)
            normal_vector[2] = 0
            
            return np.linalg.norm(normal_vector)

        # Actual plotting
        x = np.linspace(0, 1, 1000)
        y = np.linspace(0, 1, 1000)

        xx, yy = np.meshgrid(x, y)

        max_z = np.zeros((xx.shape[0], yy.shape[0]))
        best_a = (np.zeros((xx.shape[0], yy.shape[0])))
        plane = (np.zeros((xx.shape[0], yy.shape[0])))
        gradients = (np.zeros((xx.shape[0], yy.shape[0])))

        for alpha in self.alpha_vector_list:

            z = get_alpha_vect_z(xx, yy, alpha.values)

            # Action array update
            new_a_mask = np.argmax(np.array([max_z, z]), axis=0)

            best_a[new_a_mask == 1] = alpha.action
            
            plane[new_a_mask == 1] = random.randrange(100)
            
            alpha_gradient = get_plane_gradient(alpha.values)
            gradients[new_a_mask == 1] = alpha_gradient

            # Max z update
            max_z = np.max(np.array([max_z, z]), axis=0)
            
        for x_i, x_val in enumerate(x):
            for y_i, y_val in enumerate(y):
                if (x_val+y_val) > 1:
                    max_z[x_i, y_i] = np.nan
                    plane[x_i, y_i] = np.nan
                    gradients[x_i, y_i] = np.nan
                    best_a[x_i, y_i] = np.nan

        belief_points = None
        if belief_set is not None:
            belief_points = belief_set.belief_array[:,1:]
                    
        fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(size*4,size*3.5), sharex=True, sharey=True)

        # Set ticks
        ticks = [0,0.25,0.5,0.75,1]
        x_ticks = [str(t) for t in ticks]
        x_ticks[0] = self.model.state_labels[0]
        x_ticks[-1] = self.model.state_labels[1]
        
        y_ticks = [str(t) for t in ticks]
        y_ticks[0] = self.model.state_labels[0]
        y_ticks[-1] = self.model.state_labels[2]

        plt.setp([ax1,ax2,ax3,ax4], xticks=ticks, xticklabels=x_ticks, yticks=ticks, yticklabels=y_ticks)

        # Value function ax
        ax1.set_title("Value function")
        ax1_plot = ax1.contourf(x, y, max_z, 100, cmap="viridis")
        plt.colorbar(ax1_plot, ax=ax1)

        # Alpha planes ax
        ax2.set_title("Alpha planes")
        ax2_plot = ax2.contourf(x, y, plane, 100, cmap="viridis")
        plt.colorbar(ax2_plot, ax=ax2)
        
        # Gradient of planes ax
        ax3.set_title("Gradients of planes")
        ax3_plot = ax3.contourf(x, y, gradients, 100, cmap="Blues")
        plt.colorbar(ax3_plot, ax=ax3)

        # Action policy ax
        ax4.set_title("Action policy")
        ax4.contourf(x, y, best_a, 1, colors=[c['id'] for c in COLOR_LIST])
        proxy = [patches.Rectangle((0,0),1,1,fc = COLOR_LIST[int(a)]['id']) for a in self.model.actions]
        ax4.legend(proxy, self.model.action_labels, title='Actions')

        if belief_points is not None:
            for ax in [ax1,ax2,ax3,ax4]:
                ax.scatter(belief_points[:,0], belief_points[:,1], s=1, c='black')


    def _plot_grid(self, size=5, belief_set=None):
        value_table = np.max(self.alpha_vector_array, axis=0)[self.model.state_grid]
        best_action_table = np.array(self.actions)[np.argmax(self.alpha_vector_array, axis=0)][self.model.state_grid]
        best_action_colors = COLOR_ARRAY[best_action_table]

        dimensions = self.model.state_grid.shape

        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(size*2, size), width_ratios=(0.55,0.45))

        # Ticks
        x_ticks = np.arange(0, dimensions[1], (1 if dimensions[1] < 10 else int(dimensions[1] / 10)))
        y_ticks = np.arange(0, dimensions[0], (1 if dimensions[0] < 5 else int(dimensions[0] / 5)))

        ax1.set_title('Value function')
        ax1_plot = ax1.imshow(value_table)

        if dimensions[0] >= dimensions[1]: # If higher than wide 
            plt.colorbar(ax1_plot, ax=ax1)
        else:
            plt.colorbar(ax1_plot, ax=ax1, location='bottom', orientation='horizontal')
        
        ax1.set_xticks(x_ticks)
        ax1.set_yticks(y_ticks)

        ax2.set_title('Action policy')
        ax2.imshow(best_action_colors)
        p = [ patches.Patch(color=COLOR_LIST[int(i)]['id'], label=str(self.model.action_labels[int(i)])) for i in self.model.actions]
        ax2.legend(handles=p, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Actions')
        ax2.set_xticks(x_ticks)
        ax2.set_yticks(y_ticks)