import h5py
import json
import os
import scipy
import shutil

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scipy.interpolate import RegularGridInterpolator
from typing import Literal

import numpy as np
gpu_support = False
try:
    import cupy as cp
    from cupyx.scipy.interpolate import RegularGridInterpolator as CupyRegularGridInterpolator
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')


def _resize_array(array: np.ndarray,
                  new_shape: tuple,
                  interpolation: str
                  ) -> np.ndarray:
    '''
    Function to resize a numpy array in the same fashion as the cv2.resize function.
    It uses the scipy.interpolate
    '''
    # GPU support
    xp = cp.get_array_module(array) if gpu_support else np
    Interpolator = None
    if gpu_support and (xp == cp):
        Interpolator = CupyRegularGridInterpolator
    else:
        Interpolator = RegularGridInterpolator

    # Gathering initial shape and indices along each axis
    shape = array.shape
    indices = [xp.linspace(start=0, stop=(ax_shape-1), num=ax_shape) for ax_shape in shape]

    # Building the Interpolator
    interp = Interpolator((*indices,), array, method=interpolation)

    # Building new indices along each axis and building all gridpoints
    new_indices = [xp.linspace(start=0, stop=(ax_shape-1), num=ax_new_shape) for ax_shape, ax_new_shape in zip(shape, new_shape)]
    new_grid_points =  xp.meshgrid(*new_indices, indexing='ij')

    # Generating new data points on this new grid
    new_array = interp((*new_grid_points,))

    return new_array


class Environment:
    '''
    Class to represent an olfactory environment.

    It is defined based on an olfactory data set provided as either a numpy file or an array directly with shape time, y, x.
    From this environment, the various parameters are applied in the following order:

    0. The source position is set
    1. The margins are added and the shape (total size) of the environment are set.
    2. The data file's x and y components are squished and streched the to fit the inter-marginal shape of the environment.
    3. The source's position is also moved to stay at the same position within the data.
    4. The multiplier is finally applied to modify the data file's x and y components a final time by growing or shrinking the margins to account for the multiplier. (The multiplication applies with the source position as a center point)

    Note: to modify the shape of the data file's x and y components the OpenCV library's resize function is used. And the interpolation method is controlled by the interpolation_method parameter.


    Then, the starting probability map is built. Either an array can be provided directly or preset option can be chosen:

    - 'data_zone': The agent can start at any point in the data_zone (after all the modification parameters have been applied)
    - 'odor_present': The agent can start at any point where an odor cue above the odor_present_threshold can be found at any timestep during the simulation

    Parameters
    ----------
    data_file : str or np.ndarray
        The dataset containing the olfactory data. It can be provided as a path to a file containing said array.
    data_source_position : list or np.ndarray
        The center point of the source provided as a list or a 1D array with the components being x,y.
        This position is computed in the olfactory data zone (so excluding the margins).
    source_radius : float, default=1.0
        The radius from the center point of the source in which we consider the agent has reached the source.
    layers : bool or list[int] or list[str], default=False
        Whether or not the data provided contains layers or not.
        If a list of strings is provided, it will be either used to name the layers found (if numpy data), or it is used to querry the datasets of the h5 file.
    shape : list or np.ndarray, optional
        A 2-element array or list of how many units should be kept in the final array (including the margins).
        As it should include the margins, the shape should be strictly larger than the sum of the margins in each direction.
        By default, the shape of the olfactory data will be maintained.
    margins : int or list or np.ndarray, default=0
        How many units have to be added to the data as margins. (Before the multiplier is applied)
        If a unique element is provided, the margin will be this same value on each side.
        If a list or array of 2 elements is provided, the first number will be vertical margins (y-axis), while the other will be on the x-axis (horizontal).
    multiplier : list or np.ndarray, default=[1.0,1.0]
        A 2-element array or list of how much the odor field should be streched in each direction.
        If a value larger than 1 is provided, the margins will be reduced to accomodate for the larger size of the olfactory data size.
        And inversly, less than 1 will increase the margins.
        By default, the multipliers will be set to 1.0.
    interpolation_method : 'Nearest' or 'Linear' or 'Cubic', default='Linear'
        The interpolation method to be used in the case the data needs to be reshaped to fit the shape, margins and multiplier parameters.
        By default, it uses Bi-linear interpolation. The interpolation is performed using the OpenCV library.
    preprocess_data : bool, default=False
        Applicable only for data_file being a path to a h5 file.
        Whether to reshape of the data at the creation of the environment.
        Reshaping the data ahead of time will require more processing at the creation and more memory overall.
        While if this is disabled, when gathering observations, more time will be required but less memory will need to be used at once.
    boundary_condition : 'stop' or 'wrap' or 'wrap_vertical' or 'wrap_horizontal' or 'clip', default='stop'
        How the agent should behave at the boundary.
        Stop means for the agent to stop at the boundary, if the agent tries to move north while being on the top edge, it will stay in the same state.
        Wrap means for the borders to be like portals, when entering on one side, it reappears on the other side.
        Wrap can be specified to be only vertically or horizontally
    start_zone : 'odor_present' or 'data_zone' or np.ndarray, default='data_zone'
        Either an array or a string representing how the starting probabilities should be constructed.
        - odor_present: The start probabilities will be uniform where odor cues can be found above 0 (or a given odor_present_threshold)
        - data_zone: Uniform over the data zone, so without the margins.
        Note that the points within the source radius will be excluded from this probability grid.
    odor_present_threshold : float, optional
        An olfactory threshold, under which the odor is considered too low to be noticed.
        It is used only to build the starting zone if the 'odor_present' option is selected.
    name : str, optional
        A custom name to be given to the agent.
        If it is not provided, by default it will have the format:
        <shape>-marg_<margins>-edge_<boundary_condition>-start_<start_zone>-source_<source_point>_radius<source_radius>
    seed : int, default=12131415
        For reproducible randomness.

    Attributes
    ----------
    data : np.ndarray
        An array containing the olfactory data after the modification parameters have been applied.
    data_file_path : str
        If the data is loaded from a path, the path will be recorded here.
    data_source_position : np.ndarray
        The position of the source in the original data file (after modifications have been applied).
    layers : np.ndarray
        A numbered list of the IDs of the layers.
    layer_labels : list[str]
        A list of how the layers are named.
    has_layers : bool
        Whether or not the environment is made up of layers.
    margins : np.ndarray
        An array of the margins vertically and horizontally (after multiplier is applied).
    timestamps : int
        The amount of timeslices available in the environment.
    data_shape : tuple[int]
        The shape of the data's odor field (after modifications have been applied).
    dimensions : int
        The amount of dimensions of the physical space of the olfactory environment.
    shape : tuple[int]
        The shape of the environment. It is a tuple of the size in each axis of the environment.
    data_bounds : np.ndarray
        The bounds between which the original olfactory data stands in the coordinate system of the environment (after modifications have been applied).
    source_position : np.ndarray
        The position of the source in the padded grid (after modifications have been applied).
    source_radius : float
        The radius of the source.
    interpolation_method : str
        The interpolation used to modify the shape of the original data.
    data_processed : bool
        Whether the data was processed (ie the shape is at it should be) or not.
    boundary_condition : str
        How the agent should behave when reaching the boundary.
    start_probabilities : np.ndarray
        A probability map of where the agent is likely to start within the environment.
        Note: Zero within the source radius.
    start_type : str
        The type of the start probability map building. For instance: 'data_zone', 'odor_present', or 'custom' (if an array is provided).
    odor_present_threshold : float
        The threshold used to uild the start probabilities if the option 'odor_present' is used.
    name : str
        The name set to the agent as defined in the parameters.
    saved_at : str
        If the environment is saved, the path at which it is saved will be recorded here.
    on_gpu : bool
        Whether the environment's arrays are on the gpu's memory or not.
    seed : int
        The seed used for the random operations (to allow for reproducability).
    rnd_state : np.random.RandomState
        The random state variable used to generate random values.
    cpu_version : Environment
        An instance of the environment on the CPU. If it already is, it returns itself.
    gpu_version : Environment
        An instance of the environment on the CPU. If it already is, it returns itself.
    '''
    def __init__(self,
                 data_file: str | np.ndarray,
                 data_source_position: list | np.ndarray,
                 source_radius: float = 1.0,
                 layers: bool | list[str] = False,
                 shape: list | np.ndarray | None = None,
                 margins: int | list | np.ndarray = 0,
                 multiplier: list| np.ndarray = [1.0, 1.0],
                 interpolation_method: Literal['Nearest', 'Linear', 'Cubic'] = 'Linear',
                 preprocess_data: bool = False,
                 boundary_condition: Literal['stop', 'wrap', 'wrap_vertical', 'wrap_horizontal', 'clip', 'no'] = 'stop',
                 start_zone: Literal['odor_present', 'data_zone'] | np.ndarray = 'data_zone',
                 odor_present_threshold: float | None = None,
                 name: str | None = None,
                 seed: int = 12131415,
                 ) -> None:
        self.saved_at: str = None

        # Layer properties
        self.layers = None
        self.layer_labels = None
        self.has_layers = False

        if isinstance(layers, list):
            self.has_layers = True
            self.layers = np.arange(len(layers))
            self.layer_labels = [layer for layer in layers]
        elif isinstance(layers, bool):
            self.has_layers = layers

        # Load from file if string provided
        self.data_file_path = None
        self._preprocess_data: bool = preprocess_data

        loaded_data = None
        if isinstance(data_file, str):
            self.data_file_path = data_file

            # NUMPY
            if data_file.endswith('.npy'):
                loaded_data = np.load(data_file)

                # Layered data
                if self.has_layers:
                    if self.layers is None:
                        self.layers = np.arange(len(loaded_data))
                        self.layer_labels = [str(layer) for layer in range(len(loaded_data))]
                    else:
                        assert (len(self.layers) == len(loaded_data)), "The amount of layers provided dont match the amount in the dataset."

                        # Re-ordering the layers
                        loaded_data = loaded_data[self.layers]

            # H5
            elif data_file.endswith('.h5'):
                loaded_data = h5py.File(data_file,'r')

                # Layered data
                if self.has_layers:

                    # Converting layers to strings
                    data_layer_labels = list(loaded_data.keys())
                    if self.layers is None:
                        self.layers = np.arange(len(data_layer_labels))
                        self.layer_labels = data_layer_labels

                    # Getting the labels based on the list of integers provided
                    elif all(isinstance(layer, int) for layer in layers):
                        self.layer_labels = [data_layer_labels[layer_id] for layer_id in self.layers]

                    # Loading the list of slices from the data
                    loaded_data = [[loaded_data[layer][f"{t}"] for t in range(len(loaded_data[layer]))] for layer in self.layer_labels]

                else:
                    loaded_data = [loaded_data[f"{t}"] for t in range(len(loaded_data))]

            # Not supported
            else:
                raise NotImplementedError('File format loading not implemented')

        elif not isinstance(data_file, np.ndarray):
            raise NotImplementedError("Data file should be either a path or an object that is either an h5 object or a numpy array")

        self._data: np.ndarray = loaded_data if loaded_data is not None else data_file

        # Unmodified sizes
        self.timesteps = len(self._data if not self.has_layers else self._data[0])
        self.data_shape = (self._data[0] if not self.has_layers else self._data[0][0]).shape
        self.dimensions = len(self.data_shape)
        self.data_source_position = np.array(data_source_position)
        self.original_data_source_position = self.data_source_position

        original_data_shape = self.data_shape

        # Making margins a |dims|x2 array
        if isinstance(margins, int):
            self.margins = np.ones((self.dimensions, 2), dtype=int) * margins
        elif isinstance(margins, list) or isinstance(margins, np.ndarray):
            margins = np.array(margins)
            if margins.shape == (self.dimensions,): # Symmetric min and max margins
                self.margins = np.hstack((margins[:,None], margins[:,None]))
            elif margins.shape == (self.dimensions,2):
                self.margins = margins
            else:
                raise ValueError('The array or lists of Margins provided have a shape not supported. (Supported formats (2,) or (2,2))')
        else:
            raise ValueError('margins argument should be either an integer or a 1D or 2D array with either shape (2) or (2,2)')
        assert (self.margins.dtype == int), 'margins should be integers'

        # Process shape parameter
        new_data_shape = None
        if shape is not None:
            shape = np.array(shape)

            assert np.all(shape > np.sum(self.margins, axis=1)), "The shape of the environment must be strictly larger than the sum of margins."

            # Computing the new shape of the data
            new_data_shape: np.ndarray = (shape - np.sum(self.margins, axis=1)).astype(int)

            # New source position
            self.data_source_position = (self.data_source_position * (new_data_shape / self.data_shape)).astype(int)
        else:
            shape = self.data_shape + np.sum(self.margins, axis=1)

        if new_data_shape is not None:
            self.data_shape = (*new_data_shape,)

        # Process multiplier
        multiplier = np.array(multiplier)

        # Assert multiplier value is correct
        with np.errstate(divide='ignore'):
            low_max_mult = ((self.margins[:,0] / self.data_source_position) + 1)
            high_max_mult = (1 + (self.margins[:,1] / (self.data_shape - self.data_source_position)))
            max_mult = np.min(np.vstack([low_max_mult, high_max_mult]), axis=0)

            assert np.all(multiplier <= max_mult), f"The multiplier given is larger than allowed (the values should be lower than {max_mult})"

        # Compute new data shape with the multiplier
        if new_data_shape is None:
            new_data_shape = self.data_shape
        new_data_shape = (new_data_shape * multiplier).astype(int)

        # New source position based on multiplier
        new_source_position = (self.data_source_position * multiplier).astype(int)

        # Recomputing margins with new source position
        self.margins[:,0] -= (new_source_position - self.data_source_position)
        self.margins[:,1] = (shape - (self.margins[:,0] + new_data_shape))

        # Re-Setting new source position
        self.data_source_position = new_source_position

        # Interpolation method choice
        self.interpolation_method = interpolation_method

        # Input the new shape of the data if set by custom shape or multiplier
        if new_data_shape is not None:
            self.data_shape: tuple[int] = (*new_data_shape,)

        # Check if data is already processed by default
        self.data_processed = (self.data_shape == original_data_shape)

        # If requested process all the slices of data into a single
        if preprocess_data and not self.data_processed:
            assert self.dimensions == 2, "Higher dimensional data doesnt support reshaping yet, ensure it is done beforehand.."
            if self.has_layers:
                new_data = np.zeros((len(self.layers), self.timesteps, *self.data_shape))
                for layer in self.layers:
                    for i in range(self.timesteps):
                        new_data[layer, i] = _resize_array(np.array(self._data[layer][i]),
                                                           new_shape=self.data_shape,
                                                           interpolation=self.interpolation_method.lower())
            else:
                new_data = np.zeros((self.timesteps, *self.data_shape))
                for i in range(self.timesteps):
                    new_data[i] = _resize_array(np.array(self._data[i]),
                                                new_shape=self.data_shape,
                                                interpolation=self.interpolation_method.lower())
            self._data = new_data
            self.data_processed = True

        # Reading shape of data array
        self.shape = (*(self.data_shape + np.sum(self.margins, axis=1)),)

        # Converting the shape tuple to integer sets
        self.shape: tuple[int] = tuple([int(el) for el in self.shape])
        self.data_shape: tuple[int] = tuple([int(el) for el in self.data_shape])

        # Building a data bounds
        self.data_bounds = np.array([self.margins[:,0], self.margins[:,0] + np.array(self.data_shape)]).T

        # Saving arguments
        self.source_position = self.data_source_position + self.margins[:,0]
        self.source_radius = source_radius

        # Boundary conditions
        assert not ((self.dimensions > 2) and (boundary_condition in ['wrap_vertical', 'wrap_horizontal'])), "There are more than 2 dimensions, the options of 'wrap_horizontal' and 'wrap_vertical' are disabled."
        self.boundary_condition = boundary_condition

        # Starting zone
        self.start_probabilities = np.zeros(self.shape)
        self.start_type = start_zone if isinstance(start_zone, str) else 'custom'

        if isinstance(start_zone, np.ndarray):
            if start_zone.shape == (self.dimensions,2):
                slices = tuple(slice(low, high) for low, high in start_zone)
                self.start_probabilities[slices] = 1.0
                self.start_type += '_' + '_'.join([str(el) for el in start_zone.ravel()])
            elif start_zone.shape == self.shape:
                self.start_probabilities = start_zone
            else:
                raise ValueError('If an np.ndarray is provided for the start_zone it has to be |dim| x 2...')

        elif start_zone == 'data_zone':
            slices = tuple(slice(low, high) for low, high in self.data_bounds)
            self.start_probabilities[slices] = 1.0

        elif start_zone == 'odor_present':
            if self.data_processed and isinstance(self._data, np.ndarray):
                odor_present_map = (np.mean((self._data > (odor_present_threshold if odor_present_threshold is not None else 0)).astype(int), axis=0) > 0).astype(float)
                self.start_probabilities[tuple(slice(low, high) for low, high in self.data_bounds)] = odor_present_map
            else:
                odor_sum = np.zeros(self.data_shape, dtype=float)
                for i in range(self.timesteps):
                    data_slice = np.array(self._data[i]) if not self.has_layers else np.array(self._data[0][i])
                    reshaped_data_slice = _resize_array(data_slice,
                                                        new_shape=self.data_shape,
                                                        interpolation=self.interpolation_method.lower())
                    odor_sum += (reshaped_data_slice > (odor_present_threshold if odor_present_threshold is not None else 0))
                self.start_probabilities[tuple(slice(low, high) for low, high in self.data_bounds)] = (odor_sum / self.timesteps)
        else:
            raise ValueError('start_zone value is wrong')

        # Odor present tresh
        self.odor_present_threshold = odor_present_threshold

        # Removing the source area from the starting zone
        source_mask = np.fromfunction((lambda *points: np.sum((np.array(points).transpose([i+1 for i in range(len(self.shape))] + [0]) - self.source_position[None,:])**2, axis=-1) <= self.source_radius**2), shape=self.shape)
        self.start_probabilities[source_mask] = 0
        self.start_probabilities /= np.sum(self.start_probabilities) # Normalization

        # Name
        self.name = name
        if self.name is None:
            self.name =  '_'.join([str(axis_size) for axis_size in self.shape]) # Size of env
            self.name += f'-marg_' + '_'.join(['_'.join([str(marg) for marg in dim_margins]) for dim_margins in self.margins]) # margins
            self.name += f'-edge_{self.boundary_condition}' # Boundary condition
            self.name += f'-start_{self.start_type}' # Start zone
            self.name += f'-source_' + '_'.join([str(pos) for pos in self.source_position]) + f'_radius{self.source_radius}' # Source

        # gpu support
        self._alternate_version = None
        self.on_gpu = False

        # random state
        self.seed = seed
        self.rnd_state = np.random.RandomState(seed = seed)


    @property
    def data(self) -> np.ndarray:
        '''
        The whole dataset with the right shape. If not preprocessed to modify its shape the data will be processed when querrying this object.
        '''
        if not self._data_is_numpy or not self.data_processed:
            xp = cp if self.on_gpu else np
            print('[Warning] The whole dataset is being querried, it will be reshaped at this time. To avoid this, avoid querrying environment.data directly.')

            # Reshaping
            if self.has_layers:
                new_data = np.zeros((len(self.layers), self.timesteps, *self.data_shape))
                for layer in self.layers:
                    for i in range(self.timesteps):
                        new_data[layer, i] = _resize_array(np.array(self._data[layer][i]),
                                                           new_shape=self.data_shape,
                                                           interpolation=self.interpolation_method.lower())
            else:
                new_data = np.zeros((self.timesteps, *self.data_shape))
                for i in range(self.timesteps):
                    new_data[i] = _resize_array(np.array(self._data[i]),
                                                new_shape=self.data_shape,
                                                interpolation=self.interpolation_method.lower())

            self._data = xp.array(new_data)
            self.data_processed = True

        return self._data


    @property
    def _data_is_numpy(self) -> bool:
        '''
        Wheter the data is a numpy array or not.
        '''
        xp = cp if self.on_gpu else np
        return isinstance(self._data, xp.ndarray)


    def plot(self,
             frame: int = 0,
             layer: int = 0,
             ax: plt.Axes | None = None
             ) -> None:
        '''
        Simple function to plot the environment with a single frame of odor cues.
        The starting zone is also market down with a blue contour.
        The source of the odor is marked by a red circle.

        Parameters
        ----------
        frame : int, default=0
            The frame of odor cues to print.
        layer : int, default=0
            The layer of the odor cues to print. (Ignored if the environment is not layered.)
        ax : plt.Axes, optional
            An ax on which the environment can be plot.
        '''
        # If on GPU use the CPU version to plot
        if self.on_gpu:
            self._alternate_version.plot(
                frame=frame,
                ax=ax
            )
            return # Blank return

        # TODO: Implement plotting for 3D
        assert self.dimensions == 2, "Plotting function only available for 2D environments for now..."

        if ax is None:
            _, ax = plt.subplots(1, figsize=(15,5))

        legend_elements = [[],[]]

        # Gather data frame
        data_frame: np.ndarray = self._data[layer][frame] if self.has_layers else self._data[frame]
        if not isinstance(data_frame, np.ndarray):
            data_frame = np.array(data_frame)

        if not self.data_processed:
            data_frame = _resize_array(data_frame,
                                       new_shape=self.data_shape,
                                       interpolation=self.interpolation_method.lower())

        # Odor grid
        odor = Rectangle([0,0], 1, 1, color='black', fill=True)
        frame_data = (data_frame > (self.odor_present_threshold if self.odor_present_threshold is not None else 0)).astype(float)
        environment_frame = np.zeros(self.shape, dtype=float)
        environment_frame[self.data_bounds[0,0]:self.data_bounds[0,1], self.data_bounds[1,0]:self.data_bounds[1,1]] = frame_data
        ax.imshow(environment_frame, cmap='Greys')

        legend_elements[0].append(odor)
        legend_elements[1].append(f'Frame {frame}' + ('' if not self.has_layers else f' (layer {layer})') + ' odor cues')

        # Start zone contour
        start_zone = Rectangle([0,0], 1, 1, color='blue', fill=False)
        ax.contour(self.start_probabilities, levels=[0.0], colors='blue')

        legend_elements[0].append(start_zone)
        legend_elements[1].append('Start zone')

        # Source circle
        goal_circle = Circle(self.source_position[::-1], self.source_radius, color='r', fill=False, zorder=10)
        legend_elements[0].append(goal_circle)
        legend_elements[1].append('Source')

        if self.source_radius > 0.0:
            ax.add_patch(goal_circle)
        else:
            ax.scatter(self.source_position[1], self.source_position[0], c='red')

        # Legend
        ax.legend(legend_elements[0], legend_elements[1])


    def get_observation(self,
                        pos: np.ndarray,
                        time: int | np.ndarray = 0,
                        layer: int | np.ndarray = 0
                        ) -> float | np.ndarray:
        '''
        Function to get an observation at a given position on the grid at a given time.
        A set of observations can also be requested, either at a single position for multiple timestamps or with the same amoung of positions as timestamps provided.

        Note: The position will not be checked against boundary conditions; if a position is out-of-bounds it will simply return 0.0!

        Parameters
        ----------
        pos : np.ndarray
            The position or list of positions to get observations at.
        time : int or np.ndarray, default=0
            A timestamp or list of timestamps to get the observations at.
        layer : int or np.ndarray, default=0
            A layer or list of timestamps to get the observations at.
            Note: If the environment doesnt have layers, this parameter will be ignored.

        Returns
        -------
        observation : float or np.ndarray
            A single observation or list of observations.
        '''
        xp = cp if self.on_gpu else np

        # Handling the case of a single point
        is_single_point = (len(pos.shape) == 1)
        if is_single_point:
            pos = pos[None,:]

        # Counting how many position points we are dealing with
        pos_count = len(pos)

        # Time looping
        time = time % self.timesteps

        # Determine unique layers and reindexing them if needed
        unique_layers = xp.array([layer]) if isinstance(layer, int) else xp.unique(layer)
        layer = 0 if isinstance(layer, int) else xp.where(layer == unique_layers[:,None])[0]
        layer_count = len(unique_layers)

        # Determine unique times and reindexing them if needed
        unique_times = xp.array([time]) if isinstance(time, int) else xp.unique(time)
        unique_time_indices = 0 if isinstance(time, int) else xp.where(time == unique_times[:,None])[0]
        time_count = len(unique_times)

        # Handling the case where the data is a sequence of slices (h5, so not numpy array)
        data = self._data

        # Selecting the required slices
        if self._data_is_numpy:
            data = data[unique_layers][:,unique_times] if self.has_layers else data[unique_times]
        else:
            # Case where we are dealing with a h5 file
            # Note: Can't use self.data_shape because we don't know whether the data is processed yet or no
            selected_slices = xp.zeros((layer_count, time_count, *self._data[0][0].shape)) if self.has_layers else xp.zeros((time_count, *self._data[0].shape))
            for i, t in enumerate(unique_times):
                if self.has_layers:
                    for j, l in enumerate(unique_layers):
                        selected_slices[j,i] = xp.array(data[int(l)][int(t)])
                else:
                    selected_slices[i] = xp.array(data[t])
            data = xp.array(selected_slices)

        # Handle the case it needs to be processed on the fly
        if not self.data_processed:
            reshaped_data = xp.zeros((layer_count, time_count, *self.data_shape)) if self.has_layers else xp.zeros((time_count, *self.data_shape))

            for i in range(time_count):
                if self.has_layers:
                    for j in range(layer_count):
                        reshaped_data[j,i] = _resize_array(data[j,i],
                                                           new_shape=self.data_shape,
                                                           interpolation=self.interpolation_method.lower())
                else:
                    reshaped_data[i] = _resize_array(data[i],
                                                     new_shape=self.data_shape,
                                                     interpolation=self.interpolation_method.lower())

            data = xp.array(reshaped_data)

        # Return 0.0 if outside of data zone
        data_pos = pos - self.margins[:,0][None,:]
        data_pos_valid = xp.all((data_pos >= 0) & (data_pos < xp.array(self.data_shape)), axis=1)
        observation = xp.zeros(pos_count, dtype=float)

        # Gathering data on layered data on not
        if self.has_layers:
            observation[data_pos_valid] = data[(layer if isinstance(layer, int) else layer[data_pos_valid]), # layer
                                               (unique_time_indices if isinstance(unique_time_indices, int) else unique_time_indices[data_pos_valid]), # t
                                               *data_pos[data_pos_valid,:].T] # physical position
        else:
            observation[data_pos_valid] = data[(unique_time_indices if isinstance(unique_time_indices, int) else unique_time_indices[data_pos_valid]), # t
                                               *data_pos[data_pos_valid,:].T] # physical position

        return float(observation[0]) if is_single_point else observation


    def source_reached(self,
                       pos: np.ndarray
                       ) -> bool | np.ndarray:
        '''
        Checks whether a given position is within the source radius.

        Parameters
        ----------
        pos : np.ndarray
            The position to check whether in the radius of the source.

        Returns
        -------
        is_at_source : bool
            Whether or not the position is within the radius of the source.
        '''
        xp = cp if self.on_gpu else np

        # Handling the case of a single point
        is_single_point = (len(pos.shape) == 1)
        if is_single_point:
            pos = pos[None,:]

        is_at_source: np.ndarray = (xp.sum((pos - self.source_position[None,:]) ** 2, axis=-1) <= (self.source_radius ** 2))

        return bool(is_at_source[0]) if is_single_point else is_at_source


    def random_start_points(self,
                            n: int = 1
                            ) -> np.ndarray:
        '''
        Function to generate n starting positions following the starting probabilities.

        Parameters
        ----------
        n : int, default=1
            How many random starting positions to generate

        Returns
        -------
        random_states_2d : np.ndarray
            The n random 2d points in a n x 2 array.
        '''
        xp = cp if self.on_gpu else np

        assert (n > 0), "n has to be a strictly positive number (>0)"

        random_states = self.rnd_state.choice(xp.arange(int(np.prod(self.shape))), size=n, replace=True, p=self.start_probabilities.ravel())
        random_states_2d = xp.array(xp.unravel_index(random_states, self.shape)).T
        return random_states_2d


    def move(self,
             pos: np.ndarray,
             movement: np.ndarray
             ) -> np.ndarray:
        '''
        Applies a movement vector to a position point and returns a new position point while respecting the boundary conditions.

        Parameters
        ----------
        pos : np.ndarray
            The start position of the movement.
        movement : np.ndarray
            A 2D movement vector.

        Returns
        -------
        new_pos : np.ndarray
            The new position after applying the movement.
        '''
        xp = cp if self.on_gpu else np

        # Applying the movement vector
        new_pos = pos + movement

        # Handling the case we are dealing with a single point.
        is_single_point = (len(pos.shape) == 1)
        if is_single_point:
            new_pos = new_pos[None,:]

        shape_array = xp.array(self.shape)[None,:]

        # Wrap boundary
        if self.boundary_condition == 'wrap':
            new_pos = xp.where(new_pos < 0, (new_pos + shape_array), new_pos)
            new_pos = xp.where(new_pos >= shape_array, (new_pos - shape_array), new_pos)

        # Stop boundary
        elif self.boundary_condition == 'stop':
            new_pos = xp.clip(new_pos, 0, (shape_array-1))

        # Special wrap - vertical only
        elif (self.dimensions == 2) and (self.boundary_condition == 'wrap_vertical'):
            height, width = self.shape

            new_pos[new_pos[:,0] < 0, 0] += height
            new_pos[new_pos[:,0] >= height, 0] -= height

            new_pos[:,1] = xp.clip(new_pos[:,1], 0, (width-1))

        # Special wrap - horizontal only
        elif (self.dimensions == 2) and (self.boundary_condition == 'wrap_horizontal'):
            height, width = self.shape

            new_pos[new_pos[:,1] < 0, 1] += width
            new_pos[new_pos[:,1] >= width, 1] -= width

            new_pos[:,0] = xp.clip(new_pos[:,0], 0, (height-1))

        return new_pos[0] if is_single_point else new_pos


    def distance_to_source(self,
                           point: np.ndarray,
                           metric: Literal['manhattan'] = 'manhattan'
                           ) -> float | np.ndarray:
        '''
        Function to compute the distance(s) between given points and the source point.

        Parameters
        ----------
        point : np.ndarray
            A single or an Nx2 array containing N points.
        metric : 'manhattan'
            The metric to use to compute the distance.

        Returns
        -------
        dist : float or np.ndarray
            A single distance or a list of distance in a 1D distance array.
        '''
        xp = cp if self.on_gpu else np

        # Handling the case we have a single point
        is_single_point = (len(point.shape) == 1)
        if is_single_point:
            point = point[None,:]

        # Computing dist
        dist = None
        if metric == 'manhattan':
            dist = xp.sum(xp.abs(self.source_position[None,:] - point), axis=-1) - self.source_radius

        if dist is None: # Meaning it was not computed
            raise NotImplementedError('This distance metric has not yet been implemented')

        return float(dist[0]) if is_single_point else dist


    def save(self,
             folder: str | None = None,
             save_arrays: bool = False,
             force: bool = False
             ) -> None:
        '''
        Function to save the environment to the memory.

        By default it saved in a new folder at the current path in a new folder with the name 'Env-<name>' where <name> is the name set when initializing an environment.
        In this folder a file "METADATA.json" is created containing all the properties of the environment.

        The numpy arrays of the environment (grid and start_probabilities) can be saved or not. If not, when the environment is loaded it needs to be reconstructed from the original data file.
        The arrays are saved to .npy files along with the METADATA file.

        If an environment of the same name is already saved, the saving will be interupted. It can however be forced with the force parameter.

        Parameters
        ----------
        folder : str, optional
            The folder to which to save the environment data. If it is not provided, it will be created in the current folder.
        save_arrays : bool, default=False
            Whether or not to save the numpy arrays to memory. (The arrays can be heavy)
        force : bool, default=False
            In case an environment of the same name is already saved, it will be overwritten.
        '''
        # If on gpu, use the cpu version to save
        if self.on_gpu:
            self._alternate_version.save(
                folder=folder,
                save_arrays=save_arrays,
                force=force
            )
            return # Blank return

        # Assert either data_file is provided or save_arrays is enabled
        assert save_arrays or ((self.data_file_path is not None) and (self.start_type is not None)), "The environment was not created from a data file so 'save_arrays' has to be set to True."

        # Adding env name to folder path
        if folder is None:
            folder = f'./Env-{self.name}'
        else:
            folder += '/Env-' + self.name

        # Checking the folder exists or creates it
        if not os.path.exists(folder):
            os.mkdir(folder)
        elif len(os.listdir(folder)) > 0:
            if force:
                shutil.rmtree(folder)
                os.mkdir(folder)
            else:
                raise Exception(f'{folder} is not empty. If you want to overwrite the saved model, enable "force".')

        # Generating the metadata arguments dictionary
        arguments = {}
        arguments['name'] = self.name

        if self.data_file_path is not None:
            arguments['data_file_path'] = self.data_file_path

        arguments['timesteps']                     = int(self.timesteps)
        arguments['data_shape']                    = self.data_shape
        arguments['dimensions']                    = self.dimensions
        arguments['margins']                       = self.margins.tolist()
        arguments['shape']                         = self.shape
        arguments['data_bounds']                   = self.data_bounds.tolist()
        arguments['original_data_source_position'] = self.original_data_source_position.tolist()
        arguments['data_source_position']          = self.data_source_position.tolist()
        arguments['layers']                        = (self.layer_labels if self.has_layers else False)
        arguments['source_position']               = self.source_position.tolist()
        arguments['source_radius']                 = self.source_radius
        arguments['interpolation_method']          = self.interpolation_method
        arguments['preprocess_data']               = self._preprocess_data
        arguments['data_processed']                = self.data_processed
        arguments['boundary_condition']            = self.boundary_condition
        arguments['start_type']                    = self.start_type
        arguments['seed']                          = self.seed

        # Check how the start probabilities were built
        if self.start_type.startswith('custom') and len(self.start_type.split('_')) == 1 and not save_arrays:
            raise Exception('Start probabilities have been set from a custom array, please enable save_arrays to be able to reconstruct the environment later.')

        if self.odor_present_threshold is not None:
            arguments['odor_present_threshold'] = self.odor_present_threshold

        # Output the arguments to a METADATA file
        with open(folder + '/METADATA.json', 'w') as json_file:
            json.dump(arguments, json_file, indent=4)

        # Output the numpy arrays
        if save_arrays:
            if isinstance(self._data, np.ndarray):
                np.save(folder + '/data.npy', self._data)
            else:
                raise NotImplementedError('The saving of data that is not a Numpy array was not implemented yet.')
            np.save(folder + '/start_probabilities.npy', self.start_probabilities)

        # Success print
        self.saved_at = os.path.abspath(folder).replace('\\', '/')
        print(f'Environment saved to: {folder}')


    @classmethod
    def load(cls,
             folder: str
             ) -> 'Environment':
        '''
        Function to load an environment from a given folder.

        Parameters
        ----------
        folder : str
            The folder of the Environment.

        Returns
        -------
        loaded_env : Environment
            The loaded environment.
        '''
        assert os.path.exists(folder), "Folder doesn't exist..."
        assert folder.split('/')[-1].startswith('Env-'), "The folder provided is not the data of en Environment object."

        # Load arguments
        arguments: dict = None
        with open(folder + '/METADATA.json', 'r') as json_file:
            arguments = json.load(json_file)

        # Check if numpy arrays are provided, if not, recreate a new environment model
        if os.path.exists(folder + '/data.npy') and os.path.exists(folder + '/start_probabilities.npy'):
            data = np.load(folder + '/data.npy')
            start_probabilities = np.load(folder + '/start_probabilities.npy')

            loaded_env = cls.__new__(cls)

            # Set the arguments
            loaded_env.name                          = arguments['name']
            loaded_env.timesteps                     = arguments['timesteps']
            loaded_env.data_shape                    = arguments['data_shape']
            loaded_env.dimensions                    = arguments['dimensions']
            loaded_env.margins                       = np.array(arguments['margins'])
            loaded_env.shape                         = arguments['shape']
            loaded_env.data_bounds                   = np.array(arguments['data_bounds'])
            loaded_env.original_data_source_position = np.array(arguments['original_data_source_position'])
            loaded_env.data_source_position          = np.array(arguments['data_source_position'])
            loaded_env.source_position               = np.array(arguments['source_position'])
            loaded_env.source_radius                 = arguments['source_radius']
            loaded_env.has_layers                    = isinstance(arguments['layers'], list)
            loaded_env.layers                        = np.arange(len(arguments['layers'])) if loaded_env.has_layers else None
            loaded_env.layer_labels                  = arguments['layers']
            loaded_env.interpolation_method          = arguments['interpolation_method']
            loaded_env._preprocess_data              = arguments['preprocess_data']
            loaded_env.data_processed                = arguments['data_processed']
            loaded_env.boundary_condition            = arguments['boundary_condition']
            loaded_env.on_gpu                        = False
            loaded_env.seed                          = arguments['seed']
            loaded_env.rnd_state                     = np.random.RandomState(arguments['seed'])

            # Optional arguments
            loaded_env.data_file_path                = arguments.get('data_file_path')
            loaded_env.odor_present_threshold        = arguments.get('odor_present_threshold')
            loaded_env.start_type                    = arguments.get('start_type')

            # Arrays
            loaded_env._data = data
            loaded_env.start_probabilities = start_probabilities

        else:
            start_zone: str = arguments['start_type']
            start_zone_boundaries = None
            if start_zone.startswith('custom'):
                start_zone_boundaries = np.array(start_zone.split('_')[1:]).reshape((arguments['dimensions'],2)).astype(int)

            loaded_env = Environment(
                data_file              = arguments['data_file_path'],
                data_source_position   = arguments['original_data_source_position'],
                source_radius          = arguments['source_radius'],
                layers                 = arguments['layers'],
                shape                  = arguments['shape'],
                margins                = arguments['margins'],
                interpolation_method   = arguments['interpolation_method'],
                preprocess_data        = arguments['preprocess_data'],
                boundary_condition     = arguments['boundary_condition'],
                start_zone             = (start_zone_boundaries if start_zone_boundaries is not None else start_zone),
                odor_present_threshold = arguments.get('odor_present_threshold'),
                name                   = arguments['name'],
                seed                   = arguments['seed']
            )

        # Folder where the environment was pulled from
        loaded_env.saved_at = os.path.abspath(folder)

        return loaded_env


    def to_gpu(self) -> 'Environment':
        '''
        Function to send the numpy arrays of the environment to the gpu memory.
        It returns a new instance of the Environment with the arrays as cupy arrays.

        Returns
        -------
        gpu_environment : Environment
            A new environment instance where the arrays are on the gpu memory.
        '''
        # Check whether the environment is already on the gpu or not
        if self.on_gpu:
            return self

        # Warn and overwrite alternate_version in case it already exists
        if self._alternate_version is not None:
            print('[warning] A GPU instance already existed and is being recreated.')
            self._alternate_version = None

        assert gpu_support, "GPU support is not enabled..."

        # Generating a new instance
        cls = self.__class__
        gpu_environment = cls.__new__(cls)

        # Copying arguments to gpu
        for arg, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                setattr(gpu_environment, arg, cp.array(val))
            elif arg == 'rnd_state':
                setattr(gpu_environment, arg, cp.random.RandomState(self.seed))
            else:
                setattr(gpu_environment, arg, val)

        # Self reference instances
        self._alternate_version = gpu_environment
        gpu_environment._alternate_version = self

        gpu_environment.on_gpu = True
        return gpu_environment


    def to_cpu(self) -> 'Environment':
        '''
        Function to send the numpy arrays of the environment to the cpu memory.
        It returns a new instance of the Environment with the arrays as numpy arrays.

        Returns
        -------
        cpu_environment : Environment
            A new environment instance where the arrays are on the cpu memory.
        '''
        # Check whether the agent is already on the cpu or not
        if not self.on_gpu:
            return self

        if self._alternate_version is not None:
            print('[warning] A CPU instance already existed and is being recreated.')
            self._alternate_version = None

        # Generating a new instance
        cls = self.__class__
        cpu_environment = cls.__new__(cls)

        # Copying arguments to gpu
        for arg, val in self.__dict__.items():
            if isinstance(val, cp.ndarray):
                setattr(cpu_environment, arg, cp.asnumpy(val))
            elif arg == 'rnd_state':
                setattr(cpu_environment, arg, np.random.RandomState(self.seed))
            else:
                setattr(cpu_environment, arg, val)

        # Self reference instances
        self._alternate_version = cpu_environment
        cpu_environment._alternate_version = self

        cpu_environment.on_gpu = True
        return cpu_environment


    @property
    def gpu_version(self) -> 'Environment':
        '''
        A version of the Environment on the GPU.
        If the environment is already on the GPU it returns itself, otherwise the to_gpu function is called to generate a new one.
        '''
        if self.on_gpu:
            return self
        else:
            if self._alternate_version is not None: # Check if an alternate version already exists
                return self._alternate_version
            else: # Generate an alternate version on the gpu
                return self.to_gpu()


    @property
    def cpu_version(self) -> 'Environment':
        '''
        A version of the Environment on the CPU.
        If the environment is already on the CPU it returns itself, otherwise the to_cpu function is called to generate a new one.
        '''
        if not self.on_gpu:
            return self
        else:
            if self._alternate_version is not None: # Check if an alternate version already exists
                return self._alternate_version
            else: # Generate an alternate version on the cpu
                return self.to_cpu()


    def modify(self,
               data_source_position: list | np.ndarray | None = None,
               source_radius: float | None = None,
               shape: list | np.ndarray | None = None,
               margins: int | list | np.ndarray | None = None,
               multiplier: list | np.ndarray | None = None,
               interpolation_method: str | None = None,
               boundary_condition: str | None = None
               ) -> 'Environment':
        '''
        Returns a copy of the environment with one or more parameters modified.

        Parameters
        ----------
        data_source_position: list or np.ndarray, optional
            A new position for the source relative to the data file.
        source_radius: float, optional
            A new source radius.
        shape: list or np.ndarray, optional
            A new shape of environment.
        margins: int or list or np.ndarray, optional
            A new set of margins.
        multiplier: list or np.ndarray, optional
            A new multiplier to be applied to the data file (this will in turn increase or reduce the margins).
        interpolation_method: str, optional
            A new interpolation method to be used.
        boundary_condition: str, optional
            New boundary conditions for how the agent should behave at the edges.

        Returns
        -------
        modified_environment
            A copy of the environment where the modified parameters have been applied.
        '''
        if self.on_gpu:
            cpu_environment = self.cpu_version
            new_cpu_environment = cpu_environment.modify(
                data_source_position = data_source_position,
                source_radius        = source_radius,
                shape                = shape,
                margins              = margins,
                multiplier           = multiplier,
                interpolation_method = interpolation_method,
                boundary_condition   = boundary_condition
            )
            return new_cpu_environment.to_gpu()

        modified_environment = Environment(
            data_file              = (self.data_file_path if (self.data_file_path is not None) else self._data),
            data_source_position   = (data_source_position if (data_source_position is not None) else self.original_data_source_position),
            source_radius          = (source_radius if (source_radius is not None) else self.source_radius),
            layers                 = (self.layer_labels if self.has_layers else False),
            shape                  = (shape if (shape is not None) else self.shape),
            margins                = (margins if (margins is not None) else self.margins),
            multiplier             = (multiplier if (multiplier is not None) else [1.0,1.0]),
            interpolation_method   = (interpolation_method if (interpolation_method is not None) else self.interpolation_method),
            preprocess_data        = self._preprocess_data,
            boundary_condition     = (boundary_condition if (boundary_condition is not None) else self.boundary_condition),
            start_zone             = self.start_type,
            odor_present_threshold = self.odor_present_threshold,
            name                   = self.name,
            seed                   = self.seed
        )
        return modified_environment


    def modify_scale(self,
                     scale_factor: float
                     ) -> 'Environment':
        '''
        Function to modify the size of the environment by a scale factor.
        Everything will be scaled this factor. This includes: shape, margins, source radius, and data shape.

        Parameters
        ----------
        scale_factor : float
            By how much to modify the size of the current environment.

        Returns
        -------
        modified_environment : Environment
            The environment with the scale factor applied.
        '''
        modified_source_radius = self.source_radius * scale_factor
        modified_shape = (np.array(self.shape) * scale_factor).astype(int)
        modified_margins = (self.margins * scale_factor).astype(int)

        modified_environment = Environment(
            data_file              = (self.data_file_path if (self.data_file_path is not None) else self._data),
            data_source_position   = self.original_data_source_position,
            source_radius          = modified_source_radius,
            layers                 = (self.layer_labels if self.has_layers else False),
            shape                  = modified_shape,
            margins                = modified_margins,
            multiplier             = [1.0,1.0],
            interpolation_method   = self.interpolation_method,
            preprocess_data        = self._preprocess_data,
            boundary_condition     = self.boundary_condition,
            start_zone             = self.start_type,
            odor_present_threshold = self.odor_present_threshold,
            name                   = self.name,
            seed                   = self.seed
        )
        return modified_environment