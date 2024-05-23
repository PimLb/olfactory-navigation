import cv2
import json
import os
import shutil

from matplotlib import pyplot as plt
from scipy import interpolate
from typing import Literal

import numpy as np
gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')


class Environment:
    '''
    Class to represent an olfactory environment.

    It is defined based on an olfactory data set.

    margins can be provided as:
    
    - An equal margin on each side
    - A array of 2 elements for x and y margins
    - A 2D array for each element being [axis, side] where axis is [vertical, horizontal] and side is [L,R]

    Parameters
    ----------
    data_file : str or np.ndarray
        The dataset containing the olfactory data. It can be provided as a path to a file containing said array.
    source_position : list or np.ndarray
        The center point of the source provided as a list or a 1D array with the components being x,y.
        This position is computed in the olfactory data zone (so excluding the margins).
    source_radius : int, default=1
        The radius from the center point of the source in which we consider the agent has reached the source.
    discretization : list or np.ndarray, optional
        A 2-element array or list of how many units should be kept in the final array (including the margins).
        As it should include the margins, the discretization amounts should be strictly larger than the sum of the margins in each direction.
        By default, the shape of the olfactory data will be maintained.
    multiplier : int or list or np.ndarray, optional
        A single multiplier or a 2-element array or list of how much the odor field should be streched in each direction.
        If a value larger than 1 is provided, the margins will be reduced to accomodate for the larger size of the olfactory data size.
        And inversly, less than 1 will increase the margins.
        By default, the multipliers will be set to 1.0.
    margins : int or list or np.ndarray, default=0
        How many discretized units have to be added to the data as margins. (Before the multiplier is applied)
        If a unique element is provided, the margin will be this same value on each side.
        If a list or array of 2 elements is provided, the first number will be vertical margins (y-axis), while the other will be on the x-axis (horizontal).
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
        <height>_<width>-edge_<boundary_condition>-start_<start_zone>-source_<source_y>_<source_x>_radius<source_radius>
    seed : int, default=12131415
        For reproducible randomness.

    Attributes
    ----------
    data : np.ndarray
        An array containing the olfactory data.
    data_file_path : str
        If the data is loaded from a path, the path will be recorded here.
    margins : np.ndarray
        An array of the margins vertically and horizontally.
    height : int
        The height of the data's odor field.
    width : int
        The width of the data's odor field.
    padded_height : int
        The height of the environment padded with the vertical margins.
    padded_width : int
        The width of the environment passed with the horizontal margins.
    shape : tuple[int, int]
        The shape of the environment. It is a tuple of <padded_height, padded_width>.
    data_bounds : np.ndarray
        The bounds between which the original olfactory data stands in the coordinate system of the environment.
    discretization : int
        The discretization of the source data. If set to 2, the source data will be sampled every two units. (NOT IMPLEMENTED)
    data_source_position : np.ndarray
        The position of the source in the original data file.
    source_position : np.ndarray
        The position of the source in the padded grid.
    source_radius : int
        The radius of the source.
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
    on_gpu : bool
        Whether the environment's arrays are on the gpu's memory or not.
    seed : int
        The seed used for the random operations (to allow for reproducability).
    rnd_state : np.random.RandomState
        The random state variable used to generate random values.
    '''
    def __init__(self,
                 data_file: str | np.ndarray,
                 data_source_position: list | np.ndarray,
                 source_radius: int = 1,
                 discretization: np.ndarray | None = None,
                 multiplier: np.ndarray | None = None,
                 interpolation_method: Literal['Nearest', 'Linear', 'Cubic'] = 'Linear',
                 margins: int | list | np.ndarray = 0,
                 boundary_condition: Literal['stop', 'wrap', 'wrap_vertical', 'wrap_horizontal', 'clip', 'no'] = 'stop',
                 start_zone: Literal['odor_present', 'data_zone'] | np.ndarray = 'data_zone',
                 odor_present_threshold: float | None = None,
                 name: str | None = None,
                 seed: int = 12131415,
                 ) -> None:
        self.saved_at = None

        # Load from file if string provided
        self.data_file_path = None
        loaded_data = None
        if isinstance(data_file, str):
            self.data_file_path = data_file
            if data_file.endswith('.npy'):
                loaded_data = np.load(data_file)
            else:
                raise NotImplementedError('File format loading not implemented')
        
        self.data: np.ndarray = data_file if isinstance(data_file, np.ndarray) else loaded_data

        # Making margins a 2x2 array 
        if isinstance(margins, int):
            self.margins = np.ones((2,2), dtype=int) * margins
        elif isinstance(margins, list) or (margins.shape == (2,)):
            assert len(margins) == 2, 'Margins, if provided as a list must contain only two elements.'
            margins = np.array(margins)
            self.margins = np.hstack((margins[:,None], margins[:,None]))
        elif margins.shape == (2,2):
            self.margins = margins
        else:
            raise ValueError('margins argument should be either an integer or a 1D or 2D array with either shape (2) or (2,2)')
        assert self.margins.dtype == int, 'margins should be integers'

        # Unmodified sizes
        data_shape = self.data.shape[1:]
        timesteps, self.height, self.width = self.data.shape
        self.data_source_position = np.array(data_source_position)

        # Process discretization parameter
        new_data_shape = None
        if discretization is not None:
            assert np.all(discretization > np.sum(self.margins, axis=1)), "The discretization must be strictly larger than the sum of margins."

            # Computing the new shape of the data
            new_data_shape = discretization - np.sum(self.margins, axis=1)
            
            # New source position
            self.data_source_position = (self.data_source_position * (new_data_shape / data_shape)).astype(int)
        else:
            discretization = data_shape + np.sum(self.margins, axis=1)

        self.discretization = discretization

        # Process multiplier
        if multiplier is not None:
            if new_data_shape is None:
                new_data_shape = data_shape
            new_data_shape = (new_data_shape * multiplier).astype(int)

            assert np.all(new_data_shape < self.discretization), f"Multiplier goes out of bounds (Maximum allowed: {self.discretization / data_shape})"

            # New source position
            new_source_position = (self.data_source_position * multiplier).astype(int)

            # Recomputing margins
            self.margins[:,0] -= (new_source_position - self.data_source_position)
            self.margins[:,1] = (self.discretization - (self.margins[:,0] + new_data_shape))

            # Setting new source position
            self.data_source_position = new_source_position

        # Reshape data is a new_shape if set by custom discretization or multiplier
        if new_data_shape is not None:
            # Interpolation of new data
            interpolation_options = {
                'Nearest': cv2.INTER_NEAREST,
                'Linear': cv2.INTER_LINEAR,
                'Cubic': cv2.INTER_CUBIC
            }
            interpolation_choice = interpolation_options[interpolation_method]

            new_data = np.zeros((timesteps, *new_data_shape))
            for i in range(timesteps):
                new_data[i] = cv2.resize(self.data[i], dsize=new_data_shape[::-1], interpolation=interpolation_choice)

            self.data = new_data
            self.height, self.width = new_data_shape

        # Reading shape of data array
        self.padded_height:int = self.height + np.sum(self.margins[0])
        self.padded_width:int = self.width + np.sum(self.margins[1])
        self.shape:tuple[int, int] = (self.padded_height, self.padded_width)
        
        # Building a data bounds
        self.data_bounds = np.array([[self.margins[0,0], self.margins[0,0]+self.height], [self.margins[1,0], self.margins[1,0]+self.width]])

        # Saving arguments
        self.source_position = self.data_source_position + self.margins[:,0]
        self.source_radius = source_radius
        self.boundary_condition = boundary_condition

        # Starting zone
        self.start_probabilities = np.zeros(self.shape)
        self.start_type = start_zone if isinstance(start_zone, str) else 'custom'

        if isinstance(start_zone, np.ndarray):
            if start_zone.shape == (2,2):
                self.start_probabilities[start_zone[0,0]:start_zone[0,1], start_zone[1,0]:start_zone[1,1]] = 1.0
                self.start_type += '_' + '_'.join([str(el) for el in start_zone.ravel()])
            elif start_zone.shape == self.shape:
                self.start_probabilities = start_zone
            else:
                raise ValueError('If an np.ndarray is provided for the start_zone it has to be 2x2...')
        elif start_zone == 'data_zone':
            self.start_probabilities[self.data_bounds[0,0]:self.data_bounds[0,1], self.data_bounds[1,0]:self.data_bounds[1,1]] = 1.0
        elif start_zone == 'odor_present':
            odor_present_map = (np.mean((self.data > (odor_present_threshold if odor_present_threshold is not None else 0)).astype(int), axis=0) > 0).astype(float)
            self.start_probabilities[self.data_bounds[0,0]:self.data_bounds[0,1], self.data_bounds[1,0]:self.data_bounds[1,1]] = odor_present_map
        else:
            raise ValueError('start_zone value is wrong')

        # Odor present tresh
        self.odor_present_threshold = odor_present_threshold

        # Removing the source area from the starting zone
        source_mask = np.fromfunction(lambda x,y: ((x - self.source_position[0])**2 + (y - self.source_position[1])**2) <= self.source_radius**2, shape=self.shape)
        self.start_probabilities[source_mask] = 0

        self.start_probabilities /= np.sum(self.start_probabilities)

        # Name
        self.name = name
        if self.name is None:
            self.name =  f'{self.padded_height}_{self.padded_width}' # Size of env
            self.name += f'-edge_{self.boundary_condition}' # Boundary condition
            self.name += f'-start_{self.start_type}' # Start zone
            self.name += f'-source_{self.source_position[0]}_{self.source_position[1]}_radius{self.source_radius}' # Source

        # gpu support
        self._alternate_version = None
        self.on_gpu = False

        # random state
        self.seed = seed
        self.rnd_state = np.random.RandomState(seed = seed)


    def plot(self,
             frame: int = 0,
             ax: plt.Axes = None
             ) -> None:
        '''
        Simple function to plot the environment

        Parameters
        ----------
        ax : plt.Axes, optional
            An ax on which the environment can be plot
        '''
        # If on GPU use the CPU version to plot
        if self.on_gpu:
            self._alternate_version.plot(
                frame=frame,
                ax=ax
            )
            return

        if ax is None:
            _, ax = plt.subplots(1, figsize=(15,5))

        # Odor grid
        odor = plt.Rectangle([0,0], 1, 1, color='black', fill=True)
        frame_data = (self.data[frame] > (self.odor_present_threshold if self.odor_present_threshold is not None else 0)).astype(float)
        environment_frame = np.zeros(self.shape, dtype=float)
        environment_frame[self.data_bounds[0,0]:self.data_bounds[0,1], self.data_bounds[1,0]:self.data_bounds[1,1]] = frame_data
        ax.imshow(environment_frame, cmap='Greys')

        # Start zone contour
        start_zone = plt.Rectangle([0,0], 1, 1, color='blue', fill=False)
        ax.contour(self.start_probabilities, levels=[0.0], colors='blue')

        # Source circle
        goal_circle = plt.Circle(self.source_position[::-1], self.source_radius, color='r', fill=False)
        ax.add_patch(goal_circle)

        # Legend
        ax.legend([odor, start_zone, goal_circle], [f'Frame {frame} odor cues', 'Start zone', 'Source'])


    def get_observation(self,
                        pos: np.ndarray,
                        time: int | np.ndarray = 0
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

        # Time looping
        time = time % len(self.data)

        # Return 0.0 if outside of data zone
        data_pos = pos - self.margins[:,0][None,:]
        data_pos_valid = xp.all((data_pos >= 0) & (data_pos < self.data.shape[1:]), axis=1)
        observation = xp.zeros(data_pos.shape[0])
        observation[data_pos_valid] = self.data[time, data_pos[data_pos_valid,0], data_pos[data_pos_valid,1]]
#        observation = xp.where(data_pos_valid, self.data[time, data_pos[data_pos_valid,0], data_pos[data_pos_valid,1]], 0.0)

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

        is_at_source = (xp.sum((pos - self.source_position) ** 2, axis=1) <= (self.source_radius ** 2))

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

        assert n>0, "n has to be a strictly positive number (>0)"

        random_states = self.rnd_state.choice(xp.arange(self.padded_height * self.padded_width), size=n, replace=True, p=self.start_probabilities.ravel())
        random_states_2d = xp.array(xp.unravel_index(random_states, (self.padded_height, self.padded_width))).T
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

        # Wrap condition for vertical axis
        if self.boundary_condition in ['wrap', 'wrap_vertical']:
            new_pos[new_pos[:,0] < 0, 0] += self.padded_height
            new_pos[new_pos[:,0] >= self.padded_height, 0] -= self.padded_height

        # Wrap condition for horizontal axis
        if self.boundary_condition in ['wrap', 'wrap_horizontal']:
            new_pos[new_pos[:,1] < 0, 1] += self.padded_width
            new_pos[new_pos[:,1] >= self.padded_width, 1] -= self.padded_width

        # Stop condition
        if (self.boundary_condition == 'stop') or (self.boundary_condition == 'wrap_horizontal'):
            new_pos[:,0] = xp.clip(new_pos[:,0], 0, (self.padded_height-1))

        if (self.boundary_condition == 'stop') or (self.boundary_condition == 'wrap_vertical'):
            new_pos[:,1] = xp.clip(new_pos[:,1], 0, (self.padded_width-1))

        if is_single_point:
            new_pos = new_pos[0]

        return new_pos


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
            dist = xp.sum(xp.abs(self.source_position[None,:] - point), axis=1) - self.source_radius
        else:
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
            return

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

        arguments['width']                 = self.width
        arguments['height']                = self.height
        arguments['margins']               = self.margins.tolist()
        arguments['padded_width']          = int(self.padded_width)
        arguments['padded_height']         = int(self.padded_height)
        arguments['shape']                 = [int(s) for s in self.shape]
        arguments['data_bounds']           = self.data_bounds.tolist()
        arguments['discretization']        = self.discretization
        arguments['data_source_position']  = self.data_source_position.tolist()
        arguments['source_position']       = self.source_position.tolist()
        arguments['source_radius']         = self.source_radius
        arguments['boundary_condition']    = self.boundary_condition
        arguments['start_type']            = self.start_type
        arguments['seed']                  = self.seed

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
            np.save(folder + '/data.npy', self.data)
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
        arguments = None
        with open(folder + '/METADATA.json', 'r') as json_file:
            arguments = json.load(json_file)

        # Check if numpy arrays are provided, if not, recreate a new environment model
        if os.path.exists(folder + '/data.npy') and os.path.exists(folder + '/start_probabilities.npy'):
            data = np.load(folder + '/data.npy')
            start_probabilities = np.load(folder + '/start_probabilities.npy')

            loaded_env = cls.__new__(cls)

            # Set the arguments
            loaded_env.name                   = arguments['name']
            loaded_env.width                  = arguments['width']
            loaded_env.height                 = arguments['height']
            loaded_env.margins                = np.array(arguments['margins'])
            loaded_env.padded_width           = arguments['padded_width']
            loaded_env.padded_height          = arguments['padded_height']
            loaded_env.shape                  = set(arguments['shape'])
            loaded_env.data_bounds            = np.array(arguments['data_bounds'])
            loaded_env.discretization         = arguments['discretization']
            loaded_env.data_source_position   = np.array(arguments['data_source_position'])
            loaded_env.source_position        = np.array(arguments['source_position'])
            loaded_env.source_radius          = arguments['source_radius']
            loaded_env.boundary_condition     = arguments['boundary_condition']
            loaded_env.on_gpu                 = False
            loaded_env.seed                   = arguments['seed']
            loaded_env.rnd_state              = np.random.RandomState(arguments['seed'])

            # Optional arguments
            loaded_env.data_file_path         = arguments.get('data_file_path')
            loaded_env.odor_present_threshold = arguments.get('odor_present_threshold')
            loaded_env.start_type             = arguments.get('start_type')

            # Arrays
            loaded_env.data = data
            loaded_env.start_probabilities = start_probabilities

        else:
            start_zone = arguments['start_type']
            if start_zone.startswith('custom'):
                start_zone_boundaries = np.array(arguments['start_type'].split('_')[1:]).reshape((2,2)).astype(int)
                start_zone = start_zone_boundaries

            loaded_env = Environment(
                data_file              = arguments['data_file_path'],
                data_source_position        = np.array(arguments['data_source_position']),
                source_radius          = arguments['source_radius'],
                discretization         = arguments['discretization'],
                margins                = np.array(arguments['margins']),
                boundary_condition     = arguments['boundary_condition'],
                start_zone             = start_zone,
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
        if self.on_gpu:
            assert self._alternate_version is not None, "Something went wrong"
            return self._alternate_version

        return self