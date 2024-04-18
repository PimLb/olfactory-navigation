import json
import os
import shutil

from matplotlib import pyplot as plt
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

    -------------------------
    |                       |
    |   -----------------   |
    |   |               |   |
    |   -----------------   |
    |                       |
    -------------------------

    # TODO: Add support for a 'real' grid, eg in meters... 

    margins can be provided as:
    - An equal margin on each side
    - A array of 2 elements for x and y margins
    - A 2D array for each element being [axis, side] where axis is [vertical, horizontal] and side is [L,R]

    ...
    # TODO Write these
    Parameters
    ----------

    Arguments
    ---------

    '''
    def __init__(self,
                 data:str|np.ndarray,
                 source_position:list|np.ndarray,
                 source_radius:int=1,
                 discretization:int=1,
                 margins:int|list|np.ndarray=0,
                 boundary_condition:Literal['stop', 'wrap', 'wrap_vertical', 'wrap_horizontal', 'clip' ,'no']='stop',
                 start_zone:Literal['odor_present','data_zone']|np.ndarray='data_zone',
                 odor_present_treshold:float|None=None,
                 name:str|None=None,
                 seed : int = 12131415,
                 ) -> None:
        self.saved_at = None

        # Load from file if string provided
        self.source_data_file = data if isinstance(data, str) else None
        if isinstance(data, str):
            data_file = data
            if data_file.endswith('.npy'):
                data = np.load(data_file)
            else:
                raise NotImplementedError('File format loading not implemented')

        # Making margins a 2x2 array 
        if isinstance(margins, int):
            self.margins = np.ones((2,2)) * margins
        elif isinstance(margins, list) or (margins.shape == (2,)):
            assert len(margins) == 2, 'Margins, if provided as a list must contain only two elements.'
            margins = np.array(margins)
            self.margins = np.hstack((margins[:,None], margins[:,None]))
        elif margins.shape == (2,2):
            self.margins = margins
        else:
            raise ValueError('margins argument should be either an integer or a 1D or 2D array with either shape (2) or (2,2)')
        assert self.margins.dtype == int, 'margins should be integers'

        # Reading shape of data array
        timesteps, self.height, self.width = data.shape
        self.padded_height = self.height + np.sum(self.margins[0])
        self.padded_width = self.width + np.sum(self.margins[1])
        self.shape = (self.padded_height, self.padded_width)

        # Preprocess data with discretization
        self.discretization = discretization
        if discretization != 1:
            raise NotImplementedError('Different discretizations have not been implemented yet') # TODO
        self.grid : np.ndarray = data

        # Apply margins to grid
        self.grid = np.hstack([np.zeros((timesteps, self.margins[0,0], self.width)), self.grid, np.zeros((timesteps, self.margins[0,1], self.width))])
        self.grid = np.dstack([np.zeros((timesteps, self.padded_height, self.margins[1,0])), self.grid, np.zeros((timesteps, self.padded_height, self.margins[1,1]))])

        # Saving arguments
        self.data_source_position = np.array(source_position)
        self.source_position = self.data_source_position + self.margins[:,0]
        self.source_radius = source_radius
        self.boundary_condition = boundary_condition

        # Starting zone
        self.start_probabilities = np.zeros(self.shape)
        if start_zone == 'data_zone':
            self.start_probabilities[self.margins[0,0]:self.margins[0,0]+self.height, self.margins[1,0]:self.margins[1,0]+self.width] = 1.0
        elif start_zone == 'odor_present':
            self.start_probabilities = (np.mean((self.grid > (odor_present_treshold if odor_present_treshold is not None else 0)).astype(int), axis=0) > 0).astype(float)
        elif isinstance(start_zone, np.ndarray):
            if start_zone.shape == (2,2):
                self.start_probabilities[start_zone[0,0]:start_zone[0,1], start_zone[1,0]:start_zone[1,1]] = 1.0
            elif start_zone.shape == self.shape:
                self.start_probabilities = start_zone
            else:
                raise ValueError('If an np.ndarray is provided for the start_zone it has to be 2x2...')
        else:
            raise ValueError('start_zone value is wrong')
        
        self.start_type = start_zone

        # Odor present tresh
        self.odor_present_treshold = odor_present_treshold

        # Removing the source area from the starting zone
        source_mask = np.fromfunction(lambda x,y: ((x - self.source_position[0])**2 + (y - self.source_position[1])**2) <= self.source_radius**2, shape=self.shape)
        self.start_probabilities[source_mask] = 0

        self.start_probabilities /= np.sum(self.start_probabilities)

        # Name
        self.name = name
        if self.name is None:
            self.name =  f'{self.padded_height}_{self.padded_width}' # Size of env
            self.name += f'-edge_{self.boundary_condition}' # Boundary condition
            self.name += f'-start_{self.start_type if self.start_type is not None else "custom"}' # Start zone
            self.name += f'-source_{self.source_position[0]}_{self.source_position[1]}_radius{self.source_radius}' # Source

        # gpu support
        self._alternate_version = None
        self.on_gpu = False

        # random state
        xp = cp if self.on_gpu else np
        self.rnd_state = xp.random.RandomState(seed = seed)


    def plot(self, frame:int=0, ax=None) -> None:
        '''
        Simple function to plot the environment

        Parameters
        ----------
        ax : Optional
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
        ax.imshow((self.grid[frame] > (self.odor_present_treshold if self.odor_present_treshold is not None else 0)).astype(float), cmap='Greys')

        # Start zone contour
        start_zone = plt.Rectangle([0,0], 1, 1, color='blue', fill=False)
        ax.contour(self.start_probabilities, levels=[0.0], colors='blue')

        # Source circle
        goal_circle = plt.Circle(self.source_position[::-1], self.source_radius, color='r', fill=False)
        ax.add_patch(goal_circle)

        # Legend
        ax.legend([odor, start_zone, goal_circle], [f'Frame {frame} odor cues', 'Start zone', 'Source'])


    def get_observation(self,
                        pos:np.ndarray,
                        time:int|np.ndarray=0
                        ) -> float|np.ndarray:
        '''
        Function to get an observation at a given position on the grid at a given time.
        A set of observations can also be requested, either at a single position for multiple timestamps or with the same amoung of positions as timestamps provided.
        
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
        # Handling the case of a single point
        is_single_point = (len(pos.shape) == 1)
        if is_single_point:
            pos = pos[None,:]
        if self.boundary_condition is None or self.boundary_condition == 'no':
            if is_single_point:
                return float(self.grid[time, pos[0], pos[1]] ) if  0 <= pos[0] < self.grid.shape[1] and 0 <= pos[1] < self.grid.shape[2] else 0.0
            #print(pos[:, 0], pos[:, 1], self.grid.shape)
            mask = (0 <= pos[:, 0]) & (pos[:, 0] < self.grid.shape[1]) & (0 <= pos[:, 1]) & (pos[:, 1] < self.grid.shape[2])
 #           print(mask.shape, pos.shape)
            observation = np.zeros((mask.shape[0], ))
#            print(observation[mask].shape, pos[mask,0].shape, time)#, self.grid[time, pos[mask,0], pos[mask,1]].shape)
            if isinstance(time, int):
                observation[mask] = self.grid[time, pos[mask,0], pos[mask,1]]
            else:
                observation[mask] = self.grid[time[mask], pos[mask,0], pos[mask,1]]
            return observation
        observation = self.grid[time, pos[0], pos[1]] if len(pos.shape) == 1 else self.grid[time, pos[:,0], pos[:,1]]

        return float(observation[0]) if is_single_point else observation


    def source_reached(self,
                       pos:np.ndarray
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
                            n:int=1
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
             pos:np.ndarray,
             movement:np.ndarray
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
                           point:np.ndarray,
                           metric:Literal['manhattan']='manhattan'
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
             folder:str|None=None,
             save_arrays:bool=False,
             force:bool=False
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
        folder : str (optional)
            The folder to which to save the environment data. If it is not provided, it will be created in the current folder.
        save_arrays : bool (default = False)
            Whether or not to save the numpy arrays to memory. (The arrays can be heavy)
        force : bool (default = False)
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
        assert save_arrays or ((self.source_data_file is not None) and (self.start_type is not None)), "The environment was not created from a data file so 'save_arrays' has to be set to True."

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

        if self.source_data_file is not None:
            arguments['source_data_file'] = self.source_data_file

        arguments['width']                 = self.width
        arguments['height']                = self.height
        arguments['margins']               = self.margins.tolist()
        arguments['padded_width']          = int(self.padded_width)
        arguments['padded_height']         = int(self.padded_height)
        arguments['shape']                 = [int(s) for s in self.shape]
        arguments['discretization']        = self.discretization
        arguments['data_source_position']  = self.data_source_position.tolist()
        arguments['source_position']       = self.source_position.tolist()
        arguments['source_radius']         = self.source_radius
        arguments['boundary_condition']    = self.boundary_condition

        if self.odor_present_treshold is not None:
            arguments['odor_present_treshold'] = self.odor_present_treshold
        if self.start_type is not None:
            arguments['start_type'] = self.start_type

        # Output the arguments to a METADATA file
        with open(folder + '/METADATA.json', 'w') as json_file:
            json.dump(arguments, json_file, indent=4)

        # Output the numpy arrays
        if save_arrays:
            np.save(folder + '/grid.npy', self.grid)
            np.save(folder + '/start_probabilities.npy', self.start_probabilities)

        # Success print
        self.saved_at = os.path.abspath(folder).replace('\\', '/')
        print(f'Environment saved to: {folder}')


    @classmethod
    def load(cls,
             folder:str
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
        if os.path.exists(folder + '/grid.npy') and os.path.exists(folder + '/start_probabilities.npy'):
            grid = np.load(folder + '/grid.npy')
            start_probabilities = np.load(folder + '/start_probabilities.npy')

            loaded_env = cls.__new__(cls)

            # Set the arguments
            loaded_env.width                 = arguments['width']
            loaded_env.height                = arguments['height']
            loaded_env.margins               = np.array(arguments['margins'])
            loaded_env.padded_width          = arguments['padded_width']
            loaded_env.padded_height         = arguments['padded_height']
            loaded_env.shape                 = set(arguments['shape'])
            loaded_env.discretization        = arguments['discretization']
            loaded_env.data_source_position  = np.array(arguments['data_source_position'])
            loaded_env.source_position       = np.array(arguments['source_position'])
            loaded_env.source_radius         = arguments['source_radius']
            loaded_env.boundary_condition    = arguments['boundary_condition']

            # Optional arguments
            loaded_env.source_data_file      = arguments.get('source_data_file')
            loaded_env.odor_present_treshold = arguments.get('odor_present_treshold')
            loaded_env.start_type            = arguments.get('start_type')

            # Arrays
            loaded_env.grid = grid
            loaded_env.start_probabilities = start_probabilities

        else:
            loaded_env = Environment(
                data                  = arguments['source_data_file'],
                source_position       = np.array(arguments['data_source_position']),
                source_radius         = arguments['source_radius'],
                discretization        = arguments['discretization'],
                margins               = np.array(arguments['margins']),
                boundary_condition    = arguments['boundary_condition'],
                start_zone            = arguments.get('start_type'),
                odor_present_treshold = arguments.get('odor_present_treshold'),
                name                  = arguments['name']
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