import cv2
import numpy as np
from matplotlib import pyplot as plt

from typing import Literal

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
                 boundary_condition:Literal['stop', 'wrap', 'wrap_vertical', 'wrap_horizontal', 'clip']='stop',
                 start_zone:Literal['odor_present','data_zone']|np.ndarray='data_zone'
                 ) -> None:
        
        # Load from file if string provided
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
            self.margins == margins
        else:
            raise ValueError('margins argument should be either an integer or a 1D or 2D array with either shape (2) or (2,2)')
        assert self.margins.dtype == int, 'margins should be integers'

        # Reading shape of data array
        timesteps, self.height, self.width = data.shape
        self.padded_height = self.height + np.sum(self.margins[0])
        self.padded_width = self.width + np.sum(self.margins[1])
        self.shape = (self.padded_height, self.padded_width)

        # Preprocess data with discretization
        if discretization != 1:
            raise NotImplementedError('Different discretizations have not been implemented yet')
        self.grid = data

        # Apply margins to grid
        self.grid = np.hstack([np.zeros((timesteps, self.margins[0,0], self.width)), self.grid, np.zeros((timesteps, self.margins[0,1], self.width))])
        self.grid = np.dstack([np.zeros((timesteps, self.padded_height, self.margins[1,0])), self.grid, np.zeros((timesteps, self.padded_height, self.margins[1,1]))])

        # Saving arguments
        self.source_position = np.array(source_position) + self.margins[:,0]
        self.source_radius = source_radius
        self.boundary_condition = boundary_condition

        # Starting zone
        self.start_probabilities = np.zeros(self.shape)
        if start_zone == 'data_zone':
            self.start_probabilities[self.margins[0,0]:self.margins[0,0]+self.height, self.margins[1,0]:self.margins[1,0]+self.width] = 1.0
        elif start_zone == 'odor_present':
            self.start_probabilities = (np.mean(self.grid, axis=0) > 0).astype(float)
        elif isinstance(start_zone, np.ndarray):
            if start_zone.shape == (2,2):
                self.start_probabilities[start_zone[0,0]:start_zone[0,1], start_zone[1,0]:start_zone[1,1]] = 1.0
            elif start_zone.shape == self.shape:
                self.start_probabilities = start_zone
            else:
                raise ValueError('If an np.ndarray is provided for the start_zone it has to be 2x2...')
        else:
            raise ValueError('start_zone value is wrong')
        
        source_mask = np.fromfunction(lambda x,y: ((x - self.source_position[0])**2 + (y - self.source_position[1])**2) <= self.source_radius**2, shape=self.shape)
        self.start_probabilities[source_mask] = 0

        self.start_probabilities /= np.sum(self.start_probabilities)


    def plot(self, ax=None) -> None:
        '''
        Simple function to plot the environment

        Parameters
        ----------
        ax : Optional
            An ax on which the environment can be plot
        '''
        if ax is None:
            _, ax = plt.subplots(1, figsize=(15,5))
            
        odor = plt.Rectangle([0,0], 1, 1, color='black', fill=True)
        ax.imshow(self.grid[0], cmap='Greys')

        start_zone = plt.Rectangle([0,0], 1, 1, color='blue', fill=False)
        ax.contour(self.start_probabilities, levels=[0.0], colors='blue')

        goal_circle = plt.Circle(self.source_position[::-1], self.source_radius, color='r', fill=False)
        ax.add_patch(goal_circle)

        ax.legend([odor, start_zone, goal_circle], ['Frame 0 odor cues', 'Start zone', 'Source'])


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
        return self.grid[time, pos[0], pos[1]] if len(pos.shape) == 1 else self.grid[time, pos[:,0], pos[:,1]]


    def source_reached(self,
                       pos:np.ndarray
                       ) -> bool:
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
        return np.sum((pos - self.source_position) ** 2, axis=1) <= (self.source_radius ** 2)


    def random_start_points(self, n:int=1) -> np.ndarray:
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
        assert n>0, "n has to be a positive number"
        
        random_states = np.random.choice(np.arange(self.padded_height * self.padded_width), size=n, replace=True, p=self.start_probabilities.ravel())
        random_states_2d = np.array(np.unravel_index(random_states, shape=(self.padded_height, self.padded_width))).T
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
        new_pos = pos + movement

        if len(pos.shape) == 1:
            new_pos = new_pos[None,:]

        # Wrap condition for horizontal axis
        if self.boundary_condition in ['wrap', 'wrap_horizontal']:
            new_pos[new_pos[:,0] < 0, 1] += self.padded_width
            new_pos[new_pos[:,0] >= self.padded_width, 1] -= self.padded_width

        # Wrap condition for vertical axis
        if self.boundary_condition in ['wrap', 'wrap_vertical']:
            new_pos[new_pos[:,1] < 0, 1] += self.padded_height
            new_pos[new_pos[:,1] >= self.padded_height, 1] -= self.padded_height

        # Stop condition
        if self.boundary_condition == 'stop':
            new_pos[:,0] = np.clip(new_pos[:,0], 0, self.padded_height)
            new_pos[:,1] = np.clip(new_pos[:,1], 0, self.padded_width)

        if len(pos.shape) == 1:
            new_pos = new_pos[0]

        return new_pos
    

    def distance_to_source(self,
                           point:np.ndarray,
                           metric:Literal['manhattan']='manhattan'
                           ) -> np.ndarray:
        '''
        Function to compute the distance or distances between
        '''
        if metric == 'manhattan':
            return np.sum(np.abs(self.source_position[None,:] - point), axis=1) - self.source_radius
        else:
            raise NotImplementedError('This distance metric has not yet been implemented')