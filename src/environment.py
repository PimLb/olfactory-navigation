import cv2
import numpy as np

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
                 source_radius:int|list|np.ndarray=1,
                 discretization:int=1,
                 margins:int|np.ndarray=0,
                 boundary_condition:Literal['stop', 'wrap', 'wrap_vertical', 'wrap_horizontal', 'clip']='stop'
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
        elif margins.shape == (2,):
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

        # Preprocess data with discretization
        if discretization != 1:
            raise NotImplementedError('Different discretizations have not been implemented yet')
        self.grid = data

        # Apply margins to grid
        self.grid = np.hstack([np.zeros((timesteps, self.margins[0,0], self.width)), self.grid, np.zeros((timesteps, self.margins[0,1], self.width))])
        self.grid = np.dstack([np.zeros((timesteps, self.padded_height, self.margins[1,0])), self.grid, np.zeros((timesteps, self.padded_height, self.margins[1,1]))])

        # Saving arguments
        self.source_position = source_position
        self.source_radius = source_radius
        self.boundary_condition = boundary_condition


    def get_observation(self,
                        pos:np.ndarray,
                        time:int|np.ndarray
                        ) -> float|np.ndarray:
        '''
        Function to get an observation at a given position on the grid at a given time.
        A set of observations can also be requested, either at a single position for multiple timestamps or with the same amoung of positions as timestamps provided.
        
        Parameters
        ----------
        pos : np.ndarray
            The position or list of positions to get observations at.
        time : int or np.ndarray
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

        # Wrap condition for horizontal axis
        if self.boundary_condition in ['wrap', 'wrap_horizontal']:
            if new_pos[1] < 0:
                new_pos[1] += self.padded_width
            elif new_pos[1] >= self.padded_width: 
                new_pos[1] -= self.padded_width

        # Wrap condition for vertical axis
        if self.boundary_condition in ['wrap', 'wrap_vertical']:
            if new_pos[0] < 0:
                new_pos[0] += self.padded_height
            elif new_pos[0] >= self.padded_height: 
                new_pos[0] -= self.padded_height

        # Stop condition
        if self.boundary_condition == 'stop':
            new_pos[0] = min(max(new_pos[0], 0), (self.padded_height - 1))
            new_pos[1] = min(max(new_pos[1], 0), (self.padded_width - 1))

        return new_pos