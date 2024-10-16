# Imports
import sys
sys.path.append('../..')

import numpy as np
import cupy as cp
import os
import h5py

from scipy.stats import multivariate_normal

from olfactory_navigation.environment import Environment
from olfactory_navigation.agents import FSVI_Agent
from olfactory_navigation.simulation import run_test
from olfactory_navigation.test_setups import run_all_starts_test

from position_aware.model_7 import generate_model

from cupy.cuda import runtime as cuda_runtime


# Test function
def run_full_test(grid, sub_grid, cov, folder):
    # Actual data
    tank_size = np.array([111,142]) # actual size: 114cm x 89cm but it is scaled by a walking speed of 0.8 cm/s
    space_shape = tank_size
    source_radius = 2

    # State shapes
    cells = np.array([grid,grid])
    source_cell_resolution = np.array([sub_grid,sub_grid])

    # Artificial data
    data_covariance = cov

    # Data path
    data_folder = '/storage/arnaud/tank_odor_field_2024_06_13/'

    # -----------
    # Agent
    def generate_fake_data(space_shape: np.ndarray,
                           data_mean: list,
                           data_covariance,
                           samples: int
                           ) -> np.ndarray:
        y,x = np.meshgrid(np.arange(space_shape[0]), np.arange(space_shape[1]))
        pos = np.dstack((y, x))
        rv = multivariate_normal(data_mean, data_covariance)
        probability_map = rv.pdf(pos).T

        # Normalize to have 1 at the center
        probability_map /= np.max(probability_map)

        multiple_samples = (np.random.random((samples, *space_shape)) < probability_map).astype(float)

        return multiple_samples

    space_center = (space_shape/2).astype(int)

    artificial_data = generate_fake_data(space_shape=space_shape,
                                        data_mean=space_center,
                                        data_covariance=data_covariance,
                                        samples=1000)

    artificial_env  = Environment(data_file=artificial_data,
                                data_source_position=space_center,
                                source_radius=source_radius,
                                shape=space_shape,
                                boundary_condition='stop',
                                start_zone='data_zone',
                                odor_present_threshold=3e-1)

    model = generate_model(
        space_shape=space_shape,
        cells=cells,
        source_radius=source_radius,
        source_cell_resolution=source_cell_resolution,
        data_mean=space_center,
        data_covariance=data_covariance
    )

    ag = FSVI_Agent(environment=artificial_env,
                    threshold=1e-1,
                    space_aware=True,
                    spacial_subdivisions=cells,
                    model=model)

    _ = ag.train(expansions=300, use_gpu=True, print_progress=False)

    # ------------------
    # Setting up tests
    all_points = np.array(list(np.ndindex(tuple(space_shape))))
    starting_points = all_points[all_points[:,1] >= 110]

    # ---------------------
    # Testing different starting positions
    source_indices = np.array(list(np.ndindex((25,15))))
    source_indices = (source_indices * 4) + np.array([7,12])

    all_shift = source_indices - space_center
    for i, (shift, source_pos) in enumerate(zip(all_shift, source_indices)):
        print()
        print('--------------------------------------------')
        print(f'[{i+1} / {len(all_shift)}] Source position: [{source_pos[1]}, {source_pos[0]}]')
        print()

        # Setup shifted environment
        grid = np.zeros(space_shape)
        indices = np.array(list(np.ndindex(tuple(space_shape))))
        radius = 5
        in_circle = np.sum((space_center - indices)**2, axis=1) <= (radius**2)
        grid[*indices[in_circle].T] = 1.0
        uniform_artificial_data = np.array([grid for _ in range(1000)])

        shifted_artificial_data = np.zeros(uniform_artificial_data.shape)
        trimmed_artificial_data = uniform_artificial_data[:, *[slice(0, -s) if s > 0 else slice(-s, shape) for shape, s in zip(space_shape, shift)]]
        shifted_artificial_data[:, *[slice(s, shape) if s >=0 else slice(0, s) for shape, s in zip(space_shape, shift)]] = trimmed_artificial_data

        shifted_artificial_env  = Environment(data_file=shifted_artificial_data,
                                              data_source_position=(space_center + shift),
                                              source_radius=source_radius,
                                              shape=space_shape,
                                              boundary_condition='stop',
                                              start_zone='data_zone',
                                              odor_present_threshold=1e-1)
        # Run test
        hist = run_test(
            agent = ag,
            start_points = starting_points,
            environment = shifted_artificial_env,
            horizon = 1000,
            use_gpu = True,
            print_progress=False
        )

        # Saving history
        hist.save(file=f'source_pos_{source_pos[1]}_{source_pos[0]}', folder=folder)


    # Refresh memory
    cp._default_memory_pool.free_all_blocks()


def main():
    # Set GPU used
    cuda_runtime.setDevice(0)

    grid_sizes = [9]
    sub_grid_sizes = [7]
    covariances = [5,10,25,50]

    root_folder = '/storage/arnaud/test_cov_artif_uniform/'
    os.mkdir(root_folder)

    for grid_s in grid_sizes:
        for sub_grid_s in sub_grid_sizes:
            for cov in covariances:
                print()
                print('----------------------------')
                print(f'({grid_s} by {grid_s})' + ' - ' + f'({sub_grid_s} by {sub_grid_s}) - covariance {cov}')

                folder = root_folder + f'cov_test-{grid_s}_{grid_s}-uniform_rad_5' + '-' + f'{sub_grid_s}_{sub_grid_s}-cov_{cov}'
                os.mkdir(folder)

                run_full_test(grid=grid_s,
                              sub_grid=sub_grid_s,
                              cov=cov,
                              folder=folder)


if __name__ == "__main__":
    main()