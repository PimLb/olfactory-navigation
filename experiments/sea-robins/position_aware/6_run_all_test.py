# Imports
import sys
sys.path.append('../../..')

import numpy as np
import cupy as cp
import os

from olfactory_navigation.environment import Environment
from olfactory_navigation.agents import FSVI_Agent
from olfactory_navigation.agents.model_based_util.pomdp import Model
from olfactory_navigation.simulation import run_test
from olfactory_navigation.test_setups import run_all_starts_test

from matplotlib import pyplot as plt
from matplotlib import colors, patches
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix

from model_6 import generate_model

from cupy.cuda import runtime as cuda_runtime
cuda_runtime.setDevice(0)


# Test function
def run_test(grid, sub_grid):
    # Variables
    space_shape = np.array([101,101])
    cells = np.array(grid)

    source_radius = 2
    source_cell_resolution = np.array(sub_grid)

    data_mean = [50,50]
    data_covariance = 50

    folder = './' + '_'.join([str(i) for i in cells.tolist()]) + '-' + '_'.join([str(i) for i in source_cell_resolution.tolist()])
    os.mkdir(folder)

    # MODEL
    model = generate_model(
        space_shape=space_shape,
        cells=cells,
        source_radius=source_radius,
        source_cell_resolution=source_cell_resolution,
        data_mean=data_mean,
        data_covariance=data_covariance
    )

    # Fake data
    def generate_fake_data(
            space_shape: np.ndarray,
            data_mean: list,
            data_covariance,
            samples: int
            ) -> np.ndarray:
        x,y = np.meshgrid(np.arange(space_shape[0]), np.arange(space_shape[1]))
        pos = np.dstack((x, y))
        rv = multivariate_normal(data_mean, data_covariance)
        probability_map = rv.pdf(pos)

        # Normalize to have 1 at the center
        probability_map /= np.max(probability_map)

        multiple_samples = (np.random.random((samples, *space_shape)) < probability_map).astype(float)

        return multiple_samples

    # Artificial data and env
    artificial_data = generate_fake_data(space_shape=space_shape,
                                        data_mean=data_mean,
                                        data_covariance=data_covariance,
                                        samples=1000)

    artificial_env  = Environment(data_file=artificial_data,
                                data_source_position=[50, 50],
                                source_radius=source_radius,
                                shape=space_shape,
                                boundary_condition='stop',
                                start_zone='data_zone',
                                odor_present_threshold=3e-1)

    # Agent and training
    ag = FSVI_Agent(environment=artificial_env,
                    threshold=3e-8,
                    space_aware=True,
                    spacial_subdivisions=cells,
                    model=model)
    _ = ag.train(expansions=300, use_gpu=True, print_progress=False)

    # Run base test
    hist = run_all_starts_test(ag,
                           horizon=1000,
                           use_gpu=True,
                           print_progress=False)
    hist.save(file='base', folder=folder)

    # Complete test
    for i in range(1,50,2):
        shift = np.array([-i,-i])
        print(f'\n{shift = }')

        # Setup shifted environment
        shifted_artificial_data = np.zeros(artificial_data.shape)
        trimmed_artificial_data = artificial_data[:, *[slice(0, -s) if s > 0 else slice(-s, shape) for shape, s in zip(space_shape, shift)]]
        shifted_artificial_data[:, *[slice(s, shape) if s >=0 else slice(0, s) for shape, s in zip(space_shape, shift)]] = trimmed_artificial_data

        shifted_artificial_env  = Environment(data_file=shifted_artificial_data,
                                            data_source_position=(np.array([50,50]) + shift).tolist(),
                                            source_radius=source_radius,
                                            shape=space_shape,
                                            boundary_condition='stop',
                                            start_zone='data_zone',
                                            odor_present_threshold=3e-1)

        # Running test
        hist = run_all_starts_test(ag,
                                environment=shifted_artificial_env,
                                horizon=1000,
                                use_gpu=True,
                                print_progress=False)

        hist.save(file='diag-' + '_'.join([str(i) for i in shift.tolist()]), folder=folder)

    # Complete test
    point_array = np.array(list(np.ndindex(5,5))) * -2
    for shift in point_array:
        print(f'\n{shift = }')

        # Setup shifted environment
        shifted_artificial_data = np.zeros(artificial_data.shape)
        trimmed_artificial_data = artificial_data[:, *[slice(0, -s) if s > 0 else slice(-s, shape) for shape, s in zip(space_shape, shift)]]
        shifted_artificial_data[:, *[slice(s, shape) if s >=0 else slice(0, s) for shape, s in zip(space_shape, shift)]] = trimmed_artificial_data

        shifted_artificial_env  = Environment(data_file=shifted_artificial_data,
                                            data_source_position=(np.array([50,50]) + shift).tolist(),
                                            source_radius=source_radius,
                                            shape=space_shape,
                                            boundary_condition='stop',
                                            start_zone='data_zone',
                                            odor_present_threshold=3e-1)

        # Running test
        hist = run_all_starts_test(ag,
                                environment=shifted_artificial_env,
                                horizon=1000,
                                use_gpu=True,
                                print_progress=False)

        hist.save(file='grid-' + '_'.join([str(i) for i in shift.tolist()]), folder=folder)

    # Refresh memory
    cp._default_memory_pool.free_all_blocks()


def main():
    # Set GPU used
    cuda_runtime.setDevice(0)

    print('----------------------------')
    grid = np.array([3,3])
    sub_grid = np.array([3,3])
    print(' '.join([str(i) for i in grid.tolist()]) + ' - ' + ' '.join([str(i) for i in sub_grid.tolist()]))
    run_test(grid=grid, sub_grid=sub_grid)

    print('----------------------------')
    grid = np.array([3,3])
    sub_grid = np.array([5,5])
    print(' '.join([str(i) for i in grid.tolist()]) + ' - ' + ' '.join([str(i) for i in sub_grid.tolist()]))
    run_test(grid=grid, sub_grid=sub_grid)

    print('----------------------------')
    grid = np.array([5,5])
    sub_grid = np.array([3,3])
    print(' '.join([str(i) for i in grid.tolist()]) + ' - ' + ' '.join([str(i) for i in sub_grid.tolist()]))
    run_test(grid=grid, sub_grid=sub_grid)

    print('----------------------------')
    grid = np.array([5,5])
    sub_grid = np.array([5,5])
    print(' '.join([str(i) for i in grid.tolist()]) + ' - ' + ' '.join([str(i) for i in sub_grid.tolist()]))
    run_test(grid=grid, sub_grid=sub_grid)

    print('----------------------------')
    grid = np.array([5,5])
    sub_grid = np.array([7,7])
    print(' '.join([str(i) for i in grid.tolist()]) + ' - ' + ' '.join([str(i) for i in sub_grid.tolist()]))
    run_test(grid=grid, sub_grid=sub_grid)

    print('----------------------------')
    grid = np.array([7,7])
    sub_grid = np.array([3,3])
    print(' '.join([str(i) for i in grid.tolist()]) + ' - ' + ' '.join([str(i) for i in sub_grid.tolist()]))
    run_test(grid=grid, sub_grid=sub_grid)

    print('----------------------------')
    grid = np.array([7,7])
    sub_grid = np.array([5,5])
    print(' '.join([str(i) for i in grid.tolist()]) + ' - ' + ' '.join([str(i) for i in sub_grid.tolist()]))
    run_test(grid=grid, sub_grid=sub_grid)

    print('----------------------------')
    grid = np.array([7,7])
    sub_grid = np.array([7,7])
    print(' '.join([str(i) for i in grid.tolist()]) + ' - ' + ' '.join([str(i) for i in sub_grid.tolist()]))
    run_test(grid=grid, sub_grid=sub_grid)

    print('----------------------------')
    grid = np.array([9,9])
    sub_grid = np.array([3,3])
    print(' '.join([str(i) for i in grid.tolist()]) + ' - ' + ' '.join([str(i) for i in sub_grid.tolist()]))
    run_test(grid=grid, sub_grid=sub_grid)

    print('----------------------------')
    grid = np.array([9,9])
    sub_grid = np.array([5,5])
    print(' '.join([str(i) for i in grid.tolist()]) + ' - ' + ' '.join([str(i) for i in sub_grid.tolist()]))
    run_test(grid=grid, sub_grid=sub_grid)

    print('----------------------------')
    grid = np.array([9,9])
    sub_grid = np.array([7,7])
    print(' '.join([str(i) for i in grid.tolist()]) + ' - ' + ' '.join([str(i) for i in sub_grid.tolist()]))
    run_test(grid=grid, sub_grid=sub_grid)



if __name__ == "__main__":
    main()