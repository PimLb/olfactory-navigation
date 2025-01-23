import sys
sys.path.append('../../..')

from olfactory_navigation import Environment
from olfactory_navigation.agents import FSVI_Agent
from olfactory_navigation.simulation import run_test

from olfactory_navigation.agents.model_based_util.pomdp import Model
from olfactory_navigation.agents.model_based_util.belief import Belief, BeliefSet
from olfactory_navigation.environment import _resize_array
from viz import plot_trajectory_in_tank

from matplotlib import pyplot as plt
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from PIL import Image

import pandas as pd
import numpy as np
import cupy as cp
import h5py
import os
import re



from cupy.cuda import runtime as cuda_runtime
cuda_runtime.setDevice(3)



def run_experiment(
        result_folder:str,
        agent_speed:int=4, # cm/s
        dt:float=0.4, # s
        stop_dt:float=1.3, # s
        max_time:int=900, # s
        actual_tank_size:np.ndarray=np.array([90,114]), # cm
        legs_odor_radius:float=3.0, # cm
        goal_radius:float=1.0, # cm
        thresh_scale:float=1,
        ):
    # agent_speed = 4 # Walking speed cm/s
    # dt = 0.4 # s (walking for an average of 0.4s)
    # stop_dt = 1.3 # s (average stop time)
    step_dt = dt + stop_dt
    step_dist = agent_speed * dt # cm

    # max_time = 900 # s (Experiments topped after 15mins)
    max_steps = int(max_time / step_dt)

    multiplier = agent_speed * dt
    # actual_tank_size = np.array([90,114]) # actual size: 90cm x 114cm but it is scaled by a walking speed of <multiplier>
    tank_size = (actual_tank_size / multiplier).astype(int)
    space_shape = (tank_size * 2) + 1

    source_position = tank_size # Center of the space (ie the doubled tank)

    goal_radius /= multiplier
    legs_odor_radius /= multiplier # 3 cm * 0.8 -> 3.75 grid units

    print(f'Tank size: {tank_size.tolist()}; Space shape {space_shape.tolist()}; States count: {space_shape.prod()}')
    print(f'Agent speed: {agent_speed} cm/s; dt: {dt} s; stop_dt: {stop_dt} s; max_time: {max_time} s; step_dt: {step_dt} s; step_dist: {step_dist} cm')
    print(f'Goal radius: {goal_radius}; Legs odor radius: {legs_odor_radius}')
    print()

    # Odor variables
    # thresh_scale = 1.5
    thresh = 3 / 10**(thresh_scale)

    experiment_data_file = f'/storage/arnaud/datasets/2024_11_22_sea_robins_experimental_data/experimental_positions.csv'
    base_odor_plume_file = f'/storage/arnaud/datasets/2024_11_08_sea_robins_plume_averages/average_odor_base_3e{thresh_scale}.npy'
    odor_fields_folder = '/storage/arnaud/datasets/2024_06_13_tank_odor_field/'
    test_result_folder = result_folder + f'/results-thresh_3e{thresh_scale}-{datetime.now().strftime("%Y%m%d_%H%M%S")}/'


    ################################
    # Building POMDP model
    ################################
    # Legs odor field
    odor_field = np.zeros(space_shape)
    odor_field_indices = np.array(list(np.ndindex(tuple(space_shape))))

    in_odor_plume = np.sum((odor_field_indices - tank_size[None,:])**2, axis=1) <= (legs_odor_radius ** 2)
    odor_field[*odor_field_indices[in_odor_plume].T] = 1.0

    # Gathering nose odor plume
    average_odor_plume = np.load(base_odor_plume_file)

    odor_shape = np.array([408, 488])
    odor_center = np.array([500, 500])

    lower_bound = odor_center - (odor_shape / 2).astype(int)
    upper_bound = odor_center + (odor_shape / 2).astype(int)
    slices = [slice(lb, ub) for lb, ub in zip(lower_bound, upper_bound)]

    average_odor_plume = average_odor_plume[*slices]

    assert all(average_odor_plume.shape == odor_shape)

    # Nose odor field
    tank_average_odor_plume = _resize_array(average_odor_plume, tank_size, 'linear')

    # Putting odor plume average in
    nose_odor_field = np.zeros(odor_field.shape)
    lower_bounds, upper_bounds = (tank_size/2).astype(int), (tank_size/2).astype(int) + tank_size
    nose_odor_field[*[slice(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds)]] = tank_average_odor_plume

    thresholds = np.array([-np.inf,1,2,np.inf])
    action_set = np.array([
        [-1,  0], # North
        [ 0,  1], # East
        [ 1,  0], # South
        [ 0, -1]  # West
    ])
    action_labels = [
        'North',
        'East',
        'South',
        'West'
    ]

    # Base Model parameters
    state_count = np.prod(space_shape)

    state_grid = [[f's_{x}_{y}' for x in range(space_shape[1])] for y in range(space_shape[0])]
    end_states = np.argwhere(np.fromfunction(lambda x,y: ((x-source_position[0])**2 + (y-source_position[1])**2) <= goal_radius**2,
                                                shape=space_shape).ravel())[:,0].tolist()

    # Counts
    action_count = len(action_set)
    observation_count = thresholds.shape[-1] # Thresholds minus 1; plus 1 for the goal.

    # Building observation matrix
    observations = np.empty((state_count, action_count, observation_count), dtype=float)

    filt_nose_odor_field = nose_odor_field.ravel()
    filt_nose_odor_field[odor_field.ravel() > 0.0] = 0.0

    observations[:,:,1] = filt_nose_odor_field[:,None] # Nose
    observations[:,:,2] = odor_field.ravel()[:,None] # Nose AND legs
    observations[:,:,0] = 1.0 - observations[:,:,1] - observations[:,:,2] # Nothing

    # Goal observation
    observations[:,:,-1] = 0.0
    observations[end_states,:,:] = 0.0
    observations[end_states,:,-1] = 1.0

    # Assert observations sum to 1
    assert np.all(np.sum(observations, axis=2) == 1.0), "Observation table malformed, something is wrong..."

    # Observation labels
    observation_labels = ['nothing', 'something_nose', 'something_nose_legs', 'goal']

    # Compute reachable states
    points = np.array(np.unravel_index(np.arange(np.prod(space_shape)), space_shape)).T

    # For each actions compute all new grid points (using the environment.move method)
    action_new_states = []
    movements = action_set
    for move_vector in movements:

        # Applying the movement vector
        new_points = points + move_vector

        # Wrap points
        new_points = np.where(new_points < 0, (new_points + space_shape[None,:]), new_points)
        new_points = np.where(new_points >= space_shape[None,:], (new_points - space_shape[None,:]), new_points)

        new_states = np.ravel_multi_index((new_points[:,0], new_points[:,1]), dims=space_shape)
        action_new_states.append(new_states)

    # Forming it the reachable states array from the new states for each action
    reachable_states = np.array(action_new_states).T[:,:,None]

    # Instantiate the model object
    model = Model(
        states = state_grid,
        actions = action_labels,
        observations = observation_labels,
        reachable_states = reachable_states,
        observation_table = observations,
        end_states = end_states
    )


    ################################
    # Building Environment
    ################################
    odor_field_files = os.listdir(odor_fields_folder)
    odor_field_environments = []
    odor_field_source_pos = []

    for file_i, file in enumerate(odor_field_files):
        print(f'[{file_i+1}/{len(odor_field_files)}] Loading odor field: {odor_fields_folder}{file}')

        # Loading data
        data = h5py.File(odor_fields_folder + file, 'r')

        # Finding out start and end times
        time_slices = data['odor_base']
        time_slices = sorted([int(key_name) for key_name in time_slices.keys()])

        start_time = min(time_slices)
        end_time = max(time_slices)

        # Gathering data
        odor = data['odor_base']
        odor_data = [odor[str(time)] for time in range(start_time, end_time)]
        odor_data_array = np.array(odor_data)

        # Resizing the data array
        new_odor_data_array = np.zeros((len(odor_data_array), *tank_size))
        for i, data_slice in enumerate(odor_data_array):
            new_odor_data_array[i] = _resize_array(data_slice,
                                                new_shape=tank_size,
                                                interpolation='linear')

        # Interpolating the data array to have slices every 1.7 seconds
        time_indices = np.arange(0, len(odor_data_array), 1.7)
        interpolated_odor_data_array = np.zeros((len(time_indices), *tank_size))
        for i in range(tank_size[0]):
            for j in range(tank_size[1]):
                interpolated_odor_data_array[:, i, j] = np.interp(time_indices, np.arange(len(odor_data_array)), new_odor_data_array[:, i, j])
        new_odor_data_array = interpolated_odor_data_array

        # compute the odor source position
        source_pos = np.unravel_index(np.argmax(new_odor_data_array[0]), shape=new_odor_data_array[0].shape)

        # Centering the odor field
        centered_odor_data_array = np.zeros((len(new_odor_data_array), *space_shape))
        for i, data_slice in enumerate(new_odor_data_array):
            centered_odor_data_array[i,
                                    (tank_size[0] - source_pos[0]) : (tank_size[0] - source_pos[0] + tank_size[0]),
                                    (tank_size[1] - source_pos[1]) : (tank_size[1] - source_pos[1] + tank_size[1])] = data_slice

        # Filtering the odor field
        filt_odor_data_array = (centered_odor_data_array > thresh).astype(float)

        legs_odor = np.argwhere(odor_field > 0)
        for i in range(len(filt_odor_data_array)):
            filt_odor_data_array[i,*legs_odor.T] = 2.0

        # Trimming the odor data array to stop at the "max_time" amount
        filt_odor_data_array = filt_odor_data_array[:max_steps]

        # Setting up environment
        env = Environment(data_file = filt_odor_data_array,
                        data_source_position = tank_size,
                        source_radius = goal_radius,
                        boundary_condition = 'wrap')

        # Saving environment and source positions
        odor_field_environments.append(env)
        odor_field_source_pos.append(np.array(source_pos) / np.array(new_odor_data_array.shape[1:]))


    ################################
    # Agent Training
    ################################
    ag = FSVI_Agent(odor_field_environments[0],
                    model=model,
                    thresholds=[-np.inf, 1.0, 2.0, np.inf])

    _ = ag.train(expansions=800,
                max_belief_growth=100,
                eps=1e-8,
                print_progress=False,
                use_gpu=True)


    ################################
    # Run test
    ################################
    df = pd.read_csv(experiment_data_file)

    exp_source_indices = df[['y_food_cm','x_food_cm']].to_numpy().astype(float)
    exp_agent_indices = df[['y0_cm', 'x0_cm']].to_numpy().astype(float)

    exp_source_indices /= multiplier
    exp_agent_indices /= multiplier

    exp_source_indices = exp_source_indices.astype(int)
    exp_agent_indices = exp_agent_indices.astype(int)


    source_distances = []
    odor_field_source_positions_in_tank = np.array(odor_field_source_pos) * tank_size

    for source_pos in odor_field_source_positions_in_tank:
        source_distances.append(np.sum((exp_source_indices - source_pos) ** 2, axis=1) ** (1/2))

    assigned_environment = np.argmin(np.array(source_distances), axis=0)


    # Generation of points
    n = len(exp_source_indices)

    # STATE SPACE
    center_state = exp_agent_indices
    agent_start_state = center_state + (tank_size[None,:] - exp_source_indices)

    beliefs = []

    for center, start in zip(center_state, agent_start_state):
        belief = np.zeros(space_shape)
        belief[center[0]:center[0]+tank_size[0], center[1]:center[1]+tank_size[1]] = 1
        belief /= np.sum(belief)

        beliefs.append(belief.flatten())

    beliefs = np.array(beliefs)


    all_hist = None

    source_ids = np.arange(len(exp_source_indices))
    new_order_source_ids = []

    for source_i, source_pos in enumerate(odor_field_source_positions_in_tank):
        print(f'[{source_i+1}/{len(odor_field_source_positions_in_tank)}] Environment {source_i}')

        # Checking which experimental source points have been assigned to the current environment
        is_assigned_model = (assigned_environment == source_i)
        env = odor_field_environments[source_i]
        new_order_source_ids += source_ids[is_assigned_model].tolist()

        # Skip if no experimental source point is assigned to the environment
        if np.sum(is_assigned_model) == 0:
            continue

        belief = BeliefSet(ag.model, beliefs[is_assigned_model])
        hist = run_test(agent=ag,
                        start_points=agent_start_state[is_assigned_model],
                        environment=env,
                        horizon=max_steps,
                        initialization_values={'belief': belief},
                        use_gpu=True,
                        print_progress=False,
                        print_stats=False)

        # Combine the history to the previous history instance
        if all_hist is None:
            all_hist = hist
        else:
            all_hist += hist

    print(all_hist.summary)


    if not os.path.isdir(test_result_folder):
        os.mkdir(test_result_folder)


    all_hist.save(folder=test_result_folder)


    plt.imshow(tank_average_odor_plume, cmap='Greys')
    plt.colorbar()
    plt.savefig(test_result_folder + 'plume.png')


    ################################
    # Results
    ################################
    # Re-ordering source and agent positions
    new_order_exp_agent_indices = exp_agent_indices[new_order_source_ids]
    new_order_exp_source_indices = exp_source_indices[new_order_source_ids]
    new_order_assigned_environment = assigned_environment[new_order_source_ids]
    assigned_environment_file = [(odor_fields_folder + odor_field_files[env_i]) for env_i in new_order_assigned_environment]


    # Computing the amount of steps out of bounds
    list_steps_out_of_bounds = []

    for traj in range(len(new_order_exp_agent_indices)):
        # Retrieving sim
        sim = all_hist.simulation_dfs[traj]

        # Compute shift
        start_coord = sim[['y', 'x']].to_numpy()[0]
        shift = start_coord - new_order_exp_agent_indices[traj]

        # Retrieving sequence
        seq = sim[['y','x']].to_numpy() - shift[None,:]

        # Computing steps out of bounds
        steps_out_of_bounds = np.sum(np.any((seq < 0) | (seq >= tank_size), axis=1))

        # Compute the bounds for 5% and 10% of tank_size
        bounds_5_percent = tank_size * 0.05
        bounds_10_percent = tank_size * 0.10
        bounds_25_percent = tank_size * 0.25

        # Compute steps out of bounds by 5% and 10% using Manhattan distance
        steps_out_of_bounds_5_percent = np.sum(np.any((seq < -bounds_5_percent) | (seq >= tank_size + bounds_5_percent), axis=1))
        steps_out_of_bounds_10_percent = np.sum(np.any((seq < -bounds_10_percent) | (seq >= tank_size + bounds_10_percent), axis=1))
        steps_out_of_bounds_25_percent = np.sum(np.any((seq < -bounds_25_percent) | (seq >= tank_size + bounds_25_percent), axis=1))

        # Append the results to the list
        list_steps_out_of_bounds.append([steps_out_of_bounds, steps_out_of_bounds_5_percent, steps_out_of_bounds_10_percent, steps_out_of_bounds_25_percent])

    array_steps_out_of_bounds = np.array(list_steps_out_of_bounds)
    out_of_bounds_amount = np.sum(array_steps_out_of_bounds > 0, axis=1)

    count_out_of_bounds = len(np.argwhere(array_steps_out_of_bounds[:,0]))

    print(f'Simulations with steps out of bounds: {count_out_of_bounds} / {len(array_steps_out_of_bounds[:,0])}' + ('' if count_out_of_bounds == 0 else f' (avg length {np.mean(array_steps_out_of_bounds[array_steps_out_of_bounds[:,0] > 0, 0]):.2f})'))
    tot = 0
    for i, bound in enumerate([0, 5, 10, 25]):
        count = np.sum(out_of_bounds_amount == i)
        tot += count
        print(f'   - Inside {bound}%: {tot}')


    res_sim_df = all_hist.analysis_df

    res_sim_df['steps_in_05perc_marg'] = [None]*4 + array_steps_out_of_bounds[:,0].tolist()
    res_sim_df['steps_in_10perc_marg'] = [None]*4 + array_steps_out_of_bounds[:,1].tolist()
    res_sim_df['steps_in_25perc_marg'] = [None]*4 + array_steps_out_of_bounds[:,2].tolist()
    res_sim_df['steps_out_25perc_marg'] = [None]*4 + array_steps_out_of_bounds[:,3].tolist()

    # 0: no out of bounds, 1: < 5% out of bounds, 2: < 10% out of bounds, 3: < 25% out of bounds, 4: > 25% out of bounds
    res_sim_df['tank_leaving_amount'] = [None]*4 + out_of_bounds_amount.tolist()

    # Adding agent and source position info to result dataframe
    res_sim_df['y_source_cm'] = [None]*4 + (new_order_exp_source_indices[:,0] * multiplier).tolist()
    res_sim_df['x_source_cm'] = [None]*4 + (new_order_exp_source_indices[:,1] * multiplier).tolist()
    res_sim_df['y_agent_cm'] = [None]*4 + (new_order_exp_agent_indices[:,0] * multiplier).tolist()
    res_sim_df['x_agent_cm'] = [None]*4 + (new_order_exp_agent_indices[:,1] * multiplier).tolist()
    res_sim_df['odor_field_file'] = [None]*4 + assigned_environment_file

    res_sim_df = res_sim_df.drop(columns=['discounted_rewards'])

    # res_sim_df
    res_sim_df.to_csv(test_result_folder + f'results_{all_hist.start_time.strftime("%Y%m%d_%H%M%S")}.csv')


    # Save all trajectories to folder
    if not os.path.isdir(test_result_folder + 'trajectories/'):
        os.mkdir(test_result_folder + 'trajectories/')

    out_of_traj_append = ['', '-in_05perc_marg', '-in_10perc_marg', '-in_25perc_marg', '-out_25perc_marg']


    ################################
    # Traj plots
    ################################
    for i in range(len(new_order_exp_agent_indices)):
        fig, ax = plt.subplots(figsize=(10,10))
        plot_trajectory_in_tank(h = all_hist,
                                exp_agent = new_order_exp_agent_indices,
                                exp_source = new_order_exp_source_indices,
                                t_size = tank_size,
                                traj = i,
                                ax = ax)

        plt.savefig(test_result_folder + 'trajectories/' + f'run-{i}{out_of_traj_append[out_of_bounds_amount[i]]}.png')
        plt.close(fig)


    ################################
    # Plots folders
    ################################
    a_df = all_hist.analysis_df
    runs_df = a_df[[str(i).startswith('run_') for i in a_df.index]]
    run_is_success = ~runs_df['reached_horizon'].astype(bool)
    success_runs_df = runs_df.loc[run_is_success]


    plot_folders = ['plots_in_0perc_marg/', 'plots_in_5perc_marg/', 'plots_in_10perc_marg/', 'plots_in_25perc_marg/', 'plots_all/']

    # Create folders for plots
    for folder in plot_folders:
        if not os.path.isdir(test_result_folder + folder):
            os.mkdir(test_result_folder + folder)


    ################################
    # Time taken plots
    ################################
    # Saving the plots
    for i, folder in enumerate(plot_folders):
        filtered_runs_df = runs_df[out_of_bounds_amount <= i]
        ax = (filtered_runs_df['steps_taken'] * step_dt).hist(grid=False, bins=20, figsize=(10,5))
        ax.set_xlabel('Time (s)')
        plt.savefig(test_result_folder + folder + 'time_taken.png')
        plt.close()

    # Saving the plots - 50bins
    for i, folder in enumerate(plot_folders):
        filtered_runs_df = runs_df[out_of_bounds_amount <= i]
        ax = (filtered_runs_df['steps_taken'] * step_dt).hist(grid=False, bins=50, figsize=(10,5))
        ax.set_xlabel('Time (s)')
        plt.savefig(test_result_folder + folder + 'time_taken_50b.png')
        plt.close()

    # Saving the plots
    for i, folder in enumerate(plot_folders):
        filtered_success_runs_df = success_runs_df[(out_of_bounds_amount <= i)[run_is_success]]
        ax = (filtered_success_runs_df['steps_taken'] * step_dt).hist(grid=False, bins=20, figsize=(10,5))
        ax.set_xlabel('Time (s)')
        plt.savefig(test_result_folder + folder + 'time_taken_success_only.png')
        plt.close()

    # Saving the plots - 50bins
    for i, folder in enumerate(plot_folders):
        filtered_success_runs_df = success_runs_df[(out_of_bounds_amount <= i)[run_is_success]]
        ax = (filtered_success_runs_df['steps_taken'] * step_dt).hist(grid=False, bins=50, figsize=(10,5))
        ax.set_xlabel('Time (s)')
        plt.savefig(test_result_folder + folder + 'time_taken_50b_success_only.png')
        plt.close()


    ################################
    # Distance plots
    ################################
    # Saving the plots
    # Division by 100 to convert to meters
    for i, folder in enumerate(plot_folders):
        filtered_runs_df = runs_df[out_of_bounds_amount <= i]
        ax = ((filtered_runs_df['steps_taken'] * step_dist) / 100).hist(grid=False, bins=20, figsize=(10,5))
        ax.set_xlabel('Distance (m)')
        plt.savefig(test_result_folder + folder + 'distance.png')
        plt.close()

    # Saving the plots - 50bins
    # Division by 100 to convert to meters
    for i, folder in enumerate(plot_folders):
        filtered_runs_df = runs_df[out_of_bounds_amount <= i]
        ax = ((filtered_runs_df['steps_taken'] * step_dist) / 100).hist(grid=False, bins=50, figsize=(10,5))
        ax.set_xlabel('Distance (m)')
        plt.savefig(test_result_folder + folder + 'distance_50b.png')
        plt.close()

    # Saving the plots
    # Division by 100 to convert to meters
    for i, folder in enumerate(plot_folders):
        filtered_success_runs_df = success_runs_df[(out_of_bounds_amount <= i)[run_is_success]]
        ax = ((filtered_success_runs_df['steps_taken'] * step_dist) / 100).hist(grid=False, bins=20, figsize=(10,5))
        ax.set_xlabel('Distance (m)')
        plt.savefig(test_result_folder + folder + 'distance_success_only.png')
        plt.close()

    # Saving the plots - 50bins
    # Division by 100 to convert to meters
    for i, folder in enumerate(plot_folders):
        filtered_success_runs_df = success_runs_df[(out_of_bounds_amount <= i)[run_is_success]]
        ax = ((filtered_success_runs_df['steps_taken'] * step_dist) / 100).hist(grid=False, bins=50, figsize=(10,5))
        ax.set_xlabel('Distance (m)')
        plt.savefig(test_result_folder + folder + 'distance_50b_success_only.png')
        plt.close()


    ################################
    # Grid plots
    ################################
    all_success_array = []
    all_speed_array = []
    all_speed_success_array = []
    all_count_array = []
    all_count_success_array = []

    for i, folder in enumerate(plot_folders):
        filtered_runs_df = runs_df[out_of_bounds_amount <= i]
        filtered_success_runs_df = success_runs_df[(out_of_bounds_amount <= i)[run_is_success]]

        # Retrieving successes and speed from dataframe
        successes = np.array(filtered_runs_df['converged'])
        speed = np.array(filtered_runs_df['steps_taken'])
        speed_success = np.array(filtered_success_runs_df['steps_taken'])

        # Compute the grid to use
        point_array = new_order_exp_source_indices[out_of_bounds_amount <= i] # exp_agent_indices or exp_source_indices

        grid = np.array([5,8])
        cell_indices = np.array(list(np.ndindex(tuple(grid))))
        cell_sizes = tank_size / grid

        point_cell = (point_array / cell_sizes).astype(int)

        # Compute successes and speeds in the grid
        success_array = np.zeros(grid, dtype=float)
        speed_array = np.zeros(grid, dtype=float)
        speed_success_array = np.zeros(grid, dtype=float)
        count_array = np.zeros(grid, dtype=int)
        count_success_array = np.zeros(grid, dtype=int)

        for cell in cell_indices:
            point_in_cell = np.all(point_cell == cell, axis=1)
            count_in_cell = np.sum(point_in_cell)
            count_array[*cell] = count_in_cell
            if count_in_cell == 0:
                continue

            success_perc = np.mean(successes[point_in_cell])
            average_speed = np.mean(speed[point_in_cell])

            success_array[*cell] = success_perc
            speed_array[*cell] = average_speed

            count_in_cell_success = np.sum(point_in_cell[run_is_success[out_of_bounds_amount <= i]])
            count_success_array[*cell] = count_in_cell_success
            if count_in_cell_success == 0:
                continue

            average_speed_success = np.mean(speed_success[point_in_cell[run_is_success[out_of_bounds_amount <= i]]])
            speed_success_array[*cell] = average_speed_success

        all_success_array.append(success_array)
        all_speed_array.append(speed_array)
        all_speed_success_array.append(speed_success_array)
        all_count_array.append(count_array)
        all_count_success_array.append(count_success_array)

    # Convergence
    for i, folder in enumerate(plot_folders):
        plt.figure(figsize=(10,5))

        plt.imshow(all_success_array[i] * 100, cmap='Blues', vmin=0, vmax=100)
        cbar = plt.colorbar(ticks=[0,100])
        cbar.set_label('Success Rate (%)')

        no_sim = np.argwhere(all_count_array[i] == 0)
        plt.scatter(no_sim[:,1], no_sim[:,0], c='grey', marker='x', s=100, label='No Simulations')

        plt.savefig(test_result_folder + folder + 'grid_success_rate.png')
        plt.close()

    # Time taken
    for i, folder in enumerate(plot_folders):
        plt.figure(figsize=(10,5))

        upper_bound = (np.ceil(np.max(all_speed_array[i] * step_dt) / 100) * 100).astype(int)
        plt.imshow(all_speed_array[i] * step_dt, cmap='Blues', vmin=0, vmax=upper_bound)
        cbar = plt.colorbar(ticks=[0,upper_bound])
        cbar.set_label('Time Taken (s)')

        no_sim = np.argwhere(all_count_array[i] == 0)
        plt.scatter(no_sim[:,1], no_sim[:,0], c='grey', marker='x', s=100, label='No Simulations')

        plt.savefig(test_result_folder + folder + 'grid_time_taken.png')
        plt.close()

    # Time taken - success
    for i, folder in enumerate(plot_folders):
        plt.figure(figsize=(10,5))

        upper_bound = (np.ceil(np.max(all_speed_success_array[i] * step_dt) / 100) * 100).astype(int)
        plt.imshow(all_speed_success_array[i] * step_dt, cmap='Blues', vmin=0, vmax=upper_bound)
        cbar = plt.colorbar(ticks=[0,upper_bound])
        cbar.set_label('Time Taken (s)')

        no_sim = np.argwhere(all_count_success_array[i] == 0)
        plt.scatter(no_sim[:,1], no_sim[:,0], c='grey', marker='x', s=100, label='No Simulations')

        plt.savefig(test_result_folder + folder + 'grid_time_taken_success_only.png')
        plt.close()

    # Distance
    for i, folder in enumerate(plot_folders):
        plt.figure(figsize=(10,5))

        upper_bound = (np.ceil(np.max(all_speed_array[i] * step_dist) / 100)).astype(int)
        plt.imshow((all_speed_array[i] * step_dist) / 100, cmap='Blues', vmin=0, vmax=upper_bound)
        cbar = plt.colorbar(ticks=[0,upper_bound])
        cbar.set_label('Distance (m)')

        no_sim = np.argwhere(all_count_array[i] == 0)
        plt.scatter(no_sim[:,1], no_sim[:,0], c='grey', marker='x', s=100, label='No Simulations')

        plt.savefig(test_result_folder + folder + 'grid_distance.png')
        plt.close()

    # Distance - success
    for i, folder in enumerate(plot_folders):
        plt.figure(figsize=(10,5))

        upper_bound = (np.ceil(np.max(all_speed_success_array[i] * step_dist) / 100)).astype(int)
        plt.imshow((all_speed_success_array[i] * step_dist) / 100, cmap='Blues', vmin=0, vmax=upper_bound)
        cbar = plt.colorbar(ticks=[0,upper_bound])
        cbar.set_label('Distance (m)')

        no_sim = np.argwhere(all_count_success_array[i] == 0)
        plt.scatter(no_sim[:,1], no_sim[:,0], c='grey', marker='x', s=100, label='No Simulations')

        plt.savefig(test_result_folder + folder + 'grid_distance_success_only.png')
        plt.close()


    ################################
    # Pdf generation
    ################################
    folder = test_result_folder

    plot_sets = [
        'No filter',
        'Filtered all exitting',
        'Filtered all out of 5% margin',
        'Filtered all out of 10% margin',
        'Filtered all out of 25% margin'
    ]

    # Create pdf folder
    if not os.path.isdir(folder + 'results_pdfs/'):
        os.mkdir(folder + 'results_pdfs/')

    for i, plot_folder_name in enumerate(['plots_all', 'plots_in_0perc_marg', 'plots_in_5perc_marg', 'plots_in_10perc_marg', 'plots_in_25perc_marg']):
        plot_folder = folder + plot_folder_name + '/'

        # Extracting the threshold level from the folder name
        thresh_level = float(folder.split('thresh_')[1].split('-')[0].split('e')[1])

        # Basic setup
        pdf_file_name = folder + 'results_pdfs/' + 'results_' + plot_folder_name + '.pdf'
        c = canvas.Canvas(pdf_file_name, pagesize=A4)
        width, height = A4
        margin = 10

        # List of png files in the specified order
        png_files = ['grid_success_rate', 'time_taken', 'grid_time_taken', 'distance', 'grid_distance']
        png_files = [os.path.join(plot_folder, f'{name}.png') for name in png_files]

        x_left = margin
        x_right = width / 2 + margin
        y = height - margin

        # Set title
        c.setFont("Helvetica-Bold", 18)
        c.drawString(x_left, y - 15, f"Results - Threshold: 3e-{thresh_level} - {plot_sets[i]}")
        y -= 20

        for png_file in png_files:
            if '/time_taken.png' in png_file:
                c.setFont("Helvetica-Bold", 12)
                c.drawString(x_left + margin, y - 15, "Speed of the agent (seconds)")
                y -= 20

            if '/distance.png' in png_file:
                c.setFont("Helvetica-Bold", 12)
                c.drawString(x_left + margin, y - 15, "Distance travelled by the agent (meters)")
                y -= 20

            img = Image.open(png_file)
            img_width, img_height = img.size

            # Resize the image to fit the PDF page
            aspect = img_width / img_height
            if aspect > 1:
                img_width = (width / 2) - (2 * margin)
                img_height = img_width / aspect
            else:
                img_height = (height / 2) - (2 * margin)
                img_width = img_height * aspect

            if y - img_height < margin:
                c.showPage()
                y = height - margin

            c.drawImage(png_file, x_left, y - img_height, img_width, img_height)
            if 'grid_success_rate' in png_file: # Add the plume image
                img = Image.open(folder + 'plume.png')
                plume_img_width, plume_img_height = img.size
                plume_img_width = min(plume_img_width, img_width)
                plume_img_height = min(plume_img_height, img_height)

                # Resize the image to fit the PDF page
                aspect = plume_img_width / plume_img_height
                if aspect > 1:
                    plume_img_width = (width / 2) - (2 * margin)
                    plume_img_height = plume_img_width / aspect
                else:
                    plume_img_height = (height / 2) - (2 * margin)
                    plume_img_width = plume_img_height * aspect

                c.drawImage(folder + 'plume.png', x_right, y - img_height, plume_img_width, plume_img_height)
            else: # Plotting _success_only versions of the plots except for the success_rate
                c.drawImage(png_file.replace('.png', '_success_only.png'), x_right, y - img_height, img_width, img_height)
            y -= img_height + margin

        c.save()
        print(f"PDF saved as {pdf_file_name}")

    # Trajectories
    folder = test_result_folder

    # Define the folder containing the trajectory images
    trajectory_folder = os.path.join(folder, 'trajectories')

    # Get the list of trajectory PNG files
    trajectory_files = [f for f in os.listdir(trajectory_folder) if f.endswith('.png')]

    # Sort the files by the number in the filename
    trajectory_files.sort(key=lambda x: int(re.search(r'run-(\d+)', x).group(1)))

    # Create a new PDF file for the trajectories
    trajectory_pdf_file_name = os.path.join(folder + 'results_pdfs/', 'trajectories.pdf')
    c = canvas.Canvas(trajectory_pdf_file_name, pagesize=A4)
    width, height = A4
    margin = 5

    # Set the number of columns
    num_columns = 4
    column_width = (width - (num_columns + 1) * margin) / num_columns

    # Initialize the position
    x = margin
    y = height - margin

    for i, trajectory_file in enumerate(trajectory_files):
        # Open the image
        img = Image.open(os.path.join(trajectory_folder, trajectory_file))
        img_width, img_height = img.size

        # Resize the image to fit the column width
        aspect = img_width / img_height
        img_width = column_width
        img_height = img_width / aspect

        # Check if we need to move to the next row
        if x + img_width + margin > width:
            x = margin
            y -= img_height + 2 * margin + 15  # 15 for the text height

        # Check if we need to add a new page
        if y - img_height < margin:
            c.showPage()
            x = margin
            y = height - margin

        # Draw the file name above the image
        c.setFont("Helvetica", 8)
        c.drawString(x, y - 13, trajectory_file.removesuffix('.png'))

        # Draw the image
        c.drawImage(os.path.join(trajectory_folder, trajectory_file), x, y - img_height - 15, img_width, img_height)

        # Move to the next column
        x += img_width + margin

    # Save the PDF
    c.save()
    print(f"Trajectory PDF saved as {trajectory_pdf_file_name}")





def main():
    thresh_scales = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    goal_radii = [3.2, 2.4, 0.8, 0.0] # 1.6 already done



    for gi, goal_rad in enumerate(goal_radii):
        for ti, thresh_scale in enumerate(thresh_scales):
            print(f'[{gi*len(thresh_scales)+ti}/{len(thresh_scales)*len(goal_radii)}] Running experiment for goal radius {goal_rad} and threshold scale {thresh_scale}')
            print('--------------------------------------------------------------------------------\n')

            res_folder = f'/storage/arnaud/results/sea_robins/2025_01_22-results_relative_position_model-goal_rad_{goal_rad}'
            if not os.path.isdir(res_folder):
                os.mkdir(res_folder)

            run_experiment(
                result_folder=res_folder,
                thresh_scale=thresh_scale,
                goal_radius=goal_rad)

            print('\n\n--------------------------------------------------------------------------------\n')

            # Refresh memory
            cp._default_memory_pool.free_all_blocks()


if __name__ == "__main__":
    main()