import sys
sys.path.append('../../..')

from olfactory_navigation import Environment
from olfactory_navigation.agents import FSVI_Agent
from olfactory_navigation.simulation import run_test

from olfactory_navigation.agents.model_based_util.pomdp import Model
from olfactory_navigation.agents.model_based_util.belief import Belief, BeliefSet
from olfactory_navigation.environment import _resize_array
from viz import plot_trajectory_in_tank
from util import generate_results_plots, generate_results_pdf, generate_trajectories_pdf

from matplotlib import pyplot as plt
from datetime import datetime

import pandas as pd
import numpy as np
import cupy as cp
import h5py
import os



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
    # Save results
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


    res_sim_df = all_hist.runs_analysis_df

    res_sim_df['steps_in_05perc_marg'] = array_steps_out_of_bounds[:,0].tolist()
    res_sim_df['steps_in_10perc_marg'] = array_steps_out_of_bounds[:,1].tolist()
    res_sim_df['steps_in_25perc_marg'] = array_steps_out_of_bounds[:,2].tolist()
    res_sim_df['steps_out_25perc_marg'] = array_steps_out_of_bounds[:,3].tolist()

    # 0: no out of bounds, 1: < 5% out of bounds, 2: < 10% out of bounds, 3: < 25% out of bounds, 4: > 25% out of bounds
    res_sim_df['tank_leaving_amount'] = [None]*4 + out_of_bounds_amount.tolist()

    # Adding agent and source position info to result dataframe
    res_sim_df['y_source_cm'] = (new_order_exp_source_indices[:,0] * multiplier).tolist()
    res_sim_df['x_source_cm'] = (new_order_exp_source_indices[:,1] * multiplier).tolist()
    res_sim_df['y_agent_cm'] = (new_order_exp_agent_indices[:,0] * multiplier).tolist()
    res_sim_df['x_agent_cm'] = (new_order_exp_agent_indices[:,1] * multiplier).tolist()
    res_sim_df['odor_field_file'] = assigned_environment_file

    res_sim_df = res_sim_df.drop(columns=['discounted_rewards'])

    # res_sim_df
    res_sim_df.to_csv(test_result_folder + f'results_{all_hist.start_time.strftime("%Y%m%d_%H%M%S")}.csv')


    ################################
    # Results
    ################################
    params = {
        'tank_size': tank_size,
        'step_dt': step_dt,
        'step_dist': step_dist,

    }
    exp_results = {
        'new_order_exp_agent_indices': new_order_exp_agent_indices,
        'new_order_exp_source_indices': new_order_exp_source_indices,
        'out_of_bounds_amount': out_of_bounds_amount
    }
    generate_results_plots(test_result_folder, all_hist, params, exp_results)


    ################################
    # Pdf generation
    ################################
    generate_results_pdf(test_result_folder)
    generate_trajectories_pdf(test_result_folder)





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