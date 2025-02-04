from olfactory_navigation import Agent, Environment
from olfactory_navigation.agents.model_based_util.pomdp import Model

import numpy as np


def exact_converter(agent : Agent) -> Model:
    '''
    Method to create a POMDP model based on an olfactory environment object.

    This version of the converter converts the environment in an exact manner.
    This mean the amount of states is equal to the grid points in the olfactory environment object.

    It supports an environment in 2D, with or without layers. It supports a variety of different action sets from the agent.

    It also defines at least 3 different observations: Nothing, Something or Goal.
    However, if multiple thresholds are provided, the more observations will be available: |threshold| + 1 (Nothing) + 1 (Goal)

    Note: The environment and the threshold parameters are gathered from the agent instance provided.

    Parameters
    ----------
    agent : Agent
        The agent to use to get the environment and threshold parameters from.

    Returns
    -------
    model : Model
        A generate POMDP model from the environment.
    '''
    # Agent's parameters
    environment = agent.environment
    thresholds = agent.thresholds
    action_set = agent.action_set

    # Assertions
    assert environment.dimensions == 2, "This converter only works for 2D environments..." # TODO: implement for ND
    assert not agent.space_aware, "This model is not compatible with space_aware agents as the observations would not match the expected observations of this model."

    # Base Model parameters
    state_count = np.prod(environment.shape)

    state_grid = [[f's_{x}_{y}' for x in range(environment.shape[1])] for y in range(environment.shape[0])]
    end_states = np.argwhere(np.fromfunction(lambda x,y: ((x-environment.source_position[0])**2 + (y-environment.source_position[1])**2) <= environment.source_radius**2,
                                                shape=environment.shape).ravel())[:,0].tolist()

    # Counts
    action_count = len(agent.action_set)
    observation_count = thresholds.shape[-1] # Thresholds minus 1; plus 1 for the goal.

    # Computing odor probabilities
    odor_fields = None
    data_bounds_slices = tuple(slice(low, high) for low, high in environment.data_bounds)
    if environment.has_layers:
        odor_fields = []
        for layer in environment.layers:
            data_grid = environment.data[layer,:,:,:,None]
            threshs = thresholds[layer] if len(thresholds.shape) == 2 else thresholds
            data_odor_fields = np.average(((data_grid >= threshs[:-1][None,None,None,:]) & (data_grid < threshs[1:][None,None,None,:])), axis=0)

            # Increasing it to the full environment
            field = np.zeros(environment.shape + (observation_count-1,))
            field[*data_bounds_slices, :] = data_odor_fields

            odor_fields.append(field)

    else:
        data_grid = environment.data[:,:,:,None]
        threshs = thresholds
        data_odor_fields = np.average(((data_grid >= threshs[:-1][None,None,None,:]) & (data_grid < threshs[1:][None,None,None,:])), axis=0)

        # Increasing it to the full environment
        odor_fields = np.zeros(environment.shape + (observation_count-1,))
        odor_fields[*data_bounds_slices, :] = data_odor_fields

    # Building observation matrix
    observations = np.empty((state_count, action_count, observation_count), dtype=float)
    for o in range(observation_count-1): # Skipping the goal observation
        for a, action_vector in enumerate(action_set):
            if environment.has_layers:
                action_layer = action_vector[0]
                observations[:,a,o] = odor_fields[action_layer][:,:,o].ravel()
            else:
                observations[:,a,o] = odor_fields[:,:,o].ravel()

    # Setting 'Nothing' observation in the margins to 1
    data_margins_mask = np.ones(environment.shape, dtype=bool)
    data_margins_mask[data_bounds_slices] = False
    observations[data_margins_mask.ravel(),:,0] = 1.0

    # Goal observation
    observations[:,:,-1] = 0.0
    observations[end_states,:,:] = 0.0
    observations[end_states,:,-1] = 1.0

    # Assert observations sum to 1
    assert np.all(np.sum(observations, axis=2) == 1.0), "Observation table malformed, something is wrong..."

    # Observation labels
    observation_labels = ['nothing']
    if thresholds.shape[-1] > 3:
        for i in range(thresholds.shape[-1] - 2):
            observation_labels.append(f'something_l{i}')
    else:
        observation_labels.append('something')
    observation_labels.append('goal')

    # Compute reachable states
    shape = environment.shape

    points = np.array(np.unravel_index(np.arange(np.prod(shape)), shape)).T

    # For each actions compute all new grid points (using the environment.move method)
    action_new_states = []
    movements = action_set if not environment.has_layers else action_set[:,1:]
    for move_vector in movements:
        new_points = environment.move(points, movement=move_vector[None,:])
        new_states = np.ravel_multi_index((new_points[:,0], new_points[:,1]), dims=shape)
        action_new_states.append(new_states)

    # Forming it the reachable states array from the new states for each action
    reachable_states = np.array(action_new_states).T[:,:,None]

    # Instantiate the model object
    model = Model(
        states=state_grid,
        actions=agent.action_labels,
        observations=observation_labels,
        reachable_states=reachable_states,
        observation_table=observations,
        end_states=end_states,
        start_probabilities=environment.start_probabilities.ravel(),
        seed=agent.seed
    )
    return model


def minimal_converter(agent : Agent,
                      partitions: list | np.ndarray = [3,6],
                      margin_partitions: bool = False
                      ) -> Model:
    '''
    Method to create a POMDP Model based on an olfactory environment  object.

    This version of the converted, attempts to build a minimal version of the environment with just a few partitions in the x and y direction.
    This means the model will the a total of n states with n = ((|x-partitions| + 2) * (|y-partitions| + 2)).
    The +2 corresponds to two margin cells in the x and y axes.

    It supports an environment in 2D and therefore defines 4 available actions for the agent. (north, east, south, west)
    But, since the model contains so few spaces, the transitions between states are not deterministic:
    This means, if an agent takes a step in a direction, there is a chance the agent stays in the same state along with a lower chance the agent moves to a state in the actual direction it was meaning to go.

    It also defines at least 3 different observations: Nothing, Something or Goal.
    However, if multiple thresholds are provided, the more observations will be available: |threshold| + 1 (Nothing) + 1 (Goal)

    Note: The environment and the threshold parameters are gathered from the agent instance provided.

    Parameters
    ----------
    agent : Agent
        The agent to use to get the environment and threshold parameters from.
    partitions : list or np.ndarray, default=[3,6]
        How many partitions to use in respectively the y and x directions.
    margin_partitions : bool, default=False
        Whether to have seperate partitions for the margins or not. In the case it is enabled, +2 partitions are added in each dimensions.

    Returns
    -------
    model : Model
        A generated POMDP model from the environment.
    '''
    # Agent's parameters
    environment = agent.environment
    thresholds = agent.thresholds
    action_set = agent.action_set

    shape = environment.shape

    # Assertions
    assert not agent.space_aware, "This model is not compatible with space_aware agents as the observations would not match the expected observations of this model."
    assert not (len(agent.thresholds.shape) == 2), "This model doesnt allow for layer specific thresholds."

    # Getting probabilities of odor in the requested partitions and mapping grid to cells
    partitions = np.array(partitions)

    # Building cell bounds
    if margin_partitions:
        cell_shape = (environment.data_shape / partitions).astype(int)
        cell_bounds = [np.array([0, *((np.arange(ax_part+1) * cell_shape[ax_i]) + environment.margins[ax_i, 0]), shape[ax_i]]) for ax_i, ax_part in enumerate(partitions)]
        cell_counts = int(np.prod(partitions + 2)) # Add 2 in each dimensions
    else:
        cell_shape = (shape / partitions).astype(int)
        cell_bounds = [np.array(np.arange(ax_part+1) * cell_shape[ax_i]) for ax_i, ax_part in enumerate(partitions)]
        cell_counts = int(np.prod(partitions))

    lower_bounds = np.array([ax_arr.ravel() for ax_arr in np.meshgrid(*[bounds_arr[:-1] for bounds_arr in cell_bounds], indexing='xy')]).T
    upper_bounds = np.array([ax_arr.ravel() for ax_arr in np.meshgrid(*[bounds_arr[1 :] for bounds_arr in cell_bounds], indexing='xy')]).T

    # Building an array mapping cell ids to the original grid
    grid_cells = np.ones(shape) * -1
    for i, (lower_b, upper_b) in enumerate(zip(lower_bounds, upper_bounds)):
        slices = [slice(ax_lower, ax_upper) for ax_lower, ax_upper in zip(lower_b, upper_b)]

        # Grid to cell mapping
        grid_cells[*slices] = i

    # Building transition probabilities
    points = np.array(np.unravel_index(np.arange(np.prod(shape)), shape)).T
    transition_probabilities = np.full((cell_counts+1, len(action_set), cell_counts+1), -1, dtype=float)

    movements = (action_set if not environment.has_layers else action_set[:,1:])
    for a, move_vector in enumerate(movements):
        new_points = environment.move(points, movement=move_vector[None,:])
        for state_cell in np.arange(cell_counts):
            points_in_cell = (grid_cells[*points.T] == state_cell)[:,None]
            count_in_cell = np.sum(points_in_cell)

            next_cells = np.arange(cell_counts)
            points_in_next_cell = (grid_cells[*new_points.T,None] == next_cells[None,:])

            at_source = environment.source_reached(new_points)[:,None]

            transition_probabilities[state_cell, a, next_cells] = np.sum(((points_in_cell & (~at_source)) & points_in_next_cell), axis=0) / count_in_cell
            transition_probabilities[state_cell, a, -1] = np.sum(points_in_cell & at_source) / count_in_cell

    transition_probabilities[-1,:,:] = 0.0
    transition_probabilities[-1,:,-1] = 1.0

    # Observation labels
    observation_labels = ['nothing']
    if thresholds.shape[-1] > 3:
        for i in range(thresholds.shape[-1] - 2):
            observation_labels.append(f'something_l{i}')
    else:
        observation_labels.append('something')
    observation_labels.append('goal')

    # Observation probabilities
    observations = np.zeros((cell_counts+1, len(action_set), len(observation_labels)))

    # Recomputing bounds for data zone only
    data_cell_bounds = [np.array([*(np.arange(ax_part+1) * cell_shape[ax_i])]) for ax_i, ax_part in enumerate(partitions)]

    data_lower_bounds = np.array([ax_arr.ravel() for ax_arr in np.meshgrid(*[bounds_arr[:-1] for bounds_arr in data_cell_bounds], indexing='xy')]).T
    data_upper_bounds = np.array([ax_arr.ravel() for ax_arr in np.meshgrid(*[bounds_arr[1 :] for bounds_arr in data_cell_bounds], indexing='xy')]).T

    cell_observations = []
    for lower_b, upper_b in zip(data_lower_bounds, data_upper_bounds):
        slices = [slice(ax_lower, ax_upper) for ax_lower, ax_upper in zip(lower_b, upper_b)]

        observations_levels = []
        for min_thresh, max_thresh in zip(thresholds[:-1], thresholds[1:]):
            if environment.has_layers: # TODO: Adapt to different layer thresholding
                odor_within_thresh = (environment.data[:,:,*slices] > min_thresh) & (environment.data[:,:,*slices] < max_thresh)
                observations_levels.append(np.average(odor_within_thresh, axis=tuple([a+1 for a in range(environment.dimensions + 1)])))
            else:
                odor_within_thresh = (environment.data[:,*slices] > min_thresh) & (environment.data[:,*slices] < max_thresh)
                observations_levels.append(np.average(odor_within_thresh))

        cell_observations.append(observations_levels)

    # Placing observation probabilities in observation matrix
    data_cell_ids = np.arange(cell_counts).reshape(partitions+2)[1:-1,1:-1].ravel()
    observations[:-1,:,0] = 1.0 # Nothing at 1 everywhere
    if environment.has_layers:
        action_layers = action_set[:,0]
        actions = np.arange(len(action_layers))
        for i, cell_id in enumerate(data_cell_ids):
            for o in range(len(observation_labels) - 1):
                observations[cell_id,actions,o] = cell_observations[i][o][action_layers]
    else:
        for i, cell_id in enumerate(data_cell_ids):
            observations[cell_id,:,:-1] = cell_observations[i]

    observations[-1,:,-1] = 1.0 # Goal

    # Start probabilities # TODO Match data zone
    start_probabilities = np.ones(cell_counts+1, dtype=float)
    # start_probabilities = (odor_probabilities > 0).astype(float).flatten()
    start_probabilities /= np.sum(start_probabilities)

    # Creation of the Model
    model = Model(
        states = [f'cell_{cell}' for cell in range(cell_counts)] + ['goal'],
        actions = agent.action_labels,
        observations = observation_labels,
        transitions = transition_probabilities,
        observation_table = observations,
        end_states = [cell_counts], # The very last state is the goal state
        start_probabilities = start_probabilities,
        seed=agent.seed
    )

    return model