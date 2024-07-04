from olfactory_navigation import Agent, Environment
from olfactory_navigation.agents.model_based_util.pomdp import Model

import numpy as np


def exact_converter(agent : Agent) -> Model:
    '''
    Method to create a POMDP model based on an olfactory environment object.

    This version of the converter converts the environment in an exact manner.
    This mean the amount of states is equal to the grid points in the olfactory environment object.

    It supports an environment in 2D and therefore defines 4 available actions for the agent. (north, east, south, west)

    It also defines at least different observations: Nothing, Something or Goal.
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
    threshold = agent.threshold
    action_set = agent.action_set

    # Base Model parameters
    state_count = np.prod(environment.shape)

    state_grid = [[f's_{x}_{y}' for x in range(environment.shape[1])] for y in range(environment.shape[0])]
    end_states = np.argwhere(np.fromfunction(lambda x,y: ((x-environment.source_position[0])**2 + (y-environment.source_position[1])**2) <= environment.source_radius**2,
                                                shape=environment.shape).ravel())[:,0].tolist()

    # Compute observation matrix
    if not isinstance(threshold, list):
        threshold = [threshold]

    # Ensure 0.0 and 1.0 begin and end the threshold list
    if threshold[0] != -np.inf:
        threshold = [-np.inf] + threshold

    if threshold[-1] != np.inf:
        threshold = threshold + [np.inf]

    # Computing odor probabilities
    data_grid = environment.data[0,:,:,:,None] if environment.has_layers else environment.data[:,:,:,None]
    threshs = np.array(threshold)
    data_odor_fields = np.average(((data_grid >= threshs[:-1][None,None,None,:]) & (data_grid < threshs[1:][None,None,None,:])), axis=0)

    # Increasing it to the full environment
    odor_fields = np.zeros(environment.shape + ((len(threshold)-1),))
    odor_fields[environment.data_bounds[0,0]:environment.data_bounds[0,1], environment.data_bounds[1,0]:environment.data_bounds[1,1], :] = data_odor_fields

    # Building observation matrix
    observations = np.empty((state_count, 4, len(threshold)), dtype=float) # 4-actions, observations: |thresholds|-1 + goal 

    for i in range(len(threshold)-1):
        observations[:,:,i] = odor_fields[:,:,i].ravel()[:,None]

    # Setting 'Nothing' observation in the margins to 1
    data_margins_mask = np.ones(environment.shape, dtype=bool)
    data_margins_mask[environment.data_bounds[0,0]:environment.data_bounds[0,1], environment.data_bounds[1,0]:environment.data_bounds[1,1]] = False
    observations[data_margins_mask.ravel(),:,0] = 1.0

    # Goal observation
    observations[:,:,-1] = 0.0
    observations[end_states,:,:] = 0.0
    observations[end_states,:,-1] = 1.0

    # Assert observations sum to 1
    assert np.all(np.sum(observations, axis=2) == 1.0), "Observation table malformed, something is wrong..."

    # Observation labels
    observation_labels = ['nothing']
    if len(threshold) > 3:
        for i,_ in enumerate(threshold[1:-1]):
            observation_labels.append(f'something_l{i}')
    else:
        observation_labels.append('something')
    observation_labels.append('goal')

    # Compute reachable states
    shape = environment.shape

    points = np.array(np.unravel_index(np.arange(np.prod(shape)), shape)).T

    # For each actions compute all new grid points (using the environment.move method)
    action_new_states = []
    for action in action_set:
        new_points = environment.move(points, movement=action[None,:])
        new_states = np.ravel_multi_index((new_points[:,0], new_points[:,1]), dims=shape)
        action_new_states.append(new_states)

    # Forming it the reachable states array from the new states for each action
    reachable_states = np.array(action_new_states).T[:,:,None]

    # Action labels
    action_labels = [f'a_{i}' for i in range(len(action_set))] # TODO: Allow action set to be a dict with labels

    # Instantiate the model object
    model = Model(
        states=state_grid,
        actions=action_labels,
        observations=observation_labels,
        reachable_states=reachable_states,
        observation_table=observations,
        end_states=end_states,
        start_probabilities=environment.start_probabilities.ravel()
    )
    return model


def minimal_converter(agent : Agent,
                      partitions: list | np.ndarray = [3,6],
                      partition_move_out_probabilities: int | list | np.ndarray | None = None
                      ) -> Model:
    '''
    Method to create a POMDP Model based on an olfactory environment  object.

    This version of the converted, attempts to build a minimal version of the environment with just a few partitions in the x and y direction.
    This means the model will the a total of n states with n = ((|x-partitions| + 2) * (|y-partitions| + 2)).
    The +2 corresponds to two margin cells in the x and y axes.

    It supports an environment in 2D and therefore defines 4 available actions for the agent. (north, east, south, west)
    But, since the model contains so few spaces, the transitions between states are not deterministic:
    This means, if an agent takes a step in a direction, there is a chance the agent stays in the same state along with a lower chance the agent moves to a state in the actual direction it was meaning to go.

    It also defines at least different observations: Nothing, Something or Goal.
    However, if multiple thresholds are provided, the more observations will be available: |threshold| + 1 (Nothing) + 1 (Goal)

    Note: The environment and the threshold parameters are gathered from the agent instance provided.

    Parameters
    ----------
    agent : Agent
        The agent to use to get the environment and threshold parameters from.
    partitions : list or np.ndarray, default=[3,6]
        How many partitions to use in respectively the y and x directions.
    partition_move_out_probabilities : int or list or np.ndarray, optional
        A unique 'move_out' probability, or a list/array of the probabilities for respectively the horizontal and vertical movements.
        If none is provided, the probabilities will be 1 over the cell shape in the y and x directions for the probabilities horizontally and vertically.

    Returns
    -------
    model : Model
        A generated POMDP model from the environment.
    '''
    # Agent's parameters
    environment = agent.environment
    threshold = agent.threshold
    action_set = agent.action_set

    # Getting probabilities of odor in the requested partitions
    partitions = np.array(partitions)
    y_partitions = partitions[0]
    x_partitions = partitions[1]

    cell_shape = (environment.data_shape / np.array([y_partitions, x_partitions])).astype(int)
    cell_odor_probs = np.zeros((y_partitions, x_partitions))

    for y_i in range(y_partitions):
        for x_i in range(x_partitions):
            if environment.has_layers:
                cell = environment.data[0,
                                        :,
                                        (y_i * cell_shape[0]) : ((y_i + 1) * cell_shape[0]),
                                        (x_i * cell_shape[1]) : ((x_i + 1) * cell_shape[1])]
            else:
                cell = environment.data[:,
                                        (y_i * cell_shape[0]) : ((y_i + 1) * cell_shape[0]),
                                        (x_i * cell_shape[1]) : ((x_i + 1) * cell_shape[1])]

            cell_odor_probs[y_i, x_i] = np.average(cell > threshold)

    odor_probabilities = np.zeros(((y_partitions + 2), (x_partitions + 3)))
    odor_probabilities[1:-1,2:-1] = cell_odor_probs

    # General attributes
    state_count = np.prod(odor_probabilities.shape)
    shape = odor_probabilities.shape

    # State grid
    state_grid = [[f's_{x}_{y}' for x in range(shape[1])] for y in range(shape[0])]

    # Computing move out probabilities
    if partition_move_out_probabilities is None:
        move_out_prob = 1 / cell_shape
    elif isinstance(partition_move_out_probabilities, int):
        move_out_prob = np.ones(2) * partition_move_out_probabilities
    else:
        move_out_prob = np.array(partition_move_out_probabilities)

    # Transition probabilities of the model
    width = shape[1]
    transition_probabilities = np.zeros((state_count, 4, state_count))

    # TODO: Replace this by using the action set
    for y in range(shape[0]):
        for x in range(shape[1]):

            # North
            if y == 0:
                transition_probabilities[(y * width) + x, 0, (y * width) + x] = 1.0
            else:
                transition_probabilities[(y * width) + x, 0, (y * width) + x] = (1 - move_out_prob[0])
                transition_probabilities[(y * width) + x, 0, ((y - 1) * width) + x] = move_out_prob[0]

            # East
            if x == (shape[1] - 1):
                transition_probabilities[(y * width) + x, 1, (y * width) + x] = 1.0
            else:
                transition_probabilities[(y * width) + x, 1, (y * width) + x] = (1 - move_out_prob[1])
                transition_probabilities[(y * width) + x, 1, (y * width) + (x + 1)] = move_out_prob[1]
            
            # South
            if y == (shape[0] - 1):
                transition_probabilities[(y * width) + x, 2, (y * width) + x] = 1.0
            else:
                transition_probabilities[(y * width) + x, 2, (y * width) + x] = (1 - move_out_prob[1])
                transition_probabilities[(y * width) + x, 2, ((y + 1) * width) + x] = move_out_prob[1]

            # West
            if x == 0:
                transition_probabilities[(y * width) + x, 3, (y * width) + x] = 1.0
            else:
                transition_probabilities[(y * width) + x, 3, (y * width) + x] = (1 - move_out_prob[0])
                transition_probabilities[(y * width) + x, 3, (y * width) + (x - 1)] = move_out_prob[0]

    # Goal states
    end_states = [int((int(shape[0] / 2) * width) + 1)]

    # Observations
    observation_labels = ['nothing', 'something', 'goal']

    observations = np.zeros((state_count, 4, len(observation_labels)))

    observations[:, :, 0] = (1 - odor_probabilities.flatten())[:,None]
    observations[:, :, 1] = odor_probabilities.flatten()[:,None]

    observations[end_states, :, -1] = 0.0
    observations[end_states, :, :] = 0.0
    observations[end_states, :, -1] = 1.0

    # Start probabilities
    start_probabilities = np.ones(state_count, dtype=float)
    # start_probabilities = (odor_probabilities > 0).astype(float).flatten()
    start_probabilities /= np.sum(start_probabilities)

    # Creation of the Model
    model = Model(
        states = state_grid,
        actions = ['N','E','S','W'],
        observations = observation_labels,
        transitions = transition_probabilities,
        observation_table = observations,
        end_states = end_states,
        start_probabilities = start_probabilities
    )

    return model