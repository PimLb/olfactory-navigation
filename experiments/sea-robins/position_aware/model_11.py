import numpy as np

from olfactory_navigation.agents.model_based_util.pomdp import Model

from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix


def generate_model(
        space_shape : np.ndarray,
        cells: np.ndarray,
        source_radius: int,
        source_cell_resolution: np.ndarray,
        data_mean: list,
        data_covariance: int
        ) -> Model:
    space_center = (space_shape / 2).astype(int)

    # Building probability map
    x,y = np.meshgrid(np.arange(space_shape[0]), np.arange(space_shape[1]))
    pos = np.dstack((x, y))
    rv = multivariate_normal(data_mean, data_covariance)
    probability_map = rv.pdf(pos).T

    # Normalize to have 1 at the center
    probability_map /= np.max(probability_map)

    # Building a grid-cell-mapping
    def build_grid_mapping(space_shape: np.ndarray,
                           cells: np.ndarray,
                           source_position: np.ndarray,
                           source_radius: int,
                           source_cell_resolution: np.ndarray
                           ) -> np.ndarray:
        # Finding the sizes of the cells
        cell_size_standard = (space_shape / cells).astype(int)
        cell_size_overflow = (space_shape % cells).astype(int)

        # Determining cell sizes
        cell_sizes = []
        for ax_cells, ax_size, ax_overflow in zip(cells, cell_size_standard, cell_size_overflow):
            sizes = np.repeat(ax_size, ax_cells)
            if ax_overflow > 0:
                sizes[:int(np.floor(ax_overflow/2))] += 1
                sizes[-int(np.ceil(ax_overflow/2)):] += 1
            cell_sizes.append(sizes)

        # Finding the edges of the cells and filling a grid with ids
        cell_edges = [np.concatenate(([0], np.cumsum(ax_sizes))) for ax_sizes in cell_sizes]

        lower_bounds = np.array([ax_arr.ravel() for ax_arr in np.meshgrid(*[bounds_arr[:-1] for bounds_arr in cell_edges], indexing='ij')]).T
        upper_bounds = np.array([ax_arr.ravel() for ax_arr in np.meshgrid(*[bounds_arr[1 :] for bounds_arr in cell_edges], indexing='ij')]).T

        grid_cells = np.full(space_shape, -1)
        for i, (lower_b, upper_b) in enumerate(zip(lower_bounds, upper_bounds)):
            slices = [slice(ax_lower, ax_upper) for ax_lower, ax_upper in zip(lower_b, upper_b)]

            # Grid to cell mapping
            grid_cells[*slices] = i

        # Finding cell the replace
        source_cell_id = grid_cells[*source_position]
        source_cell = np.array(np.unravel_index(source_cell_id, cells))

        # Finding center and replacing with closest side
        source_cell_shape = np.array([sizes[int(ax_id)] for sizes, ax_id in zip(cell_sizes, source_cell)])

        # Splitting the center cell grid into subcells
        sub_cell_size_standard = (source_cell_shape / source_cell_resolution).astype(int)
        sub_cell_size_overflow = (source_cell_shape % source_cell_resolution).astype(int)

        sub_cell_sizes = []
        for ax_cells, ax_size, ax_overflow in zip(source_cell_resolution, sub_cell_size_standard, sub_cell_size_overflow):
            sizes = np.repeat(ax_size, ax_cells)
            if ax_overflow > 0:
                sizes[:int(np.floor(ax_overflow/2))] += 1
                sizes[-int(np.ceil(ax_overflow/2)):] += 1
            sub_cell_sizes.append(sizes)

        # Finding the edges of the cells and filling a grid with ids
        sub_cell_edges = [np.concatenate(([0], np.cumsum(ax_sizes))) for ax_sizes in sub_cell_sizes]

        sub_cell_lower_bounds = np.array([ax_arr.ravel() for ax_arr in np.meshgrid(*[bounds_arr[:-1] for bounds_arr in sub_cell_edges], indexing='ij')]).T
        sub_cell_upper_bounds = np.array([ax_arr.ravel() for ax_arr in np.meshgrid(*[bounds_arr[1 :] for bounds_arr in sub_cell_edges], indexing='ij')]).T

        sub_grid_cells = np.full(source_cell_shape, -1)
        for i, (lower_b, upper_b) in enumerate(zip(sub_cell_lower_bounds, sub_cell_upper_bounds)):
            slices = [slice(ax_lower, ax_upper) for ax_lower, ax_upper in zip(lower_b, upper_b)]

            # Grid to cell mapping
            sub_grid_cells[*slices] = i + np.prod(cells)

        # Injecting the sub_grid in the grid
        grid_cells[*[slice(lower_b, upper_b) for lower_b, upper_b in zip(lower_bounds[source_cell_id], upper_bounds[source_cell_id])]] = sub_grid_cells

        # Spacial indices
        spacial_indices = list(np.ndindex(tuple(space_shape)))
        spacial_indices_array = np.array(spacial_indices).astype(int)

        # Indices are at source
        at_source = np.sum((source_position[None,:] - spacial_indices_array) ** 2, axis=1) <= source_radius ** 2
        grid_cells[*spacial_indices_array[at_source].T] = -1

        return grid_cells

    # Generating state labels
    # Get the list of all indices
    indices_list = list(np.ndindex((*cells, *cells)))

    cell_indices = list(np.ndindex((*cells,)))
    source_cell_indices = list(np.ndindex((*source_cell_resolution,)))

    state_labels = []
    for (s_y, s_x) in [(cells / 2).astype(int)]:
        for (a_y, a_x) in cell_indices:
            state_labels.append(f's_{s_y}_{s_x}-a_{a_y}_{a_x}')
        state_labels += [f's_{s_y}_{s_x}-sr_{source_y}_{source_x}' for (source_y, source_x) in source_cell_indices]
    state_labels.append('goal')

    # Generating action labels
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

    # Spacial indices
    spacial_indices = list(np.ndindex(tuple(space_shape)))
    spacial_indices_array = np.array(spacial_indices).astype(int)

    # Set the transition probability matrix
    cell_count = np.prod(cells) + np.prod(source_cell_resolution) # ! Plus the sub cells zones of the source cell
    state_count = len(state_labels)
    action_count = len(action_set)
    transition_probabilities = np.zeros((state_count, action_count, state_count))

    for source_i, source_pos in enumerate([space_center]):
        grid_cells = build_grid_mapping(space_shape=space_shape,
                                        cells=cells,
                                        source_position=source_pos,
                                        source_radius=source_radius,
                                        source_cell_resolution=source_cell_resolution)
        cells_from_indices = grid_cells[*spacial_indices_array.T]

        move_probabilities = []
        for move in action_set:
            # Applying the moves to the indices
            moved_indices = spacial_indices_array + move
            moved_indices_clipped = np.clip(moved_indices, 0, space_shape-1)

            # Converting moved indices to cell indices
            cells_from_indices_after_move = grid_cells[*moved_indices_clipped.T]

            # Building a confusion matrix of the moves
            conf = confusion_matrix(cells_from_indices, cells_from_indices_after_move, labels=[-1] + list(np.arange(cell_count)), normalize='true')
            move_probabilities.append(conf[:,None,:])

        # Stack probabilities
        move_probabilities = np.hstack(move_probabilities)

        # In case a cell is not used, make the transition loop on itself
        zero_move_probabilities = np.argwhere(move_probabilities.sum(axis=-1) == 0).T
        move_probabilities[zero_move_probabilities[0], zero_move_probabilities[1], zero_move_probabilities[0]] = 1.0

        # Set the values in the transition matrix
        source_slice = slice((source_i * cell_count), ((source_i + 1) * cell_count))
        transition_probabilities[source_slice, :, source_slice] = move_probabilities[1:,:,1:]
        transition_probabilities[source_slice, :, -1] = move_probabilities[1:,:,0]
        transition_probabilities[-1, :, -1] = 1.0

    assert np.all(transition_probabilities.sum(axis=-1).round(8) == 1.0)

    # Observations
    threshold = 3e-6

    # Compute observation matrix
    if not isinstance(threshold, list):
        threshold = [threshold]

    # Ensure 0.0 and 1.0 begin and end the threshold list
    if threshold[0] != -np.inf:
        threshold = [-np.inf] + threshold

    if threshold[-1] != np.inf:
        threshold = threshold + [np.inf]

    #  Observation labels
    observation_labels = ['nothing']
    if len(threshold) > 3:
        for i,_ in enumerate(threshold[1:-1]):
            observation_labels.append(f'something_l{i}')
    else:
        observation_labels.append('something')

    # Add goal
    observation_labels.append('goal')

    # Observation probabilities
    # Finding the sizes of the cells
    cell_size_standard = (space_shape / cells).astype(int)
    cell_size_overflow = (space_shape % cells).astype(int)

    # Determining cell sizes
    cell_sizes = []
    for ax_cells, ax_size, ax_overflow in zip(cells, cell_size_standard, cell_size_overflow):
        sizes = np.repeat(ax_size, ax_cells)
        if ax_overflow > 0:
            sizes[:int(np.floor(ax_overflow/2))] += 1
            sizes[-int(np.ceil(ax_overflow/2)):] += 1
        cell_sizes.append(sizes)

    # Finding the edges of the cells and filling a grid with ids
    cell_edges = [np.concatenate(([0], np.cumsum(ax_sizes))) for ax_sizes in cell_sizes]

    lower_bounds = np.array([ax_arr.ravel() for ax_arr in np.meshgrid(*[bounds_arr[:-1] for bounds_arr in cell_edges], indexing='ij')]).T
    upper_bounds = np.array([ax_arr.ravel() for ax_arr in np.meshgrid(*[bounds_arr[1 :] for bounds_arr in cell_edges], indexing='ij')]).T


    cell_probabilities = np.full(cells, -1, dtype=float)
    for cell, lower_b, upper_b in zip(list(np.ndindex(tuple(cells))), lower_bounds, upper_bounds):
        slices = [slice(ax_lower, ax_upper) for ax_lower, ax_upper in zip(lower_b, upper_b)]

        # TODO: Make it flexible to multi thresh
        cell_probabilities[*cell] = np.average(probability_map[*slices].astype(float))

    source_cell_shape = np.array(cell_size_standard)

    # Splitting the center cell grid into subcells
    sub_cell_size_standard = (source_cell_shape / source_cell_resolution).astype(int)
    sub_cell_size_overflow = (source_cell_shape % source_cell_resolution).astype(int)

    sub_cell_sizes = []
    for ax_cells, ax_size, ax_overflow in zip(source_cell_resolution, sub_cell_size_standard, sub_cell_size_overflow):
        sizes = np.repeat(ax_size, ax_cells)
        if ax_overflow > 0:
            sizes[:int(np.floor(ax_overflow/2))] += 1
            sizes[-int(np.ceil(ax_overflow/2)):] += 1
        sub_cell_sizes.append(sizes)

    # Finding the edges of the cells and filling a grid with ids
    sub_cell_edges = [np.concatenate(([0], np.cumsum(ax_sizes))) for ax_sizes in sub_cell_sizes]

    sub_cell_lower_bounds = np.array([ax_arr.ravel() for ax_arr in np.meshgrid(*[bounds_arr[:-1] for bounds_arr in sub_cell_edges], indexing='ij')]).T
    sub_cell_upper_bounds = np.array([ax_arr.ravel() for ax_arr in np.meshgrid(*[bounds_arr[1 :] for bounds_arr in sub_cell_edges], indexing='ij')]).T

    # Finding the center cell and its lower bound
    center_cell = np.ravel_multi_index(np.floor(cells/2).astype(int), cells)
    lower_bound_center_cell = lower_bounds[center_cell]

    sub_grid_cell_probabilities = np.full(source_cell_resolution, -1, dtype=float)
    for cell, lower_b, upper_b in zip(list(np.ndindex(tuple(source_cell_resolution))), sub_cell_lower_bounds, sub_cell_upper_bounds):
        lower_b += lower_bound_center_cell
        upper_b += lower_bound_center_cell
        slices = [slice(ax_lower, ax_upper) for ax_lower, ax_upper in zip(lower_b, upper_b)]

        sub_grid_cell_probabilities[*cell] = np.average(probability_map[*slices].astype(float))

    # Indices array
    indices_array = np.array(indices_list)
    relative_indices = indices_array[:,:len(cells)] - indices_array[:,len(cells):]

    # Shifting the relative positions by the position of the center
    center_pos = (cells / 2).astype(int)
    centered_positions = relative_indices + center_pos

    # Filtering to the valid centered positions
    valid_positions = np.all((centered_positions >= 0) & (centered_positions < cells), axis=1)
    filtered_centered_positions = centered_positions[valid_positions]
    filtered_indices_array = indices_array[valid_positions]

    # Gathering the cell_probabilities
    odor_probabilities = np.zeros((*cells, *cells), dtype=float)
    odor_probabilities[*filtered_indices_array.T] = cell_probabilities[*filtered_centered_positions.T]

    # Generating an empty observation matrix
    observations = np.zeros((len(state_labels), len(action_set), len(observation_labels)))

    # Filling the something/nothing observations
    agent_cells = indices_array[:np.prod(cells),2:]
    sub_cell_cues = sub_grid_cell_probabilities[*np.array(list(np.ndindex(tuple(source_cell_resolution)))).T]
    for obs_i, agent_cell in enumerate([(cells / 2).astype(int)]):
        # For observation matrix
        agent_ids_in_states = (np.arange(np.prod(cells))) + obs_i

        # To read odor probability matrix
        agent_ids_in_space = np.all(indices_array[:,2:] == agent_cell, axis=1)
        filtered_agent_ids_in_space = indices_array[agent_ids_in_space]

        # Setting the probabilities in the matrix
        probabilities_at_agent_position = odor_probabilities[*filtered_agent_ids_in_space.T][:,None]
        observations[agent_ids_in_states,:,obs_i] = 1 - probabilities_at_agent_position # Nothing
        observations[agent_ids_in_states,:,obs_i+1] = probabilities_at_agent_position # Something

        # Finding sub-cells of the source cell
        sub_cell_source_ids = ((obs_i * cell_count) + np.prod(cells)) + np.arange(np.prod(source_cell_resolution))
        observations[sub_cell_source_ids,:,obs_i] = 1 - sub_cell_cues[:,None] # Nothing
        observations[sub_cell_source_ids,:,obs_i+1] = sub_cell_cues[:,None] # Something

        # Finding the cells where the agent is at the source
        agent_at_source_id = (obs_i * cell_count) + obs_i
        observations[agent_at_source_id,:,obs_i] = 1.0 # Nothing
        observations[agent_at_source_id,:,obs_i+1] = 0.0 # Something

    # Goal observations
    observations[-1,:,0] = 0.0
    observations[:,:,-1] = 0.0
    observations[-1,:,-1] = 1.0 # Goal

    assert np.all(observations.sum(axis=-1).round(8) == 1.0)

    # Building model
    model = Model(
        states = state_labels,
        actions = action_labels,
        observations = observation_labels,
        transitions = transition_probabilities,
        observation_table = observations,
        end_states = [len(state_labels)-1], # The very last state is the goal state
        # start_probabilities = start_probabilities,
        seed=12131415
    )

    return model