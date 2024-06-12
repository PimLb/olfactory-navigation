from olfactory_navigation import Environment
from olfactory_navigation.agents.model_based_util.pomdp import Model

import numpy as np


def exact_converter(environment: Environment,
                    threshold : float | list
                    ) -> Model:
    '''
    Method to create a POMDP model based on an olfactory environment object.

    This version of the converter converts the environment in an exact manner.
    This mean the amount of states is equal to the grid points in the olfactory environment object.

    It supports an environment in 2D and therefore defines 4 available actions for the agent.

    It also defines at least different observations: Nothing, Something or Goal.
    However, if multiple thresholds are provided, the more observations will be available: |threshold| + 1 (Nothing) + 1 (Goal)

    Parameters
    ----------
    environment : Environment
        The olfactory environment object to create the POMDP model from.
    threshold : float or list
        A threshold for the odor cues.
        If a single is provided, the agent will smell something when an odor is above the threshold and nothing when it is bellow.
        If a list is provided, the agent will able to distinguish different levels of smell.

    Returns
    -------
    model : Model
        A generate POMDP model from the environment.
    '''
    # TODO: Implement different action sets for the POMDP Model Converter

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
    data_grid = environment.data[:,:,:,None]
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
    row_w = environment.shape[1]

    reachable_states = np.zeros((state_count, 4, 1), dtype=int)
    for s in range(state_count):
        reachable_states[s,0,0] = s - row_w if s - row_w >= 0 else (state_count - row_w) + s # North
        reachable_states[s,1,0] = s + 1 if (s + 1) % row_w > 0 else s # East
        reachable_states[s,2,0] = s + row_w if s + row_w < state_count else s % row_w # South
        reachable_states[s,3,0] = s - 1 if (s - 1) % row_w < (row_w - 1) else s # West

    reachable_states = np.array(reachable_states)

    # Instantiate the model object
    model = Model(
        states=state_grid,
        actions=['N','E','S','W'],
        observations=observation_labels,
        reachable_states=reachable_states,
        observation_table=observations,
        end_states=end_states,
        start_probabilities=environment.start_probabilities.ravel()
    )
    return model