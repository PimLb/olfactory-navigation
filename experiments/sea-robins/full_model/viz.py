import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.patches import Circle

from olfactory_navigation.simulation import SimulationHistory


def plot_trajectory_in_tank(h:SimulationHistory,
                            exp_agent:np.ndarray,
                            exp_source:np.ndarray,
                            t_size=np.ndarray,
                            traj:int=0,
                            view_margin:int=4,
                            ax=None):
    # Generate ax is not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(18,3))

    # Retrieving sim
    sim = h.simulation_dfs[traj]

    # Plot setup
    ax.imshow(np.zeros(tuple(t_size)), cmap='Greys', zorder=-100)

    # Start
    start_coord = sim[['x', 'y']].to_numpy()[0]

    # Compute shift
    shift = start_coord - exp_agent[traj][::-1]

    # Plot start coordinate
    start_coord -= shift
    ax.scatter(start_coord[0], start_coord[1], c='green', label='Start')

    # Source circle
    goal_circle = Circle(exp_source[traj][::-1], h.environment_source_radius, color='r', fill=False, label='Source')
    ax.add_patch(goal_circle)

    # Until step
    seq = sim[['x','y']].to_numpy() - shift[None,:]

    # Path
    ax.plot(seq[:,0], seq[:,1], zorder=-1, c='black', label='Path')

    # Layer observations
    if h.environment_layer_labels is not None:
        obs_layer = sim[['layer']][1:].to_numpy()
        layer_colors = np.array(list(colors.TABLEAU_COLORS.values()))

        for layer_i, layer_label in enumerate(h.environment_layer_labels[1:]):
            layer_i += 1
            layer_mask = (obs_layer == layer_i)[:,0] # Reshaping to a single vector and not an n by 1 array
            ax.scatter(seq[1:][layer_mask,0], seq[1:][layer_mask,1], # X, Y
                        marker='x',
                        color=layer_colors[(layer_i-1) % len(layer_colors)], # Looping over the colors in case there are more layers than colors
                        zorder=2,
                        label=layer_label)

    # Process odor cues
    odor_cues = sim['o'][1:].to_numpy()
    observation_ids = None
    if h.environment.has_layers and len(h.agent_thresholds.shape) == 2:
        layer_ids = sim[['layer']][1:].to_numpy()
        action_layer_thresholds = h.agent_thresholds[layer_ids]
        observation_ids = np.argwhere((odor_cues[:,None] >= action_layer_thresholds[:,:-1]) & (odor_cues[:,None] < action_layer_thresholds[:,1:]))[:,1]
    else:
        # Setting observation ids
        observation_ids = np.argwhere((odor_cues[:,None] >= h.agent_thresholds[:-1][None,:]) & (odor_cues[:,None] < h.agent_thresholds[1:][None,:]))[:,1]

    # Check whether the odor detection is binary or by level
    odor_bins = h.agent_thresholds.shape[-1] - 1
    if odor_bins > 2:
        odor_levels = np.arange(odor_bins - 1) + 1
        for level in odor_levels:
            cues_at_level = (observation_ids == level)
            ax.scatter(seq[1:][cues_at_level,0], seq[1:][cues_at_level,1],
                        zorder=1,
                        alpha=(level / odor_bins),
                        label=f'Sensed level {level}')
    else:
        something_sensed = (observation_ids == 1)
        ax.scatter(seq[1:][something_sensed,0], seq[1:][something_sensed,1],
                    zorder=1,
                    label='Something observed')

    # Generate legend
    ax.legend()

    # Plot tank boundaries
    ax.hlines([-0.3,t_size[0]+0.3], xmin=-0.3, xmax=t_size[1]+0.3, linestyles='dashed')
    ax.vlines([-0.3,t_size[1]+0.3], ymin=-0.3, ymax=t_size[0]+0.3, linestyles='dashed')

    # Limit view
    ax.set_xlim(-view_margin, t_size[1]+view_margin)
    ax.set_ylim(t_size[0]+view_margin, -view_margin)