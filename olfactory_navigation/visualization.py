import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from typing import Literal


def plot_shape_robustness_performance(robustness_analysis_df: pd.DataFrame,
                                      metric: Literal['converged', 'steps_taken', 'discounted_rewards', 'extra_steps', 't_min_over_t'] = 'converged',
                                      success_only: bool = False,
                                      ax: plt.Axes = None
                                      ) -> None:
    '''
    Function to visualize the performance according to some metric for each multiplier on the horizontal and vertical axes.
    The metrics that can be chosen are among 'converged', 'steps_taken', 'discounted_rewards', 'extra_steps', or 't_min_over_t'.

    Parameters
    ----------
    robustness_analysis_df : pd.DataFrame
        The analysis dataframe from which to plot the comparison plot.
    metric : 'converged' or 'steps_taken' or 'discounted_rewards' or 'extra_steps' or 't_min_over_t', default = 'converged'
        The metric to be used to compare the performance between different multiplier combinations.
    success_only : bool, default=False
        Whether to use the results filtered only to the successful trajectories.
    ax : plt.Axes, optional
        A matplotlib axis on which to plot the comparison plot. If not provided, a new one will be created.
    '''
    if ax is None:
        _, ax = plt.subplots()

    assert not ((metric == 'converged') and success_only), "The 'converged' metric is not available for the 'success_only' option."

    # Invert is lower is better
    inverted = metric in ['steps_taken', 'extra_steps']

    # If success only
    metric += ('_success' if success_only else '')

    # Processing data
    mult_lists = {}
    for col in ['y','x']:
        mult_lists[col] = np.unique(robustness_analysis_df[col + '_multiplier'].to_numpy(dtype=int)).tolist()

    grid_data = robustness_analysis_df[metric + '_mean'].to_numpy().reshape([len(mults) for mults in mult_lists.values()])

    # Plotting data
    im = ax.imshow(grid_data, cmap='RdYlGn' + ('_r' if inverted else ''))
    cbar = ax.get_figure().colorbar(im)
    cbar.set_label((' '.join(metric.split('_'))).capitalize() + ' Mean')

    # Processing standard deviations
    std_data = robustness_analysis_df[metric + '_standard_deviation'].to_numpy()
    min_std, max_std = std_data.min(), std_data.max()
    normalized_std_data = (99 * (std_data - min_std) / (max_std - min_std)) + 1
    X,Y = np.meshgrid(np.arange(len(mult_lists['x'])), np.arange(len(mult_lists['y'])))
    scatter = ax.scatter(X, Y, s=normalized_std_data, c='black')

    dots = [1,100]
    dot_labels = [f'{min_std:.2f}', f'{max_std:.2f}']

    legend_handles,_ = scatter.legend_elements(prop='sizes', num=dots)
    ax.legend(handles=legend_handles, labels=dot_labels, title='Standard\ndeviations', loc='upper right', bbox_to_anchor=(1.6,1.0))

    # Adding axis labels
    ax.set_xlabel('Horizontal multiplier')
    ax.set_xticks(np.arange(len(mult_lists['x'])), labels=[f'{mult}%' for mult in mult_lists['x']])

    ax.set_ylabel('Vertical multiplier')
    ax.set_yticks(np.arange(len(mult_lists['y'])), labels=[f'{mult}%' for mult in mult_lists['y']])


def plot_full_shape_robustness_analysis(robustness_analysis_df: pd.DataFrame) -> None:
    '''
    Function to build a combined plot of all the metrics with and without the success trajectory filtering.
    The metrics used are 'converged', 'steps_taken', 'discounted_rewards', 'extra_steps', and 't_min_over_t'

    Parameters
    ----------
    robustness_analysis_df : pd.DataFrame
        The pandas dataframe to plot the full analysis on.
    '''
    _, axes = plt.subplots(5,2, figsize=(15,30))

    # For each metric on each row and columns being success filtering off then on.
    for row, metric in enumerate(['converged', 'steps_taken', 'discounted_rewards', 'extra_steps', 't_min_over_t']):
        for col, success_only in enumerate([False, True]):
            if (metric == 'converged') and (success_only):
                axes[row, col].set_axis_off()
                continue

            plot_shape_robustness_performance(robustness_analysis_df, metric=metric, success_only=success_only, ax=axes[row, col])
