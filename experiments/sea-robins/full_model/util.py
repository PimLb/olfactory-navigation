import sys
sys.path.append('../../..')

from olfactory_navigation.simulation import SimulationHistory
from viz import plot_trajectory_in_tank

from matplotlib import pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from PIL import Image

import os
import numpy as np
import re


def generate_results_plots(test_result_folder:str, all_hist:SimulationHistory, params:dict, exp_results:dict) -> None:
    tank_size = params['tank_size']
    step_dt = params['step_dt']
    step_dist = params['step_dist']

    new_order_exp_agent_indices = exp_results['new_order_exp_agent_indices']
    new_order_exp_source_indices = exp_results['new_order_exp_source_indices']
    out_of_bounds_amount = exp_results['out_of_bounds_amount']

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
    runs_df = all_hist.runs_analysis_df
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


def generate_results_pdf(folder:str) -> None:
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


def generate_trajectories_pdf(folder:str) -> None:
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