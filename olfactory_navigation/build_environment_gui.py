import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog


def buildWindow():
    # GLOBAL VARIABLES
    bold_font = 'Helvetica 12 bold'
    frame_padding = 10

    entry_fields = {}
    linked_fields = {
        'margin_all': ['discretization_y', 'discretization_y', 'discretization_x', 'discretization_x', 'margin_up', 'margin_down', 'margin_left', 'margin_right', 'margin_ver', 'margin_hor'],
        'margin_ver': ['discretization_y', 'discretization_y', 'margin_up', 'margin_down'],
        'margin_hor': ['discretization_x', 'discretization_x', 'margin_left', 'margin_right'],
        'margin_up': ['discretization_y', 'margin_ver', 'margin_all'],
        'margin_down': ['discretization_y', 'margin_ver', 'margin_all'],
        'margin_left': ['discretization_x', 'margin_hor', 'margin_all'],
        'margin_right': ['discretization_x', 'margin_hor', 'margin_all']
    }

    class EnvironmentConfig:
        def __init__(self):
            self._data = None
            self.data_frame = None
            self.config = {}
            self.mpl_frame = None
            self.canvas = None
        
        @property
        def data(self):
            return self._data
        
        @data.setter
        def data(self, dat):
            self._data = dat
            self.data_frame = self._data[0]

            # Discretization
            self.config['discretization_y'] = str(self.data_frame.shape[0])
            self.config['discretization_x'] = str(self.data_frame.shape[1])

            # Source
            self.config['data_source_y'] = "0"
            self.config['data_source_x'] = "0"
            self.config['source_radius'] = "1"

            # Margins
            self.config['margin_all'] = "0"
            self.config['margin_ver'] = "0"
            self.config['margin_hor'] = "0"
            self.config['margin_left'] = "0"
            self.config['margin_right'] = "0"
            self.config['margin_up'] = "0"
            self.config['margin_down'] = "0"

            # Multiplier
            self.config['multiplier_y'] = "100"
            self.config['multiplier_x'] = "100"
            
            # Other
            self.config['interpolation'] = "Linear"
            self.config['boundary'] = "stop"
            self.config['start_zone'] = "odor_present"
            self.config['threshold'] = "3e-6"
            self.config['name'] = ""
            self.config['seed'] = "12131415"

            # Matplotlib frame
            self.mpl_frame, _ = plt.subplots(figsize=(10,3))


        def refresh(self):
            # REFRESH PLOT
            ax = self.mpl_frame.get_axes()[0]
            ax.clear()

            shape = np.array([int(self.config['discretization_y']), int(self.config['discretization_x'])])
            data_bounds = np.array([
                [int(self.config['margin_up']), shape[0] - int(self.config['margin_down'])],
                [int(self.config['margin_left']), shape[1] - int(self.config['margin_right'])]
            ])

            # Interpolation of new data
            interpolation_options = {
                'Nearest': cv2.INTER_NEAREST,
                'Linear': cv2.INTER_LINEAR,
                'Cubic': cv2.INTER_CUBIC
            }

            new_data_shape = np.diff(data_bounds, axis=1)[:,0].astype(int).tolist()
            new_data = np.zeros((len(self.data), *new_data_shape))
            for i in range(len(self.data)):
                new_data[i] = cv2.resize(self.data[i], new_data_shape[::-1], interpolation=interpolation_options[self.config['interpolation']])

            # Source position
            source_position = np.array([(int(self.config['data_source_y']) + int(self.config['margin_up'])), 
                                        (int(self.config['data_source_x']) + int(self.config['margin_left']))])
            source_radius = int(self.config['source_radius'])

            # Computing start zone
            start_probabilities = np.zeros(shape, dtype=float)
            if self.config['start_zone'] == 'odor_present':
                start_probabilities[data_bounds[0,0]:data_bounds[0,1], data_bounds[1,0]:data_bounds[1,1]] = np.where(np.sum((new_data > float(self.config['threshold'])).astype(float), axis=0) > 0, 1.0, 0.0)
            else:
                start_probabilities[data_bounds[0,0]:data_bounds[0,1], data_bounds[1,0]:data_bounds[1,1]] = 1.0

            source_mask = np.fromfunction(lambda x,y: ((x - source_position[0])**2 + (y - source_position[1])**2) <= source_radius**2, shape=shape)
            start_probabilities[source_mask] = 0

            start_probabilities /= np.sum(start_probabilities)

            # Odor grid
            odor = plt.Rectangle([0,0], 1, 1, color='black', fill=True)
            frame_data = (new_data[0] > float(self.config['threshold'])).astype(float)
            environment_frame = np.zeros(shape, dtype=float)
            environment_frame[data_bounds[0,0]:data_bounds[0,1], data_bounds[1,0]:data_bounds[1,1]] = frame_data
            ax.imshow(environment_frame, cmap='Greys')

            # Start zone contour
            start_zone = plt.Rectangle([0,0], 1, 1, color='blue', fill=False)
            ax.contour(start_probabilities, levels=[0.0], colors='blue')

            # Source circle
            goal_circle = plt.Circle(source_position[::-1], source_radius, color='r', fill=False)
            ax.add_patch(goal_circle)
            
            # Legend
            ax.legend([odor, start_zone, goal_circle], [f'Frame odor cues', 'Start zone', 'Source'])

            # Crop to size
            ax.set_ylim([shape[0]-1, 0])
            ax.set_xlim([0, shape[1]-1])

            # Add axis ticks
            ax.set_yticks(np.arange(0, shape[0], step=10).tolist() + [shape[0]-1 if shape[0]-1 % 10 != 0 else None])
            ax.set_xticks(np.arange(0, shape[1], step=10).tolist() + [shape[1]-1 if shape[1]-1 % 10 != 0 else None])

            # Refresh canvas
            self.canvas.draw()


    data_config = EnvironmentConfig()

    def gather_entries_and_refresh():
        for k, v in entry_fields.items():
            data_config.config[k] = v.get()
        
        data_config.refresh()

    # Base of the window
    root = tk.Tk()
    root.title('olfactory-navigation - Environment Builder')

    root.bind('<Return>', lambda _: gather_entries_and_refresh())

    def closeAction():
        root.destroy()
        quit()

    root.protocol('WM_DELETE_WINDOW', closeAction)

    root_panel = tk.Frame(root)
    root_panel.pack(side="bottom", fill="both", expand="yes")

    # FILE CHOOSER
    file_panel = tk.Frame(root_panel)
    file_panel.grid(row=0, columnspan=2, sticky="w", padx=frame_padding, pady=frame_padding)

    file_chooser_label = tk.Label(master=file_panel, text='Base file:')
    file_chooser_label.pack(side="left", padx=5)

    file_path_entry = tk.Entry(master=file_panel, width=100)
    file_path_entry.insert(0, "G:/My Drive/Documents/Universita di Genova/Pimlb work/nose_data_27_123.npy")
    file_path_entry.pack(side="left", fill="x", padx=5)

    def browseFiles():
        data_file = filedialog.askopenfilename(initialdir = "./",
                                            title = "Select a File",
                                            filetypes = (("Numpy files",
                                                            "*.npy*"),
                                                        ("all files",
                                                            "*.*")))

        file_path_entry.delete(0, tk.END)
        file_path_entry.insert(0, data_file)

    browse_button = tk.Button(master=file_panel, text="Browse", command=browseFiles) 
    browse_button.pack(side="left", padx=5)

    def loadFiles():
        data_config.data = np.load(file_path_entry.get())
        print(f'Data loaded with shape {data_config.data.shape}')

        printDataInfo()
        otherParameters()
        totalSizeConfig()
        sourceConfig()
        marginConfig()
        multiplierConfig()
        finalizeFrame()
        createPreviewWindow()

    load_data_button = tk.Button(master=file_panel, text="Load", command=loadFiles)
    load_data_button.pack(side="left", padx=5)

    # Def print data info:
    def printDataInfo():
        data_info_frame = tk.Frame(root_panel)
        data_info_frame.grid(row=1, column=0, sticky="nw", padx=frame_padding, pady=frame_padding)

        data_info_label = tk.Label(data_info_frame, text='Data info', font=bold_font)
        data_info_label.grid(row=0, columnspan=2, sticky="w", pady=5)

        for i, cat in enumerate(['Timestamps', 'Height', 'Width']):

            cat_label = tk.Label(master=data_info_frame, text=(cat + ": "))
            cat_label.grid(column=0, row=i+1, sticky="w")

            frame = tk.Frame(master=data_info_frame, relief=tk.SUNKEN, borderwidth=2)
            frame.grid(column=1, row=i+1, sticky="w")

            info_label = tk.Label(master=frame, text=str(data_config.data.shape[i]))
            info_label.pack(side="left")


    def otherParameters():
        other_params_frame = tk.Frame(root_panel)
        other_params_frame.grid(row=3, column=1, sticky="nw", padx=frame_padding, pady=frame_padding)

        other_params_label = tk.Label(other_params_frame, text="Other Parameters", font=bold_font)
        other_params_label.grid(row=0, columnspan=2, sticky="nw", pady=5)

        # Interpolation
        interpolation_label = tk.Label(other_params_frame, text="Interpolation")
        interpolation_label.grid(row=1, column=0, sticky="nw")

        interpolation_variable = tk.StringVar(other_params_frame)
        interpolation_variable.set(data_config.config['interpolation'])
        interpolation_variable.trace_add("write", lambda *args: gather_entries_and_refresh())

        interpolation_choice = tk.OptionMenu(other_params_frame, interpolation_variable, *['Nearest', 'Linear', 'Cubic'])
        interpolation_choice.grid(row=1, column=1, sticky="nw")
        entry_fields['interpolation'] = interpolation_variable

        # Boundary
        boundary_label = tk.Label(other_params_frame, text="Boundary")
        boundary_label.grid(row=2, column=0, sticky="nw")

        boundary_variable = tk.StringVar(other_params_frame)
        boundary_variable.set(data_config.config['boundary'])

        boundary_choice = tk.OptionMenu(other_params_frame, boundary_variable, *['stop', 'wrap', 'wrap_vertical', 'wrap_horizontal', 'clip', 'no'])
        boundary_choice.grid(row=2, column=1, sticky="nw")
        entry_fields['boundary'] = boundary_variable

        # Start
        start_label = tk.Label(other_params_frame, text="Start zone")
        start_label.grid(row=3, column=0, sticky="nw")

        start_variable = tk.StringVar(other_params_frame)
        start_variable.set(data_config.config['start_zone'])
        start_variable.trace_add("write", lambda *args: gather_entries_and_refresh())

        start_choice = tk.OptionMenu(other_params_frame, start_variable, *['odor_present', 'data_zone'])
        start_choice.grid(row=3, column=1, sticky="nw")
        entry_fields['start_zone'] = start_variable

        # Odor Present
        threshold_label = tk.Label(other_params_frame, text="Odor threshold")
        threshold_label.grid(row=4, column=0, sticky="nw")

        threshold_entry = tk.Entry(other_params_frame, width=5, validate="focusout", validatecommand=gather_entries_and_refresh)
        threshold_entry.grid(row=4, column=1, sticky="nw")
        threshold_entry.insert(0, data_config.config['threshold'])
        entry_fields['threshold'] = threshold_entry

        # Name
        name_label = tk.Label(other_params_frame, text="Environment Name")
        name_label.grid(row=5, column=0, sticky="nw")

        name_entry = tk.Entry(other_params_frame, width=20)
        name_entry.grid(row=5, column=1, sticky="nw")
        name_entry.insert(0, data_config.config['name'])
        entry_fields['name'] = name_entry

        # Seed
        seed_label = tk.Label(other_params_frame, text="Seed")
        seed_label.grid(row=6, column=0, sticky="nw")

        seed_entry = tk.Entry(other_params_frame, width=10)
        seed_entry.grid(row=6, column=1, sticky="nw")
        seed_entry.insert(0, data_config.config['seed'])
        entry_fields['seed'] = seed_entry


    def createValueConfig(in_frame, at_row, name, config_name):
        value_label = tk.Label(in_frame, text=(name + ":"))
        value_label.grid(row=at_row, column=0, sticky="w")

        value_entry_frame = tk.Frame(in_frame)
        value_entry_frame.grid(row=at_row, column=1, sticky="w")

        value_entry = tk.Entry(value_entry_frame, width=4, validate="focusout", validatecommand=gather_entries_and_refresh)
        value_entry.insert(0, str(data_config.config[config_name])) # Set default value
        
        entry_fields[config_name] = value_entry

        def changeEntryValue(entry:tk.Entry, val:int):
            current_val = int(entry.get())
            entry.delete(0, tk.END)

            new_val = current_val + val
            entry.insert(0, str(new_val))

            if config_name in linked_fields:
                for linked_config in linked_fields[config_name]:
                    linked_entry = entry_fields[linked_config]

                    current_val = int(linked_entry.get())
                    linked_entry.delete(0, tk.END)

                    new_val = current_val + val
                    linked_entry.insert(0, str(new_val))                

            gather_entries_and_refresh()

        value_decrease_10 = tk.Button(value_entry_frame, text="--", command=(lambda: changeEntryValue(value_entry, -10)))
        value_decrease_10.grid(row=0, column=0, sticky="w", padx=2)
        value_decrease_1 = tk.Button(value_entry_frame, text="-", command=(lambda: changeEntryValue(value_entry, -1)))
        value_decrease_1.grid(row=0, column=1, sticky="w", padx=2)

        value_entry.grid(row=0, column=2, sticky="w", padx=2)

        value_increase_1 = tk.Button(value_entry_frame, text="+", command=(lambda: changeEntryValue(value_entry, 1)))
        value_increase_1.grid(row=0, column=3, sticky="w", padx=2)
        value_increase_10 = tk.Button(value_entry_frame, text="++", command=(lambda: changeEntryValue(value_entry, 10)))
        value_increase_10.grid(row=0, column=4, sticky="w", padx=2)


    # FINAL (or TOTAL) SIZE CONFIG
    def totalSizeConfig():
        total_size_frame = tk.Frame(root_panel)
        total_size_frame.grid(row=2, column=0, sticky="nw", padx=frame_padding, pady=frame_padding)

        total_size_label = tk.Label(master=total_size_frame, text="Total Size Configuration", font=bold_font)
        total_size_label.grid(row=0, columnspan=2, sticky="w", pady=5)

        createValueConfig(in_frame=total_size_frame, at_row=1, name="Total Height", config_name="discretization_y")
        createValueConfig(in_frame=total_size_frame, at_row=2, name="Total Width", config_name="discretization_x")


    # SOURCE CONFIG
    def sourceConfig():
        source_frame = tk.Frame(root_panel)
        source_frame.grid(row=1, column=1, sticky="nw", padx=frame_padding, pady=frame_padding)

        source_label = tk.Label(master=source_frame, text="Source Configuration", font=bold_font)
        source_label.grid(row=0, columnspan=2, sticky="w", pady=5)

        createValueConfig(in_frame=source_frame, at_row=1, name="Source Y", config_name="data_source_y")
        createValueConfig(in_frame=source_frame, at_row=2, name="Source X", config_name="data_source_x")
        createValueConfig(in_frame=source_frame, at_row=3, name="Source radius", config_name="source_radius")


    # MARGINS CONFIG
    def marginConfig():
        margin_config_frame = tk.Frame(root_panel)
        margin_config_frame.grid(row=3, column=0, sticky="nw", padx=frame_padding, pady=frame_padding)

        margin_config_label = tk.Label(master=margin_config_frame, text="Margins Configuration", font=bold_font)
        margin_config_label.grid(row=0, columnspan=2, sticky="w", pady=5)

        createValueConfig(in_frame=margin_config_frame, at_row=1, name="All margins", config_name="margin_all")
        createValueConfig(in_frame=margin_config_frame, at_row=2, name="Margins Vertical", config_name="margin_ver")
        createValueConfig(in_frame=margin_config_frame, at_row=3, name="Margins Horizontal", config_name="margin_hor")

        createValueConfig(in_frame=margin_config_frame, at_row=4, name="Margin Up", config_name="margin_up")
        createValueConfig(in_frame=margin_config_frame, at_row=5, name="Margin Down", config_name="margin_down")
        createValueConfig(in_frame=margin_config_frame, at_row=6, name="Margin Left", config_name="margin_left")
        createValueConfig(in_frame=margin_config_frame, at_row=7, name="Margin Right", config_name="margin_right")


    # MULT CONFIG
    def multiplierConfig():
        mult_config_frame = tk.Frame(root_panel)
        mult_config_frame.grid(row=2, column=1, sticky="nw", padx=frame_padding, pady=frame_padding)

        mult_config_label = tk.Label(master=mult_config_frame, text="Multiplier Configuration", font=bold_font)
        mult_config_label.grid(row=0, columnspan=2, sticky="w", pady=5)

        createValueConfig(in_frame=mult_config_frame, at_row=1, name="Height Multiplier (%)", config_name="multiplier_y")
        createValueConfig(in_frame=mult_config_frame, at_row=2, name="Width Multiplier (%)", config_name="multiplier_x")


    # FINALIZE
    def finalizeFrame():
        finalize_frame = tk.Frame(root_panel)
        finalize_frame.grid(row=4, columnspan=2, padx=frame_padding, pady=frame_padding)

        # REFRESH
        refresh_button = tk.Button(finalize_frame, text="RELOAD", font=bold_font, command=gather_entries_and_refresh)
        refresh_button.pack(side="left", padx=30)

        # SAVE TO FILE
        def save_to_file():
            pass
        save_button = tk.Button(finalize_frame, text="Save", font=bold_font, command=save_to_file)
        save_button.pack(side="left", padx=5)

        # PRINT
        def print_to_cmd():
            pass
        print_button = tk.Button(finalize_frame, text="Print", font=bold_font, command=print_to_cmd)
        print_button.pack(side="left", padx=5)



    # PREVIEW WINDOW
    def createPreviewWindow():
        preview_frame = tk.Frame(root_panel)
        preview_frame.grid(row=5, columnspan=2, sticky="nw", padx=frame_padding, pady=frame_padding)

        preview_label = tk.Label(preview_frame, text="Preview", font=bold_font)
        preview_label.pack(anchor="nw", pady=5)

        canvas = FigureCanvasTkAgg(data_config.mpl_frame, master=preview_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=1)

        data_config.canvas = canvas
        data_config.refresh()

    tk.mainloop()


if __name__ == "__main__":
    buildWindow()