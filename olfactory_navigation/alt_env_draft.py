

class LayeredEnvironment(Environment):
    '''
    # TODO: Write documentation for layered environment
    # Shape should be ltyx
    '''
    def __init__(self,
                 data_file: str | np.ndarray,
                 data_source_position: list | np.ndarray,
                 base_layer: int | str | None = None,
                 source_radius: float = 1,
                 shape: list | np.ndarray | None = None,
                 margins: int | list | np.ndarray = 0,
                 multiplier: list | np.ndarray = [1, 1],
                 interpolation_method: Literal['Nearest', 'Linear', 'Cubic'] = 'Linear',
                 preprocess_data: bool = False,
                 boundary_condition: Literal['stop', 'wrap', 'wrap_vertical', 'wrap_horizontal', 'clip', 'no'] = 'stop',
                 start_zone: np.ndarray | Literal['odor_present', 'data_zone'] = 'data_zone',
                 odor_present_threshold: float | None = None,
                 name: str | None = None,
                 seed: int = 12131415
                 ) -> None:

        # Loading data
        layer_labels = None
        loaded_data = None
        if isinstance(data_file, str):
            # NUMPY
            if data_file.endswith('.npy'):
                loaded_data = np.load(data_file)
            # H5
            elif data_file.endswith('.h5'):
                loaded_data = h5py.File(data_file, 'r')
                layer_labels = list(loaded_data.keys())
                loaded_data = [layer for layer in loaded_data.values()]
            # Not supported
            else:
                raise NotImplementedError('File format loading not implemented')
        elif not any(isinstance(data_file, cls) for cls in [np.ndarray, h5py.File, h5py.Group, h5py.Dataset]):
            raise NotImplementedError("Data file should be either a path or an object that is either an h5 object or a numpy array")
        
        layered_data = loaded_data if loaded_data is not None else data_file

        layer_count = len(loaded_data)
        if layer_labels is None:
            layer_labels = [f'{i}' for i in range(layer_count)]
        
        # Choosing base layer id
        base_layer_id = 0
        if isinstance(base_layer, int):
            assert base_layer < layer_count, "The base layer id is higher than the amount of layers available..."
            base_layer_id = base_layer
        elif isinstance(base_layer, str):
            for i, layer_label in enumerate(layer_labels):
                if base_layer == layer_label:
                    base_layer_id = i
                    break
            else:
                raise ValueError("The base_layer label provided is not found.")

        # Encoding base layer in the parent object
        super().__init__(data_file = layered_data[base_layer_id],
                         data_source_position = data_source_position,
                         source_radius = source_radius,
                         shape = shape,
                         margins = margins,
                         multiplier = multiplier,
                         interpolation_method = interpolation_method,
                         preprocess_data = preprocess_data,
                         boundary_condition = boundary_condition,
                         start_zone = start_zone,
                         odor_present_threshold = odor_present_threshold,
                         name = name,
                         seed = seed)
        
        # Setting data_file parameters
        if isinstance(data_file, str):
            self.data_file_path = data_file

        # Setting array of layers
        self.layers = [None for _ in range(layer_count)]
        self.layers[base_layer_id] = super(self)

        # Encoding other layers
        for layer_i in range(layer_count):
            if layer_i == base_layer_id:
                continue
            self.layers[layer_i] = Environment(data_file = layered_data[layer_i],
                                               data_source_position = data_source_position,
                                               source_radius = source_radius,
                                               shape = shape,
                                               margins = margins,
                                               multiplier = multiplier,
                                               interpolation_method = interpolation_method,
                                               preprocess_data = preprocess_data,
                                               boundary_condition = boundary_condition,
                                               start_zone = start_zone,
                                               odor_present_threshold = odor_present_threshold,
                                               name = name,
                                               seed = seed)



    def get_observation(self,
                        pos: np.ndarray,
                        time: int | np.ndarray = 0
                        ) -> float | np.ndarray: #TODO
        return super().get_observation(pos, time)
        

class Environment3D(Environment):
    '''
    # TODO: Write documentation of 3D environment
    '''
    def __init__(self,
                 data_file: str | np.ndarray,
                 data_source_position: list | np.ndarray,
                 source_radius: float = 1,
                 shape: list | np.ndarray | None = None,
                 margins: int | list | np.ndarray = 0,
                 multiplier: list | np.ndarray = [1, 1],
                 interpolation_method: Literal['Nearest', 'Linear', 'Cubic'] = 'Linear',
                 preprocess_data: bool = False,
                 boundary_condition: Literal['stop', 'wrap', 'wrap_vertical', 'wrap_horizontal', 'clip', 'no'] = 'stop',
                 start_zone: np.ndarray | Literal['odor_present', 'data_zone'] = 'data_zone',
                 odor_present_threshold: float | None = None,
                 name: str | None = None,
                 seed: int = 12131415) -> None:
        super().__init__(data_file = data_file,
                         data_source_position = data_source_position,
                         source_radius = source_radius,
                         shape = shape,
                         margins = margins,
                         multiplier = multiplier,
                         interpolation_method = interpolation_method,
                         preprocess_data = preprocess_data,
                         boundary_condition = boundary_condition,
                         start_zone = start_zone,
                         odor_present_threshold = odor_present_threshold,
                         name = name,
                         seed = seed)