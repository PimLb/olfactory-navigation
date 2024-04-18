from datetime import datetime
from .pbvi_agent import PBVI_Agent, TrainingHistory
from .model_based_util.value_function import ValueFunction
from .model_based_util import vi_solver

import numpy as np
gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')


class QMDP_Agent(PBVI_Agent):
    def train(self,
              expansions:int,
              initial_value_function:ValueFunction|None=None,
              gamma:float=0.99,
              eps:float=1e-6,
              use_gpu:bool=False,
              history_tracking_level:int=1,
              force:bool=False,
              print_progress:bool=True
              ) -> TrainingHistory:
        # GPU support
        if use_gpu:
            assert gpu_support, "GPU support is not enabled, Cupy might need to be installed..."

        # Handeling the case where the agent is already trained
        if (self.value_function is not None) and (not force):
            raise Exception('Agent has already been trained. The force parameter needs to be set to "True" if training should still happen')
        else:
            self.trained_at = None
            self.name = '-'.join(self.name.split('-')[:-1])
            self.value_function = None

        model = self.model if not use_gpu else self.model.gpu_model

        # Value Iteration solving
        value_function, hist = vi_solver.solve(model=model,
                                           horizon=expansions,
                                           initial_value_function=initial_value_function,
                                           gamma=gamma,
                                           eps=eps,
                                           use_gpu=use_gpu,
                                           history_tracking_level=history_tracking_level,
                                           print_progress=print_progress)

        # Record when it was trained
        self.trained_at = datetime.now().strftime("%m%d%Y_%H%M%S")
        self.name += f'-trained_{self.trained_at}'

        self.value_function = value_function.to_cpu() if not self.on_gpu else value_function.to_gpu()

        return hist