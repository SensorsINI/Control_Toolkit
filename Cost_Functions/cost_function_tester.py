from Control_Toolkit.Cost_Functions.cost_function_wrapper import CostFunctionWrapper
from types import SimpleNamespace

import matplotlib.pyplot as plt

import platform
if platform.system() == 'Darwin':
    import matplotlib
    matplotlib.use("MacOSX")


class CostFunctionTester:
    def __init__(self, cost_function, mode='single point'):
        # self.cost_function = CostFunctionWrapper()
        #
        self.mode = mode

        self.cost_function = cost_function.cost_function
        self.lib = self.cost_function.lib

        self.buffers = SimpleNamespace()

        for k in self.cost_function.cost_components.__dict__:
            setattr(self.buffers, k, [])

        # if mode == 'single_point':
        #     batch_size = 1
        #     horizon = 0
        # elif mode == 'single trajectory':
        #     batch_size = 1
        #     horizon = cost_function.cost_function.horizon
        # else:
        #     batch_size = cost_function.horizon
        #     horizon = cost_function.horizon
        #
        # self.cost_function.configure(
        #     batch_size=batch_size,
        #     horizon=horizon,
        #     variable_parameters=self.cost_function.variable_parameters,
        #     environment_name=self.cost_function.environment_name,
        #     computation_library=self.cost_function.computation_library,
        #     cost_function_specification=self.cost_function.cost_function_specification
        # )

    def collect_costs(self):
        for k in self.buffers.__dict__:
            self.buffers.__dict__[k].append(self.lib.to_numpy(self.cost_function.cost_components.__dict__[k][0, 0]))

    def plot(self):
        plt.figure()
        for k in self.buffers.__dict__:
            feature = self.lib.to_numpy(self.lib.stack(self.buffers.__dict__[k]))
            plt.plot(feature, label=k)

        plt.legend()
        plt.show()
        self.clear_buffers()

    def clear_buffers(self):
        for k in self.buffers.__dict__:
            self.buffers.__dict__[k] = []




