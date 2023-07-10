from types import SimpleNamespace
from SI_Toolkit.computation_library import TensorType, NumpyLibrary

import numpy as np

from Control_Toolkit.Controllers import template_controller
from SI_Toolkit.Functions.General.Normalising import get_normalization_function, get_denormalization_function

try:
    from SI_Toolkit_ASF.predictors_customization import STATE_INDICES
except ModuleNotFoundError:
    print("SI_Toolkit_ASF not yet created")

from SI_Toolkit.Functions.General.Initialization import (get_net,
                                                         get_norm_info_for_net)
from SI_Toolkit.Functions.TF.Compile import CompileAdaptive


class controller_neural_imitator(template_controller):
    _computation_library = NumpyLibrary

    def configure(self):
        NET_NAME = self.config_controller["net_name"]
        PATH_TO_MODELS = self.config_controller["PATH_TO_MODELS"]

        self.input_at_input = self.config_controller["input_at_input"]

        a = SimpleNamespace()
        self.batch_size = 1  # It makes sense only for testing (Brunton plot for Q) of not rnn networks to make bigger batch, this is not implemented

        a.path_to_models = PATH_TO_MODELS
        a.net_name = NET_NAME

        # Create a copy of the network suitable for inference (stateful and with sequence length one)
        self.net, self.net_info = \
            get_net(a, time_series_length=1,
                    batch_size=self.batch_size, stateful=True)

        self.normalization_info = get_norm_info_for_net(self.net_info)
        self.normalize_inputs = get_normalization_function(self.normalization_info, self.net_info.inputs, self.lib)
        self.denormalize_outputs = get_denormalization_function(self.normalization_info, self.net_info.outputs,
                                                                self.lib)

        self.net_input_normed = self.lib.to_variable(
            np.zeros([len(self.net_info.inputs), ], dtype=np.float32), self.lib.float32)

        self.step_compilable = CompileAdaptive(self._step_compilable)

        self.state_2_input_idx = []
        self.remaining_inputs = self.net_info.inputs.copy()
        for key in self.net_info.inputs:
            if key in STATE_INDICES.keys():
                self.state_2_input_idx.append(STATE_INDICES.get(key))
                self.remaining_inputs.remove(key)
            else:
                break  # state inputs must be adjacent in the current implementation

        if self.net_info.library == 'Pytorch':
            from SI_Toolkit.computation_library import PyTorchLibrary
            self._computation_library = PyTorchLibrary
        elif self.net_info.library == 'TF':
            from SI_Toolkit.computation_library import TensorFlowLibrary
            self._computation_library = TensorFlowLibrary

        if self.lib.lib == 'Pytorch':
            from SI_Toolkit.Functions.Pytorch.Network import get_device
            self.device = get_device()
            self.net.reset()
            self.net.eval()

        print('Configured neural imitator with {} network with {} library'.format(self.net_info.net_full_name, self.net_info.library))

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):

        if self.input_at_input:
            net_input = s
        else:
            self.update_attributes(updated_attributes)
            net_input = s[..., self.state_2_input_idx]
            for key in self.remaining_inputs:
                net_input = np.append(net_input, getattr(self.variable_parameters, key))

        net_input = self.lib.to_tensor(net_input, self.lib.float32)

        if self.lib.lib == 'Pytorch':
            net_input = net_input.to(self.device)

        net_output = self.step_compilable(net_input)

        if self.lib.lib == 'Pytorch':
            net_output = net_output.detach().numpy()

        Q = net_output

        return Q

    def controller_reset(self):
        self.configure()

    def _step_compilable(self, net_input):

        self.lib.assign(self.net_input_normed, self.normalize_inputs(net_input))

        net_input = self.lib.reshape(self.net_input_normed, (-1, 1, len(self.net_info.inputs)))

        net_output = self.net(net_input)

        net_output = self.denormalize_outputs(net_output)

        return net_output
