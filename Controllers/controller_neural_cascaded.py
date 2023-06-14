from types import SimpleNamespace

import numpy

from SI_Toolkit.computation_library import TensorType, NumpyLibrary

import numpy as np

from Control_Toolkit.Controllers import template_controller
from SI_Toolkit.load_and_normalize import normalize_numpy_array, denormalize_numpy_array

try:
    from SI_Toolkit_ASF.predictors_customization import STATE_INDICES
except ModuleNotFoundError:
    print("SI_Toolkit_ASF not yet created")

from SI_Toolkit.Functions.General.Initialization import (get_net,
                                                         get_norm_info_for_net)
from SI_Toolkit.Functions.TF.Compile import CompileAdaptive


class controller_neural_cascaded(template_controller):
    _computation_library = NumpyLibrary

    def configure(self):
        NET_NAME_1 = self.config_controller["net_name_1"]
        PATH_TO_MODEL_1 = self.config_controller["PATH_TO_MODEL_1"]
        NET_NAME_2 = self.config_controller["net_name_2"]
        PATH_TO_MODEL_2 = self.config_controller["PATH_TO_MODEL_2"]

        a = SimpleNamespace()
        b = SimpleNamespace()
        self.batch_size = 1  # It makes sense only for testing (Brunton plot for Q) of not rnn networks to make bigger batch, this is not implemented

        a.path_to_models = PATH_TO_MODEL_1
        a.net_name = NET_NAME_1
        b.path_to_models = PATH_TO_MODEL_2
        b.net_name = NET_NAME_2

        # Create a copy of the network suitable for inference (stateful and with sequence length one)
        self.netA, self.netA_info = \
            get_net(a, time_series_length=1,
                    batch_size=self.batch_size, stateful=True)

        self.netB, self.netB_info = \
            get_net(b, time_series_length=1,
                    batch_size=self.batch_size, stateful=True)

        self.normalization_infoA = get_norm_info_for_net(self.netA_info)
        self.normalization_infoB = get_norm_info_for_net(self.netB_info)

        self.evaluate_netA = CompileAdaptive(self._evaluate_netA)
        self.evaluate_netB = CompileAdaptive(self._evaluate_netB)

        self.state_2_input_idx = []
        self.remaining_inputs = self.netA_info.inputs.copy()
        for key in self.netA_info.inputs:
            if key in STATE_INDICES.keys():
                self.state_2_input_idx.append(STATE_INDICES.get(key))
                self.remaining_inputs.remove(key)
            else:
                break  # state inputs must be adjacent in the current implementation

        if self.netA_info.library == 'Pytorch':
            from SI_Toolkit.computation_library import PyTorchLibrary
            self._computation_library = PyTorchLibrary
        elif self.netA_info.library == 'TF':
            from SI_Toolkit.computation_library import TensorFlowLibrary
            self._computation_library = TensorFlowLibrary

        if self.lib.lib == 'Pytorch':
            from SI_Toolkit.Functions.Pytorch.Network import get_device
            self.device = get_device()
            self.netA.reset()
            self.netA.eval()
            self.netB.reset()
            self.netB.eval()

        print('Configured neural imitator with {} network with {} library'.format(self.netA_info.net_full_name, self.netA_info.library))

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        self.update_attributes(updated_attributes)

        net_input = s[..., self.state_2_input_idx]
        for key in self.remaining_inputs:
            net_input = np.append(net_input, getattr(self.variable_parameters, key))

        net_inputB = net_input

        net_input = normalize_numpy_array(
            net_input, self.netA_info.inputs, self.normalization_infoA
        )

        net_input = np.reshape(net_input, [-1, 1, len(self.netA_info.inputs)])

        net_input = self.lib.to_tensor(net_input, dtype=self.lib.float32)

        if self.lib.lib == 'Pytorch':
            net_input = net_input.to(self.device)

        net_output = self.evaluate_netA(net_input)

        if self.lib.lib == 'Pytorch':
            net_output = net_output.detach().numpy()

        net_output = denormalize_numpy_array(net_output, self.netA_info.outputs, self.normalization_infoA)

        L = net_output
        L_app = L.flatten()[0]
        #L_app = 0.1975

        net_inputB = numpy.insert(net_inputB,5,L_app)

        net_inputB = normalize_numpy_array(
            net_inputB, self.netB_info.inputs, self.normalization_infoB
        )
        net_inputB = np.reshape(net_inputB, [-1, 1, len(self.netB_info.inputs)])

        net_inputB = self.lib.to_tensor(net_inputB, dtype=self.lib.float32)

        if self.lib.lib == 'Pytorch':
            net_inputB = net_inputB.to(self.device)

        net_outputB = self.evaluate_netB(net_inputB)

        if self.lib.lib == 'Pytorch':
            net_outputB = net_outputB.detach().numpy()

        net_outputB = denormalize_numpy_array(net_outputB, self.netB_info.outputs, self.normalization_infoB)

        Q = net_outputB
        #print('L: ', L_app, 'Q: ', Q)

        return Q

    def _evaluate_netA(self, net_input):
        net_output = self.netA(net_input)
        return net_output

    def _evaluate_netB(self, net_input):
        net_output = self.netB(net_input)
        return net_output
