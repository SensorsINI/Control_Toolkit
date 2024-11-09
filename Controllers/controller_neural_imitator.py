from SI_Toolkit.Functions.FunctionalDict import FunctionalDict
from SI_Toolkit.Predictors.neural_network_evaluator import neural_network_evaluator
from SI_Toolkit.computation_library import TensorType, NumpyLibrary

import numpy as np

from Control_Toolkit.Controllers import template_controller

try:
    from SI_Toolkit_ASF.ToolkitCustomization.predictors_customization import STATE_INDICES
except ModuleNotFoundError:
    print("SI_Toolkit_ASF not yet created")


class controller_neural_imitator(template_controller):
    _computation_library = NumpyLibrary

    def configure(self):

        self.net_evaluator = neural_network_evaluator(
            net_name=self.config_controller["net_name"],
            path_to_models=self.config_controller["PATH_TO_MODELS"],
            batch_size=1, # It makes sense only for testing (Brunton plot for Q) of not rnn networks to make bigger batch, this is not implemented
            input_precision=self.config_controller["input_precision"],
            hls4ml=self.config_controller["hls4ml"])

        self._computation_library = self.net_evaluator.lib

        self.input_at_input = self.config_controller["input_at_input"]

        self.state_2_input_idx = []
        self.remaining_inputs = self.net_evaluator.net_info.inputs.copy()
        for key in self.net_evaluator.net_info.inputs:
            if key in STATE_INDICES.keys():
                self.state_2_input_idx.append(STATE_INDICES.get(key))
                self.remaining_inputs.remove(key)
            else:
                break  # state inputs must be adjacent in the current implementation

        if self.controller_logging:
            self.controller_data_for_csv = FunctionalDict(get_memory_states(self.net_evaluator.net))

        print('Configured neural imitator with {} network with {} library'.format(self.net_evaluator.net_info.net_full_name, self.net_evaluator.net_info.library))

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):

        if self.input_at_input:
            net_input = s
        else:
            self.update_attributes(updated_attributes)
            net_input = s[..., self.state_2_input_idx]
            for key in self.remaining_inputs:
                net_input = np.append(net_input, getattr(self.variable_parameters, key))

        Q = self.net_evaluator.step(net_input)

        return Q

    def controller_reset(self):
        self.configure()


def get_memory_states(net):
    import json
    memory_state = {}
    recurrent_layer_index = 0

    for layer in net.layers:
        # Determine the type of recurrent layer
        if 'gru' in layer.name.lower():
            memory_cell_name = 'GRU_H'
        elif 'lstm' in layer.name.lower():
            memory_cell_name = 'LSTM_H'
        elif 'rnn' in layer.name.lower():
            memory_cell_name = 'RNN_H'
        else:
            memory_cell_name = None

        if memory_cell_name:
            recurrent_layer_index += 1
            memory_cell_name += str(recurrent_layer_index)

            for state_idx, single_state in enumerate(layer.states):
                if 'LSTM' in memory_cell_name:
                    # LSTM has two states: hidden (h) and cell (c)
                    if state_idx == 0:
                        key = f"{memory_cell_name}_h"
                    else:
                        key = f"{memory_cell_name}_c"
                else:
                    # GRU and RNN typically have a single hidden state
                    key = memory_cell_name

                # Use a lambda with a default argument to capture the current single_state
                key = key+f'({len(single_state.numpy().flatten().tolist())})'
                memory_state[key] = lambda s=single_state: json.dumps(s.numpy().flatten().tolist())
                # single_state_template = single_state.numpy().flatten().tolist()
                # max_index = len(single_state_template) - 1
                # padding_length = len(str(max_index))
                #
                # # Iterate and add functions to memory_state with dynamic zero-padding
                # for i in range(len(single_state_template)):
                #     key_element = f"{key}_{i:0{padding_length}d}"  # Pad with zeros dynamically based on max index
                #     memory_state[key_element] = lambda s=single_state: s.numpy().flatten().tolist()[i]

    return memory_state
