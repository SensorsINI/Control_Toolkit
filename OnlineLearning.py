import tensorflow as tf
from tensorflow import keras
import numpy as np
from SI_Toolkit.Functions.TF.Loss import loss_msr_sequence_customizable
from SI_Toolkit.computation_library import TensorFlowLibrary
from SI_Toolkit.Functions.General.Normalising import get_denormalization_function, get_normalization_function


class OnlineLearning:
    def __init__(self, predictor, dt):
        self.predictor = predictor
        self.dt = dt

        self.s_norm_previous = None
        self.u_norm_previous = None

        self.s_previous = None
        self.u_previous = None

        self.normalization_info = self.predictor.predictor.normalization_info
        self.lib = TensorFlowLibrary
        self.normalize_outputs = get_normalization_function(self.normalization_info, self.predictor.predictor.net_info.outputs, self.lib)
        self.denormalize_outputs = get_denormalization_function(self.normalization_info, self.predictor.predictor.net_info.outputs, self.lib)

        self.predictor.predictor.net.compile(
            loss=loss_msr_sequence_customizable(wash_out_len=0,
                                                post_wash_out_len=1,
                                                discount_factor=1.0),
            optimizer=keras.optimizers.Adam(1e-2)
        )

    def step(self, s, u, time, updated_attributes):
        s_norm = self.predictor.predictor.normalize_inputs(s)
        u_norm = self.predictor.predictor.normalize_control_inputs(u)

        if self.s_norm_previous is not None and self.u_norm_previous is not None:
            net_input = tf.concat([self.u_norm_previous, self.s_norm_previous], 0) # This order is neede for normalization
            net_input = tf.reshape(net_input, [1, 1, len(net_input)])

            if np.any(['D_' in output_name for output_name in self.predictor.predictor.net_info.outputs]):
                delta_s = (s - self.s_previous) / self.dt
                delta_s = tf.reshape(delta_s, [1, len(delta_s)])
                delta_s_norm = self.normalize_outputs(delta_s)
                s_measured = delta_s_norm
            else:
                s_measured = s_norm

            self.predictor.predictor.net.fit(net_input, s_measured, batch_size=1, epochs=1)

        self.s_norm_previous = s_norm
        self.u_norm_previous = u_norm

        self.s_previous = s
        self.u_previous = u
