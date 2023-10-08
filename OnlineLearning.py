import os
import time
import pandas as pd
import shutil
import tensorflow as tf
from tensorflow import keras
from types import SimpleNamespace
import numpy as np

from SI_Toolkit.Functions.TF.Loss import loss_msr_sequence_customizable
from SI_Toolkit.computation_library import TensorFlowLibrary
from SI_Toolkit.Functions.General.Normalising import get_denormalization_function, get_normalization_function
from SI_Toolkit.load_and_normalize import normalize_df
from utilities.state_utilities import FULL_STATE_VARIABLES, CONTROL_INPUTS
from SI_Toolkit.Functions.TF.Dataset import Dataset
from SI_Toolkit.Functions.General.Initialization import create_log_file, create_full_name
from SI_Toolkit.Functions.TF.Network import load_pretrained_net_weights

import logging
logging.getLogger('tensorflow').disabled = True


class TrainingBuffer:
    def __init__(self, buffer_length, net_info, normalization_info, batch_size):
        self.buffer_length = buffer_length

        self.net_info = net_info
        self.use_diff_output = np.any(['D_' in output_name for output_name in self.net_info.outputs])

        self.full_state_names = np.concatenate((CONTROL_INPUTS, FULL_STATE_VARIABLES))
        if self.use_diff_output:
            delta_state_names = [f'D_{var}' for var in FULL_STATE_VARIABLES]
            self.full_state_names = np.concatenate((self.full_state_names, delta_state_names))
        self.data_buffer = pd.DataFrame([], columns=self.full_state_names, dtype=np.float32)

        if normalization_info is not None:
            self.normalize = True
            self.normalization_info = normalization_info
        else:
            self.normalize = False

        self.batch_size = batch_size

    def clear(self):
         self.data_buffer = pd.DataFrame([], columns=self.full_state_names, dtype=np.float32)

    def append(self, datapoint):
        new_state = pd.DataFrame([datapoint], columns=self.full_state_names)
        self.data_buffer = pd.concat((self.data_buffer, new_state), axis=0)
        self._cut_buffer()

    def full(self):
        return len(self.data_buffer) == self.buffer_length + 1

    def _cut_buffer(self):
        if len(self.data_buffer) > self.buffer_length + 1:
            self.data_buffer = self.data_buffer.iloc[1:]

    def get_data(self):
        if self.normalize:
            return Dataset(normalize_df([self.data_buffer], self.normalization_info), self.net_info, shuffle=False, inputs=self.net_info.inputs, outputs=self.net_info.outputs, batch_size=self.batch_size)
        else:
            return Dataset([self.data_buffer], self.net_info, shuffle=False, inputs=self.net_info.inputs, outputs=self.net_info.outputs, batch_size=self.batch_size)


class AddMetricsToLogger(keras.callbacks.Callback):
    def __init__(self, training_step):
        super().__init__()
        self.training_step = training_step

    def on_epoch_end(self, epoch, logs):
        try:
            mu = self.model.car_parameters_tf['mu'].numpy()
            logs['mu'] = mu
            logs['step'] = int(self.training_step)
            logs['lr'] = self.model.optimizer.lr.numpy()
        except:
            pass


class OnlineLearning:
    def __init__(self, predictor, dt, config):
        self.predictor = predictor
        self.dt = dt
        self.config = config
        self.normalize = self.predictor.predictor.net_info.normalize
        self.use_diff_output = np.any(['D_' in output_name for output_name in self.predictor.predictor.net_info.outputs])

        self.s_previous = None
        self.u_previous = None

        self.normalization_info = self.predictor.predictor.normalization_info
        self.lib = TensorFlowLibrary

        self.batch_size = config['batch_size']
        if self.normalize:
            self.training_buffer = TrainingBuffer(config['buffer_length'], self.predictor.predictor.net_info, self.normalization_info, self.batch_size)
        else:
            self.training_buffer = TrainingBuffer(config['buffer_length'], self.predictor.predictor.net_info, None, self.batch_size)

        self.net = keras.models.clone_model(self.predictor.predictor.net)
        self.net.set_weights(self.predictor.predictor.net.get_weights())
        self.get_optimizer()
        
        self.net.compile(
            loss=loss_msr_sequence_customizable(wash_out_len=0,
                                                post_wash_out_len=1,
                                                discount_factor=1.0),
            optimizer=self.optimizer,
        )
        self.net.optimizer.lr = self.lr

        self.setup_model_dir()

        self.N_step = 0
        self.training_step = 0
        self.last_steps_with_higher_loss = 0
        self.hist = None
        self.last_loss = np.inf

    def setup_model_dir(self):
        df_placeholder = pd.DataFrame({'time': [0.0, self.dt, 2 * self.dt]})
        net_info = self.predictor.predictor.net_info
        path_to_models = os.path.dirname(self.predictor.predictor.net_info.path_to_net) + '/'

        a = SimpleNamespace()
        a.path_to_models = os.path.dirname(net_info.path_to_net) + '/'
        a.normalize = self.normalize
        a.training_files = a.validation_files = a.test_files = 'Continual Learning'
        a.shift_labels = net_info.shift_labels

        create_full_name(net_info, path_to_models)
        create_log_file(net_info, a, [df_placeholder])

        dst_folder = net_info.path_to_net
        shutil.copy('SI_Toolkit_ASF/config_predictors.yml', dst_folder)
        shutil.copy('Control_Toolkit_ASF/config_controllers.yml', dst_folder)
        shutil.copy('utilities/Settings.py', dst_folder)
        shutil.copy(net_info.path_to_normalization_info, dst_folder)

        self.net.save_weights(f'{dst_folder}/ckpt.ckpt')

    def update_learning_rate(self):
        config = self.config['exponential_lr_decay']
        if config['activated']:
            self.lr = self.lr_init * pow(config['decay_rate'], self.training_step)

        config = self.config['reduce_lr_on_plateau']
        loss = self.hist.history['loss'][0]  # Only loss from last epoch
        if config['activated']:
            if loss - self.last_loss > config['min_delta']:
                self.last_steps_with_higher_loss += 1
            else:
                self.last_steps_with_higher_loss = 0

            if self.last_steps_with_higher_loss >= config['patience']:
                self.lr = max(config['min_lr'], config['factor'] * self.lr)
                print(f'Reduce lr on plateau to {self.lr}')
            self.last_loss = loss

        self.net.optimizer.lr = self.lr

    def get_optimizer(self):
        optimizer = self.config['optimizer']
        if optimizer.lower() == 'sgd':
            self.lr = self.config['optimizers']['SGD']['lr']
            self.momentum = self.config['optimizers']['SGD']['momentum']
            self.optimizer = tf.keras.optimizers.SGD(self.lr, self.momentum)
        elif optimizer.lower() == 'adam':
            self.lr = self.config['optimizers']['adam']['lr']
            self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.lr_init = self.lr

    def input_contains_nan(self, s, u):
        '''
        Checks for NaN in s and u. If there is nan, it clears the data buffer, since
        a missing value would lead to problems in the calculation of the delta values and the prediction.
        Having NaN values in the state is mostly a problem when running on the real car.
        '''
        contains_nan = False
        if np.isnan(np.sum(s)) or np.isnan(np.sum(u)):
            self.data_buffer.clear()
            contains_nan = True
        return contains_nan

    def step(self, s, u, time_control, updated_attributes):
        if self.s_previous is not None and self.u_previous is not None and not self.input_contains_nan(s, u):
            net_input = np.concatenate([u, s], axis=0)

            if self.use_diff_output:
                delta_s = (s - self.s_previous) / self.dt
                net_input = np.concatenate((net_input, delta_s), axis=0)

            # start = time.process_time()
            self.training_buffer.append(net_input)
            # print(f'Appending to buffer took: {time.process_time() - start}')

            # Retrain network
            path_to_net = self.predictor.predictor.net_info.path_to_net
            add_metrics = AddMetricsToLogger(self.N_step)
            csv_logger = keras.callbacks.CSVLogger(path_to_net + 'log_training.csv', append=True, separator=';')
            model_checkpoint_latest = keras.callbacks.ModelCheckpoint(
                        filepath=f'{path_to_net}/ckpt.ckpt',
                        save_weights_only=True,
                        monitor='val_loss',
                        mode='auto',
                        save_best_only=False)
            callbacks = [add_metrics, csv_logger, model_checkpoint_latest]

            if self.N_step % self.config['train_every_n_steps'] == 0 and self.training_buffer.full():
                tf.print(f'Doing training at step {self.N_step}')
                if self.training_step % self.config['save_net_every_n_training_steps'] == 0:
                    model_checkpoint_history = keras.callbacks.ModelCheckpoint(
                        filepath=f'{path_to_net}/log/step-{self.N_step}/ckpt.ckpt',
                        save_weights_only=True,
                        monitor='val_loss',
                        mode='auto',
                        save_best_only=False) 
                    callbacks.extend([model_checkpoint_history])
                self.hist = self.net.fit(self.training_buffer.get_data(),
                                         epochs=self.config['epochs_per_training'],
                                         callbacks=callbacks)
                # self.predictor.predictor.net = self.net  # Not needed, since net is loaded from disk in controller
                self.training_step += 1
                self.update_learning_rate()

        self.N_step += 1
        self.s_previous = s
        self.u_previous = u
