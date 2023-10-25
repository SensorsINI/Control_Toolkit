import tensorflow as tf
from tensorflow import keras
import numpy as np
from SI_Toolkit.Functions.TF.Loss import loss_msr_sequence_customizable
from SI_Toolkit.computation_library import TensorFlowLibrary
from SI_Toolkit.Functions.General.Normalising import get_denormalization_function, get_normalization_function

import shutil
import os


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

        if self.predictor.predictor.net_info.net_type == 'Dense':
            self.predictor.predictor.net.compile(
                loss=loss_msr_sequence_customizable(wash_out_len=0,
                                                    post_wash_out_len=1,
                                                    discount_factor=1.0),
                optimizer=keras.optimizers.Adam(5e-3) # need to be further designed
            )
            self.flag_train_at_each_step = False
            if not self.flag_train_at_each_step:
                self.bs = 16
                self.duration = 8

        elif self.predictor.predictor.net_info.net_type == 'GRU':
            self.predictor.predictor.net_copy.compile(
                loss=loss_msr_sequence_customizable(wash_out_len=10,
                                                    post_wash_out_len=20,
                                                    discount_factor=1.0),
                optimizer=keras.optimizers.Adam(3e-4) # need to be further designed
            )
            self.bs = 16
            self.duration = 5

        self.save_model_flag = False

        self.input_dataset = None
        self.output_dataset = None

        self.near_balance = None

    def step(self, s, angle, u, time, updated_attributes):
        u_norm = self.predictor.predictor.normalize_control_inputs(u)
        s_norm = self.predictor.predictor.normalize_inputs(s)

        if self.s_norm_previous is not None and self.u_norm_previous is not None:
            if self.predictor.predictor.net_info.net_type == 'Dense':
                net_input = tf.concat([self.u_norm_previous, self.s_norm_previous], 0) # This order is neede for normalization
                net_input = tf.reshape(net_input, [1, 1, len(net_input)])

                if np.any(['D_' in output_name for output_name in self.predictor.predictor.net_info.outputs]):
                    delta_s = (s - self.s_previous) / self.dt
                    delta_s = tf.reshape(delta_s, [1, len(delta_s)])
                    delta_s_norm = self.normalize_outputs(delta_s)
                    s_measured = delta_s_norm
                else:
                    s_measured = s_norm

                s_measured = tf.reshape(s_measured, [1, 1, len(s_measured)])

                if self.flag_train_at_each_step:
                    self.predictor.predictor.net.fit(net_input, s_measured, batch_size=1, epochs=1)
                else:
                    if self.input_dataset is None:
                        self.input_dataset = net_input
                        self.output_dataset = s_measured
                    else:
                        self.input_dataset = tf.concat([self.input_dataset,net_input],0)
                        self.output_dataset = tf.concat([self.output_dataset,s_measured],0)
                    
                    if self.input_dataset.shape[0]==50*self.duration:

                        n_batch = (50*self.duration)//self.bs
                        n_used_samples = n_batch * self.bs
                        self.predictor.predictor.net.fit(self.input_dataset[:n_used_samples], self.output_dataset[:n_used_samples], batch_size=self.bs, epochs=15)

                        self.input_dataset = None
                        self.output_dataset = None
            
            elif self.predictor.predictor.net_info.net_type == 'GRU':
                net_input = tf.concat([self.u_norm_previous, self.s_norm_previous], 0) # This order is neede for normalization
                net_input = tf.reshape(net_input, [1, 1, len(net_input)])

                if np.any(['D_' in output_name for output_name in self.predictor.predictor.net_info.outputs]):
                    delta_s = (s - self.s_previous) / self.dt
                    delta_s = tf.reshape(delta_s, [1, len(delta_s)])
                    delta_s_norm = self.normalize_outputs(delta_s)
                    s_measured = delta_s_norm
                else:
                    s_measured = s_norm

                s_measured = tf.reshape(s_measured, [1, 1, len(s_measured)])

                if self.input_dataset is None:
                    self.input_dataset = net_input
                    self.output_dataset = s_measured
                else:
                    self.input_dataset = tf.concat([self.input_dataset,net_input],0)
                    self.output_dataset = tf.concat([self.output_dataset,s_measured],0)

                # if abs(angle)<3.14/36:
                #     dp = tf.concat([net_input,s_measured],2)
                #     if self.near_balance is None:
                #         self.near_balance = dp
                #     else:
                #         self.near_balance = tf.concat([self.near_balance,dp],0)
                
                if self.input_dataset.shape[0]==50*self.duration:

                    # if self.near_balance is None:
                    #     input = self.input_dataset
                    #     output = self.output_dataset
                    # elif self.near_balance.shape[0]<=250:
                    #     input = tf.concat([self.input_dataset, self.near_balance[:,:,:6]], 0)
                    #     output = tf.concat([self.output_dataset, self.near_balance[:,:,6:]], 0)
                    # else:
                    #     tf.random.shuffle(self.near_balance)
                    #     input = tf.concat([self.input_dataset, self.near_balance[:250,:,:6]], 0)
                    #     output = tf.concat([self.output_dataset, self.near_balance[:250,:,6:]], 0)

                    n_samples = 50*self.duration-30+1
                    x_samples = []
                    y_samples = []
                    for i in range(n_samples):
                        ele_x = [self.input_dataset[p] for p in range(i,i+30)]
                        ele_y = [self.output_dataset[p] for p in range(i,i+30)]
                        x_samples.append(tf.expand_dims(tf.concat(ele_x,0),axis=0))
                        y_samples.append(tf.expand_dims(tf.concat(ele_y,0),axis=0))

                    n_batch = n_samples//self.bs
                    n_used_samples = n_batch * self.bs
                    x_in = tf.concat(x_samples[:n_used_samples],0)
                    y_target = tf.concat(y_samples[:n_used_samples],0)

                    self.predictor.predictor.net_copy.fit(x_in, y_target, batch_size=self.bs, epochs=10)

                    self.predictor.predictor.net_copy.save_weights("model_temp/ckpt.ckpt")
                    # print('Hello! Before reloading:')
                    # for i,l in enumerate(self.predictor.predictor.net.layers):
                    #     print('{}:'.format(i+1))
                    #     print(l.weights)
                    #     if i<2:
                    #         print(l.states)
                    self.predictor.predictor.net.load_weights("model_temp/ckpt.ckpt").expect_partial()
                    # print('Hello! After reloading:')
                    # for i,l in enumerate(self.predictor.predictor.net.layers):
                    #     print('{}:'.format(i+1))
                    #     print(l.weights)
                    #     if i<2:
                    #         print(l.states)
                    shutil.rmtree("model_temp")
                    os.mkdir("model_temp")

                    self.input_dataset = None
                    self.output_dataset = None

                    # if self.near_balance.shape[0]>5000:
                    #     self.near_balance = self.near_balance[250:,:,:]

        self.s_norm_previous = s_norm
        self.u_norm_previous = u_norm

        self.s_previous = s
        self.u_previous = u
        print("Doing online learning now!")
        print("time recorder:")
        print(time)
        if time>72 and self.save_model_flag:
            self.predictor.predictor.net.save_weights("model_retrain/ckpt.ckpt")
            self.save_model_flag = False