import os
import serial
import struct
import time

from SI_Toolkit.computation_library import TensorType, NumpyLibrary

import numpy as np

from Control_Toolkit.Controllers import template_controller

from Control_Toolkit.serial_interface_helper import get_serial_port, set_ftdi_latency_timer

try:
    from SI_Toolkit_ASF.ToolkitCustomization.predictors_customization import STATE_INDICES
except ModuleNotFoundError:
    print("SI_Toolkit_ASF not yet created")

from SI_Toolkit.Functions.General.Initialization import load_net_info_from_txt_file


class controller_fpga(template_controller):
    _computation_library = NumpyLibrary()

    def configure(self):


        SERIAL_PORT = get_serial_port(serial_port_number=self.config_controller["SERIAL_PORT"])
        SERIAL_BAUD = self.config_controller["SERIAL_BAUD"]
        set_ftdi_latency_timer(SERIAL_PORT=self.config_controller["SERIAL_PORT"])
        self.InterfaceInstance = Interface()
        self.InterfaceInstance.open(SERIAL_PORT, SERIAL_BAUD)

        NET_NAME = self.config_controller["net_name"]
        PATH_TO_MODELS = self.config_controller["PATH_TO_MODELS"]
        path_to_model_info = os.path.join(PATH_TO_MODELS, NET_NAME, NET_NAME + ".txt")

        self.input_at_input = self.config_controller["input_at_input"]

        self.net_info = load_net_info_from_txt_file(path_to_model_info)

        self.state_2_input_idx = []
        self.remaining_inputs = self.net_info.inputs.copy()
        for key in self.net_info.inputs:
            if key in STATE_INDICES.keys():
                self.state_2_input_idx.append(STATE_INDICES.get(key))
                self.remaining_inputs.remove(key)
            else:
                break  # state inputs must be adjacent in the current implementation

        self.just_restarted = True

        print('Configured fpga controller with {} network with {} library\n'.format(self.net_info.net_full_name, self.lib.lib))

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        self.just_restarted = False
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

        net_input = self.lib.reshape(net_input, (-1, 1, len(self.net_info.inputs)))
        net_input = self.lib.to_numpy(net_input)

        net_output = self.get_net_output_from_fpga(net_input)

        net_output = self.lib.to_tensor(net_output, self.lib.float32)
        net_output = net_output[self.lib.newaxis, self.lib.newaxis, :]

        if self.lib.lib == 'Pytorch':
            net_output = net_output.detach().numpy()

        Q = net_output

        return Q

    def controller_reset(self):

        if not self.just_restarted:
            self.configure()

    def get_net_output_from_fpga(self, net_input):
        self.InterfaceInstance.send_net_input(net_input)
        net_output = self.InterfaceInstance.receive_net_output(len(self.net_info.outputs))
        return net_output






PING_TIMEOUT            = 1.0       # Seconds
CALIBRATE_TIMEOUT       = 10.0      # Seconds
READ_STATE_TIMEOUT      = 1.0      # Seconds
SERIAL_SOF              = 0xAA
CMD_PING                = 0xC0

class Interface:
    def __init__(self):
        self.device         = None
        self.msg            = []
        self.start = None
        self.end = None

        self.encoderDirection = None

    def open(self, port, baud):
        self.port = port
        self.baud = baud
        self.device = serial.Serial(port, baudrate=baud, timeout=None)
        self.device.reset_input_buffer()

    def close(self):
        if self.device:
            time.sleep(2)
            self.device.close()
            self.device = None

    def clear_read_buffer(self):
        self.device.reset_input_buffer()

    def ping(self):
        msg = [SERIAL_SOF, CMD_PING, 4]
        msg.append(self._crc(msg))
        self.device.write(bytearray(msg))
        return self._receive_reply(CMD_PING, 4, PING_TIMEOUT) == msg

    def send_net_input(self, net_input):
        self.device.reset_output_buffer()
        bytes_written = self.device.write(bytearray(net_input))
        # print(bytes_written)

    def receive_net_output(self, net_output_length):
        net_output_length_bytes = net_output_length * 4  # We assume float32
        net_output = self.device.read(size=net_output_length_bytes)
        net_output = struct.unpack(f'<{net_output_length}f', net_output)
        # net_output=reply
        return net_output

    def _receive_reply(self, cmdLen, timeout=None, crc=True):
        self.device.timeout = timeout
        self.start = False

        while True:
            c = self.device.read()
            # Timeout: reopen device, start stream, reset msg and try again
            if len(c) == 0:
                print('\nReconnecting.')
                self.device.close()
                self.device = serial.Serial(self.port, baudrate=self.baud, timeout=timeout)
                self.clear_read_buffer()
                time.sleep(1)
                self.msg = []
                self.start = False
            else:
                self.msg.append(ord(c))
                if self.start == False:
                    self.start = time.time()

            while len(self.msg) >= cmdLen:
                # print('I am looping! Hurra!')
                # Message must start with SOF character
                if self.msg[0] != SERIAL_SOF:
                    #print('\nMissed SERIAL_SOF')
                    del self.msg[0]
                    continue

                # Check message packet length
                if self.msg[2] != cmdLen and cmdLen < 256:
                    print('\nWrong Packet Length.')
                    del self.msg[0]
                    continue

                # Verify integrity of message
                if crc and self.msg[cmdLen-1] != self._crc(self.msg[:cmdLen-1]):
                    print('\nCRC Failed.')
                    del self.msg[0]
                    continue

                self.device.timeout = None
                reply = self.msg[:cmdLen]
                del self.msg[:cmdLen]
                return reply

    def _crc(self, msg):
        crc8 = 0x00

        for i in range(len(msg)):
            val = msg[i]
            for b in range(8):
                sum = (crc8 ^ val) & 0x01
                crc8 >>= 1
                if sum > 0:
                    crc8 ^= 0x8C
                val >>= 1

        return crc8
