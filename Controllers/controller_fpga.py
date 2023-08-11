import serial
import struct
import time

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


class controller_fpga(template_controller):
    _computation_library = NumpyLibrary

    def configure(self):


        SERIAL_PORT = get_serial_port(serial_port_number=self.config_controller["SERIAL_PORT"])
        SERIAL_BAUD = self.config_controller["SERIAL_BAUD"]

        self.InterfaceInstance = Interface()
        self.InterfaceInstance.open(SERIAL_PORT, SERIAL_BAUD)

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

        self.state_2_input_idx = []
        self.remaining_inputs = self.net_info.inputs.copy()
        for key in self.net_info.inputs:
            if key in STATE_INDICES.keys():
                self.state_2_input_idx.append(STATE_INDICES.get(key))
                self.remaining_inputs.remove(key)
            else:
                break  # state inputs must be adjacent in the current implementation

        self.net_inputs_buffer = []
        self.net_outputs_buffer = []

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

        self.lib.assign(self.net_input_normed, self.normalize_inputs(net_input))

        net_input = self.lib.reshape(self.net_input_normed, (-1, 1, len(self.net_info.inputs)))
        net_input = self.lib.to_numpy(net_input)

        net_output = self.get_net_output_from_fpga(net_input)
        net_output = np.flip(net_output)
        net_output = self.lib.to_tensor(net_output, self.lib.float32)

        net_output = self.denormalize_outputs(net_output)

        if self.lib.lib == 'Pytorch':
            net_output = net_output.detach().numpy()

        Q = net_output

        return Q

    def controller_reset(self):
        if self.net_inputs_buffer and len(self.net_inputs_buffer) > 1:
            net_inputs = self.lib.stack(self.net_inputs_buffer)
            net_outputs = self.lib.stack(self.net_outputs_buffer)

            net_inputs = self.lib.to_numpy(net_inputs)
            net_outputs = self.lib.to_numpy(net_outputs)

            np.savetxt('net_inputs.csv', net_inputs, delimiter=',')
            np.savetxt('net_outputs.csv', net_outputs, delimiter=',')

        self.configure()

    def get_net_output_from_fpga(self, net_input):
        self.InterfaceInstance.send_net_input(net_input)
        net_output = self.InterfaceInstance.receive_net_output()
        return net_output


def get_serial_port(serial_port_number=''):
    import platform
    import subprocess
    serial_port_number = str(serial_port_number)
    SERIAL_PORT = None
    try:
        system = platform.system()
        if system == 'Darwin':  # Mac
            SERIAL_PORT = subprocess.check_output('ls -a /dev/tty.usbserial*', shell=True).decode("utf-8").strip()  # Probably '/dev/tty.usbserial-110'
        elif system == 'Linux':
            SERIAL_PORT = '/dev/ttyUSB' + serial_port_number  # You might need to change the USB number
        elif system == 'Windows':
            SERIAL_PORT = 'COM' + serial_port_number
        else:
            raise NotImplementedError('For system={} connection to serial port is not implemented.')
    except Exception as err:
        print(err)

    return SERIAL_PORT




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
        self.device.flush()
        return self._receive_reply(CMD_PING, 4, PING_TIMEOUT) == msg

    def send_net_input(self, net_input):
        self.device.reset_output_buffer()
        # self.clear_read_buffer()
        # msg = [SERIAL_SOF, net_input]
        # msg.append(self._crc(msg))
        bytes_written = self.device.write(bytearray(net_input))
        # print(bytes_written)
        self.device.flush()

    def receive_net_output(self):
        # self.clear_read_buffer()
        # reply = self._receive_reply(4, READ_STATE_TIMEOUT)
        net_outputs = []
        for i in range(2):
            net_output = self.device.read(size=4)
            # net_output = struct.unpack('=3hBIH', bytes(net_output[3:16]))
            net_output = struct.unpack('<f', net_output)
            net_outputs.append(net_output[0])
        net_outputs = np.array(net_outputs)
        # net_output=reply
        return net_outputs

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