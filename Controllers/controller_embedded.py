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
    raise Exception("SI_Toolkit_ASF not yet created")


class controller_embedded(template_controller):
    _computation_library = NumpyLibrary()

    def configure(self):

        SERIAL_PORT_NAME = get_serial_port(serial_port_number=self.config_controller["SERIAL_PORT"])
        SERIAL_BAUD = self.config_controller["SERIAL_BAUD"]
        set_ftdi_latency_timer(SERIAL_PORT_NAME)
        self.InterfaceInstance = Interface()
        self.InterfaceInstance.open(SERIAL_PORT_NAME, SERIAL_BAUD)

        # --- PC↔SoC handshake: SoC declares input names and output count ---
        self.spec_version, self.input_names, self.n_outputs = self.InterfaceInstance.get_spec()

        self._state_idx = dict(STATE_INDICES)

        self.just_restarted = True

        print('Configured SoC controller (spec v{}) with {} library\n'.format(self.spec_version, self.lib.lib))

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = None):
        self.just_restarted = False
        if updated_attributes is None:
            updated_attributes = {}
        self.update_attributes(updated_attributes)

        # Build inputs *exactly* in the wire order requested by the SoC.
        # Precedence: updated_attributes > state vector > variable_parameters > 0.0
        arr = np.empty(len(self.input_names), dtype=np.float32)
        for i, name in enumerate(self.input_names):
            if name == "time":
                if time is None:
                    raise Exception("Controller input 'time' is required but not provided.")
                else:
                    val = float(time)   # use simulator's timestamp (seconds, monotonic in sim time)
                arr[i] = val
                continue

            if name in updated_attributes:                       # external override wins
                val = float(updated_attributes[name])
            elif name in self._state_idx:                        # pick from s by name→index map
                val = float(s[..., self._state_idx[name]])
            elif hasattr(self, 'variable_parameters') and hasattr(self.variable_parameters, name):
                val = float(getattr(self.variable_parameters, name))
            else:
                val = 0.0                                        # explicit default to prevent UB
            arr[i] = val

        controller_output = self.get_controller_output_from_chip(arr)          # raw float32 bytes over UART
        controller_output = self.lib.to_tensor(controller_output, self.lib.float32)
        controller_output = controller_output[self.lib.newaxis, self.lib.newaxis, :]
        controller_output = self.lib.nan_to_num(controller_output, nan=0.0)

        if self.lib.lib == 'Pytorch':
            controller_output = controller_output.detach().numpy()

        Q = controller_output

        return Q

    def controller_reset(self):

        if not self.just_restarted:
            self.configure()

    def get_controller_output_from_chip(self, controller_input):
        self.InterfaceInstance.send_controller_input(controller_input)
        controller_output = self.InterfaceInstance.receive_controller_output(self.n_outputs)

        # if a cookie-triggered GET_SPEC happened, adopt it for NEXT step
        if self.InterfaceInstance.pending_spec is not None:
            self.spec_version, self.input_names, self.n_outputs = self.InterfaceInstance.pending_spec
            self.InterfaceInstance.pending_spec = None
            print(f"Refreshed SoC spec (v{self.spec_version}): "
                  f"{len(self.input_names)} inputs, {self.n_outputs} outputs")

        return controller_output




PING_TIMEOUT            = 1.0       # Seconds
READ_STATE_TIMEOUT      = 1.0      # Seconds
SERIAL_SOF              = 0xAA

# Unified protocol message types
MSG_TYPE_STATE      = 0x01    # State data for controller
MSG_TYPE_GET_SPEC   = 0x02    # Request controller specification
MSG_TYPE_PING       = 0x03    # Ping/keepalive
MSG_TYPE_SPEC_COOKIE = 0x04   # Announce spec change (CHIP->PC)

NAME_TOKEN_LEN    = 24     # fixed ASCII token length per name

class Interface:
    def __init__(self):
        self.device         = None
        self.msg            = []
        self.start = None
        self.end = None

        self.encoderDirection = None

        self.pending_spec = None

    def open(self, port, baud):
        self.port = port
        self.baud = baud
        try:
            self.device = serial.Serial(port, baudrate=baud, timeout=None)
            self.device.reset_input_buffer()
            self.device.reset_output_buffer()
        except Exception as e:
            print(f"ERROR opening serial device: {e}")
            raise

    def close(self):
        if self.device:
            time.sleep(2)
            self.device.close()
            self.device = None

    def clear_read_buffer(self):
        self.device.reset_input_buffer()
    
    def read_available_data(self, timeout=0.1):
        """Read any available data from the device for debugging"""
        old_timeout = self.device.timeout
        try:
            self.device.timeout = timeout
            data = self.device.read(1024)  # Read up to 1KB
            return data
        finally:
            self.device.timeout = old_timeout

    def ping(self):
        msg = [SERIAL_SOF, MSG_TYPE_PING, 4]
        msg.append(self._crc(msg))
        self.device.write(bytearray(msg))
        
        # Simple ping response check - just wait for 4 bytes
        old_timeout = self.device.timeout
        try:
            self.device.timeout = PING_TIMEOUT
            response = self.device.read(4)
            
            # Check for unified protocol ping response: [SOF, MSG_TYPE_PING, 4, CRC]
            if len(response) == 4 and response[0] == SERIAL_SOF and response[1] == MSG_TYPE_PING and response[2] == 4:
                return True
            else:
                return False
        finally:
            self.device.timeout = old_timeout

    def get_spec(self):
        """
        Request SoC declaration of its input wire-order and output count.
        Uses the unified protocol with MSG_TYPE_GET_SPEC = 0x02
        """
        # First check if there's any data available from the device
        available_data = self.read_available_data()
        
        # First try to ping the device to see if it's responsive
        if not self.ping():
            pass  # Continue anyway
        
        # Retry logic for startup synchronization
        max_retries = 3
        retry_delay = 0.5  # seconds
        
        for attempt in range(max_retries):
            self.clear_read_buffer()
            
            # Send framed request using unified protocol (MSG_TYPE_GET_SPEC = 0x02)
            msg = bytearray([SERIAL_SOF, MSG_TYPE_GET_SPEC, 4])
            msg.append(self._crc(msg))
            self.device.write(msg)

            # Handshake is a control exchange: use a bounded timeout so we fail fast instead of hanging.
            old_timeout = self.device.timeout
            try:
                self.device.timeout = 2.0
                hdr = self.device.read(4)
                
                if len(hdr) != 4:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        break
                    
                version, n_inputs, n_outputs, token_len = hdr[0], hdr[1], hdr[2], hdr[3]
                
                # Check if we got valid data
                if token_len == NAME_TOKEN_LEN and n_inputs > 0 and n_outputs > 0:
                    need = n_inputs * token_len
                    raw = self.device.read(need)
                    
                    if len(raw) == need:
                        names = []
                        for i in range(n_inputs):
                            chunk = raw[i*token_len:(i+1)*token_len]
                            # Cut at first NUL; ignore non-ASCII silently.
                            name = chunk.split(b'\x00', 1)[0].decode('ascii', errors='ignore')
                            
                            # Assert that input name is not longer than our buffer
                            if len(name) > NAME_TOKEN_LEN:
                                raise ValueError(f"Input name '{name}' is {len(name)} characters long, but NAME_TOKEN_LEN is only {NAME_TOKEN_LEN}")
                            
                            names.append(name)
                            
                        return version, names, n_outputs
                
                # If we get here, the firmware response is invalid
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    break
                    
            finally:
                self.device.timeout = old_timeout  # restore streaming behavior
        
        # Use hardcoded specifications for neural imitator controller
        hardcoded_version = 1
        hardcoded_input_names = [
            "angleD", "angle_cos", "angle_sin", "position", "positionD",
            "target_equilibrium", "target_position"
        ]
        hardcoded_n_outputs = 1
        
        return hardcoded_version, hardcoded_input_names, hardcoded_n_outputs

    def send_controller_input(self, controller_input):
        self.device.reset_output_buffer()
        if not isinstance(controller_input, np.ndarray) or controller_input.dtype != np.float32:
            controller_input = np.asarray(controller_input, dtype=np.float32)
        
        # Use unified protocol with MSG_TYPE_STATE = 0x01
        data_bytes = controller_input.tobytes()
        msg_length = 4 + len(data_bytes)  # SOF + type + length + data + CRC
        
        # Build message: [SOF, MSG_TYPE_STATE, length, data..., CRC]
        msg = bytearray([SERIAL_SOF, MSG_TYPE_STATE, msg_length])
        msg.extend(data_bytes)
        msg.append(self._crc(msg))
        
        self.device.write(msg)

    def receive_controller_output(self, controller_output_length):
        """
        Reads controller outputs. With the new unified protocol, the chip automatically
        sends controller outputs when it receives state data, so we just need to read
        the raw float data directly.
        """
        # Read the expected number of float32 bytes directly
        nbytes = controller_output_length * 4
        data = self.device.read(size=nbytes)
        if len(data) != nbytes:
            raise IOError(f"receive_controller_output: expected {nbytes} bytes, got {len(data)}")
        
        # Unpack the float32 data
        try:
            result = struct.unpack(f'<{controller_output_length}f', data)
            return result
        except struct.error as e:
            print(f"ERROR: Failed to unpack controller output data: {e}")
            # Return zeros as fallback
            return (0.0,) * controller_output_length

    def _receive_reply(self, cmdLen, timeout=None, crc=True):
        self.device.timeout = timeout
        self.start = False
        self.msg = []

        while True:
            c = self.device.read(1)
            if len(c) == 0:
                print('\nReconnecting.')
                self.device.close()
                self.device = serial.Serial(self.port, baudrate=self.baud, timeout=timeout)
                self.clear_read_buffer()
                time.sleep(1)
                self.msg = []
                self.start = False
            else:
                # Py3: bytes→int via c[0]; ord() on bytes is a TypeError.
                self.msg.append(c[0])
                if self.start is False:
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