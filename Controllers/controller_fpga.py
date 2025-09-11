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


class controller_fpga(template_controller):
    _computation_library = NumpyLibrary()

    def configure(self):

        SERIAL_PORT_NAME = get_serial_port(serial_port_number=self.config_controller["SERIAL_PORT"])
        SERIAL_BAUD = self.config_controller["SERIAL_BAUD"]
        print(f"DEBUG: Connecting to serial port: {SERIAL_PORT_NAME} at {SERIAL_BAUD} baud")
        set_ftdi_latency_timer(SERIAL_PORT_NAME)
        self.InterfaceInstance = Interface()
        self.InterfaceInstance.open(SERIAL_PORT_NAME, SERIAL_BAUD)
        print("DEBUG: Serial connection opened successfully")

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

        controller_output = self.get_controller_output_from_fpga(arr)          # raw float32 bytes over UART
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

    def get_controller_output_from_fpga(self, controller_input):
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

# New unified message types
MSG_TYPE_STATE      = 0x01    # State data for controller
MSG_TYPE_GET_SPEC   = 0x02    # Request controller specification
MSG_TYPE_PING       = 0x03    # Ping/keepalive
MSG_TYPE_SPEC_COOKIE = 0x04   # Announce spec change (FPGA->PC)

# Legacy constants for backward compatibility
CMD_PING                = 0xC0
CMD_GET_SPEC     = 0xC6
NAME_TOKEN_LEN    = 16     # fixed ASCII token length per name

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
        print(f"DEBUG: Opening serial device: {port} at {baud} baud")
        try:
            self.device = serial.Serial(port, baudrate=baud, timeout=None)
            print(f"DEBUG: Serial device opened successfully. Device info: {self.device}")
            self.device.reset_input_buffer()
            self.device.reset_output_buffer()
            print("DEBUG: Serial buffers reset")
        except Exception as e:
            print(f"DEBUG: ERROR opening serial device: {e}")
            raise

    def close(self):
        if self.device:
            time.sleep(2)
            self.device.close()
            self.device = None

    def clear_read_buffer(self):
        self.device.reset_input_buffer()

    def ping(self):
        print("DEBUG: Sending ping command...")
        msg = [SERIAL_SOF, MSG_TYPE_PING, 4]
        msg.append(self._crc(msg))
        print(f"DEBUG: Ping message: {[hex(b) for b in msg]}")
        self.device.write(bytearray(msg))
        
        # Simple ping response check - just wait for 4 bytes
        old_timeout = self.device.timeout
        try:
            self.device.timeout = PING_TIMEOUT
            response = self.device.read(4)
            print(f"DEBUG: Ping response: {len(response)} bytes - {[hex(b) for b in response] if response else 'None'}")
            
            if len(response) == 4 and response == bytearray(msg):
                print("DEBUG: Ping successful - exact match")
                return True
            else:
                print("DEBUG: Ping failed - no response or mismatch")
                return False
        finally:
            self.device.timeout = old_timeout

    def get_spec(self):
        """
        Request SoC declaration of its input wire-order and output count.

        SoC reply (raw, no frame): 4-byte header + names block
          byte 0: version (u8)
          byte 1: n_inputs (u8)
          byte 2: n_outputs (u8)
          byte 3: token_len (u8) == NAME_TOKEN_LEN
          bytes 4.. : n_inputs * token_len ASCII names (NUL-padded), wire order
        """
        print("DEBUG: Starting GET_SPEC handshake...")
        
        # Retry logic for startup synchronization
        max_retries = 10
        retry_delay = 0.5  # seconds
        
        for attempt in range(max_retries):
            print(f"DEBUG: Handshake attempt {attempt + 1}/{max_retries}")
            self.clear_read_buffer()
            
            # Send framed request using new unified protocol
            msg = bytearray([SERIAL_SOF, MSG_TYPE_GET_SPEC, 4])
            msg.append(self._crc(msg))
            print(f"DEBUG: Sending GET_SPEC command: {[hex(b) for b in msg]}")
            self.device.write(msg)
            print("DEBUG: GET_SPEC command sent, waiting for response...")

            # Handshake is a control exchange: use a bounded timeout so we fail fast instead of hanging.
            old_timeout = self.device.timeout
            try:
                self.device.timeout = 1.0
                hdr = self.device.read(4)
                print(f"DEBUG: Received header bytes: {len(hdr)} bytes - {[hex(b) for b in hdr] if hdr else 'None'}")
                
                if len(hdr) != 4:
                    print(f"DEBUG: ERROR - Expected 4 bytes, got {len(hdr)}")
                    # Try to read more data to see what we actually got
                    additional = self.device.read(100)  # Try to read more
                    print(f"DEBUG: Additional data available: {len(additional)} bytes - {[hex(b) for b in additional] if additional else 'None'}")
                    
                    if attempt < max_retries - 1:
                        print(f"DEBUG: Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise IOError("GET_SPEC: short header")
                    
                version, n_inputs, n_outputs, token_len = hdr[0], hdr[1], hdr[2], hdr[3]
                print(f"DEBUG: Header parsed - version={version}, n_inputs={n_inputs}, n_outputs={n_outputs}, token_len={token_len}")
                
                if token_len != NAME_TOKEN_LEN:
                    print(f"DEBUG: ERROR - token_len mismatch: got {token_len}, expected {NAME_TOKEN_LEN}")
                    if attempt < max_retries - 1:
                        print(f"DEBUG: Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise IOError(f"GET_SPEC: unexpected token_len={token_len} (expected {NAME_TOKEN_LEN})")

                need = n_inputs * token_len
                print(f"DEBUG: Need to read {need} bytes for input names...")
                raw = self.device.read(need)
                print(f"DEBUG: Received names block: {len(raw)} bytes")
                
                if len(raw) != need:
                    print(f"DEBUG: ERROR - Expected {need} bytes for names, got {len(raw)}")
                    if attempt < max_retries - 1:
                        print(f"DEBUG: Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise IOError("GET_SPEC: short names block")

                names = []
                for i in range(n_inputs):
                    chunk = raw[i*token_len:(i+1)*token_len]
                    # Cut at first NUL; ignore non-ASCII silently.
                    name = chunk.split(b'\x00', 1)[0].decode('ascii', errors='ignore')
                    names.append(name)
                    print(f"DEBUG: Input name {i}: '{name}'")
                    
                print(f"DEBUG: Handshake successful - {len(names)} inputs, {n_outputs} outputs")
                return version, names, n_outputs
            finally:
                self.device.timeout = old_timeout  # restore streaming behavior
        
        # If we get here, all retries failed
        raise IOError("GET_SPEC: All retry attempts failed")

    def send_controller_input(self, controller_input):
        self.device.reset_output_buffer()
        if not isinstance(controller_input, np.ndarray) or controller_input.dtype != np.float32:
            controller_input = np.asarray(controller_input, dtype=np.float32)
        
        # Create state message with new unified format
        data_bytes = controller_input.tobytes()
        msg_length = 4 + len(data_bytes)  # SOF + type + length + data + CRC
        msg = bytearray([SERIAL_SOF, MSG_TYPE_STATE, msg_length])
        msg.extend(data_bytes)
        msg.append(self._crc(msg))
        
        self.device.write(msg)

    def receive_controller_output(self, controller_output_length):
        """
        Reads controller outputs. With the new unified protocol, the FPGA automatically
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
            print(f"Data length: {len(data)}, Expected: {nbytes}")
            print(f"Data bytes: {[hex(b) for b in data[:16]]}...")  # Show first 16 bytes
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