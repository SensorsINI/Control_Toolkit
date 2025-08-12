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


        SERIAL_PORT = get_serial_port(serial_port_number=self.config_controller["SERIAL_PORT"])
        SERIAL_BAUD = self.config_controller["SERIAL_BAUD"]
        set_ftdi_latency_timer(SERIAL_PORT=self.config_controller["SERIAL_PORT"])
        self.InterfaceInstance = Interface()
        self.InterfaceInstance.open(SERIAL_PORT, SERIAL_BAUD)

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
        self.device = serial.Serial(port, baudrate=baud, timeout=None)
        self.device.reset_input_buffer()
        self.device.reset_output_buffer()

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
        return self._receive_reply(4, PING_TIMEOUT) == msg

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
        self.clear_read_buffer()
        # Send framed request (SOF, CMD, LEN, CRC) to stay consistent with existing protocol.
        msg = bytearray([SERIAL_SOF, CMD_GET_SPEC, 4])
        msg.append(self._crc(msg))
        self.device.write(msg)

        # Handshake is a control exchange: use a bounded timeout so we fail fast instead of hanging.
        old_timeout = self.device.timeout
        try:
            self.device.timeout = 1.0
            hdr = self.device.read(4)
            if len(hdr) != 4:
                raise IOError("GET_SPEC: short header")
            version, n_inputs, n_outputs, token_len = hdr[0], hdr[1], hdr[2], hdr[3]
            if token_len != NAME_TOKEN_LEN:
                raise IOError(f"GET_SPEC: unexpected token_len={token_len} (expected {NAME_TOKEN_LEN})")

            need = n_inputs * token_len
            raw = self.device.read(need)
            if len(raw) != need:
                raise IOError("GET_SPEC: short names block")

            names = []
            for i in range(n_inputs):
                chunk = raw[i*token_len:(i+1)*token_len]
                # Cut at first NUL; ignore non-ASCII silently.
                names.append(chunk.split(b'\x00', 1)[0].decode('ascii', errors='ignore'))
            return version, names, n_outputs
        finally:
            self.device.timeout = old_timeout  # restore streaming behavior

    def send_controller_input(self, controller_input):
        self.device.reset_output_buffer()
        if not isinstance(controller_input, np.ndarray) or controller_input.dtype != np.float32:
            controller_input = np.asarray(controller_input, dtype=np.float32)
        self.device.write(controller_input.tobytes())

    def receive_controller_output(self, controller_output_length):
        """
        Reads controller outputs. If a spec-change cookie arrives, we immediately
        re-handshake (GET_SPEC) for the next cycle, then still read and return
        THIS cycle's outputs (old spec) so the control loop doesn't stall.
        """
        # Peek first 4 bytes
        head = self.device.read(size=4)
        if len(head) != 4:
            raise IOError(f"receive_controller_output: expected 4 bytes head, got {len(head)}")

        # Check for spec-change cookie: [SOF, CMD_SPEC_COOKIE, gen, CRC]
        if head[0] == SERIAL_SOF and head[1] == 0xC7 and head[3] == self._crc(head[:3]):
            # Re-handshake now so *next* step uses the new spec
            version, names, n_outputs = self.get_spec()
            # Stash for the controller to pick up after this receive
            self.pending_spec = (version, names, n_outputs)
            # Now read THIS cycle's outputs (old spec) and return them
            nbytes = controller_output_length * 4
            data = self.device.read(size=nbytes)
            if len(data) != nbytes:
                raise IOError(f"receive_controller_output: expected {nbytes} bytes after cookie, got {len(data)}")
            return struct.unpack(f'<{controller_output_length}f', data)

        # No cookie: head belongs to outputs; read the rest
        rest_bytes = controller_output_length * 4 - 4
        if rest_bytes < 0:
            raise ValueError("controller_output_length must be >= 1")
        rest = self.device.read(size=rest_bytes) if rest_bytes else b""
        if len(rest) != rest_bytes:
            raise IOError(f"receive_controller_output: expected {rest_bytes} tail bytes, got {len(rest)}")
        data = head + rest
        return struct.unpack(f'<{controller_output_length}f', data)

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
