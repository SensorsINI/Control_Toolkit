from SI_Toolkit.computation_library import NumpyLibrary
from Control_Toolkit.Controllers import template_controller
import numpy as np
import zmq
import zmq.error

ENFORCE_TIMEOUT = True  # Set to False to disable the timeout feature

class controller_remote(template_controller):
    _computation_library = NumpyLibrary()
    """
    Thin proxy: forwards inputs to a remote process and returns its output.
    """

    def configure(self):
        # You still keep the same config dict keys as in your local controller
        self.endpoint = self.config_controller.get(
            "remote_endpoint", "tcp://localhost:5555"
        )

        # A ZeroMQ REQ socket is perfectly fine for synchronous request-reply.
        self._ctx  = zmq.Context()
        self._sock = self._ctx.socket(zmq.DEALER)
        self._sock.connect(self.endpoint)

        # ─── impose a 50 ms receive deadline ──────────────────────────────
        if ENFORCE_TIMEOUT:
            self._sock.setsockopt(zmq.RCVTIMEO, 50)

        print(f"Neural-imitator proxy connected to {self.endpoint}")

    # ------------------------------------------------------------------ STEP
    def step(
        self,
        s: np.ndarray,
        time=None,
        updated_attributes: "dict[str, np.ndarray]" = {},
    ):
        """
        Serialises the data, ships it to the server, waits up to 50 ms for Q,
        and returns it—or zeros if the server doesn’t reply in time.
        """
        # ❶ -- send (always non-blocking for REQ)
        self._sock.send_json(
            {
                "state": s.tolist(),  # JSON-friendly
                "time": time,
                "updated_attributes": updated_attributes,
            }
        )

        # ❷ -- receive with timeout
        try:
            resp = self._sock.recv_json()  # may raise zmq.Again after 50 ms
        except zmq.error.Again:
            # no reply within 50 ms → default to zero output
            # np.zeros_like(s) preserves the expected shape; cast to float32
            return np.array(0.0, dtype=np.float32)

        if "error" in resp:
            # Re-raise server-side exceptions locally for easier debugging
            raise RuntimeError(f"Remote controller error: {resp['error']}")

        # ❸ -- final result
        return np.asarray(resp["Q"], dtype=np.float32)

    # ---------------------------------------------------------------- RESET
    def controller_reset(self):
        """
        Nothing to reset locally; the server keeps the network state.
        """
        pass
