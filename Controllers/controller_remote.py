from __future__ import annotations
import numpy as np
import zmq
import zmq.error

from SI_Toolkit.computation_library import NumpyLibrary
from Control_Toolkit.Controllers import template_controller


ENFORCE_TIMEOUT = True  # Set to False to disable the timeout feature
DEFAULT_RCVTIMEO = 50            # [ms]


class controller_remote(template_controller):
    _computation_library = NumpyLibrary()
    """
    ZeroMQ DEALER proxy.
    • Sends each state to the server together with a monotonically
      increasing *request-id* (`rid`).
    • Drops or purges every reply whose rid ≠ last request’s rid.
    • After a timeout the motor command falls back to 0.
    """

    def configure(self):
        # You still keep the same config dict keys as in your local controller
        self.endpoint = self.config_controller.get(
            "remote_endpoint", "tcp://localhost:5555"
        )

        # A ZeroMQ DEALER socket is perfectly fine for synchronous request-reply.
        self._ctx  = zmq.Context()
        self._sock = self._ctx.socket(zmq.DEALER)
        self._sock.connect(self.endpoint)

        # ─── impose a 50 ms receive deadline ──────────────────────────────
        if ENFORCE_TIMEOUT:
            self._sock.setsockopt(zmq.RCVTIMEO, DEFAULT_RCVTIMEO)

        self._next_rid: int = 0      # starts at 0, increments each step
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
        if updated_attributes is None:
            updated_attributes = {}

        rid = self._next_rid         # snapshot current rid
        self._next_rid += 1          # prepare for next call

        self._sock.send_json(
            {
                "rid": rid,
                "state": s.tolist(),  # JSON-friendly
                "time": time,
                "updated_attributes": updated_attributes,
            }
        )

        # ❷ -- receive with timeout
        try:
            resp = self._sock.recv_json()  # may raise zmq.Again
        except zmq.error.Again:
            self._purge_stale()             # empty the queue
            return np.array(0.0, dtype=np.float32)

        # —— discard stale packets ——————————
        while resp.get("rid") != rid:
            try:
                # block (up to RCVTIMEO) for the *right* reply
                resp = self._sock.recv_json()  # ← remove DONTWAIT
            except zmq.error.Again:
                # genuine timeout – treat as lost reply
                return np.array(0.0, dtype=np.float32)

        if "error" in resp:
            # Re-raise server-side exceptions locally for easier debugging
            raise RuntimeError(f"Remote controller error: {resp['error']}")

        # ❸ -- final result
        return np.asarray(resp["Q"], dtype=np.float32)

    # ---------------------------------------------------------- helpers
    def _purge_stale(self) -> None:
        """Discard every pending message in the inbound queue."""
        while True:
            try:
                self._sock.recv(flags=zmq.DONTWAIT)
            except zmq.error.Again:
                break

    # ---------------------------------------------------------------- RESET
    def controller_reset(self):
        """
        Nothing to reset locally; the server keeps the network state.
        """
        pass
