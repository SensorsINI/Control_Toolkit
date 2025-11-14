from __future__ import annotations
import numpy as np
import zmq
import zmq.error

from SI_Toolkit.computation_library import NumpyLibrary
from Control_Toolkit.Controllers import template_controller
from Control_Toolkit.others.globals_and_utils import import_controller_by_name

ENFORCE_TIMEOUT = True  # Set to False to disable the timeout feature
DEFAULT_RCVTIMEO = 50            # [ms]


class controller_remote(template_controller):
    _computation_library = NumpyLibrary()
    """
    ZeroMQ DEALER proxy.
    • Sends each state to the server together with a monotonically
      increasing *request-id* (`rid`).
    • Drops or purges every reply whose rid ≠ last request’s rid.
    • After a timeout the motor command falls back to 0 or to a local controller.
    """

    def configure(self):
        # ─── remote socket setup ────────────────────────────────────────
        self.endpoint = self.config_controller.get(
            "remote_endpoint", "tcp://localhost:5555"
        )
        self._ctx  = zmq.Context()
        self._sock = self._ctx.socket(zmq.DEALER)
        self._sock.connect(self.endpoint)
        if ENFORCE_TIMEOUT:
            self._sock.setsockopt(zmq.RCVTIMEO, DEFAULT_RCVTIMEO)

        self._next_rid: int = 0
        print(f"Neural-imitator proxy connected to {self.endpoint}")

        # ─── fallback to a local controller or 0 control ──────────────────────
        # retrieve fallback-controller parameters from config
        self.fallback_controller_name = self.config_controller["fallback_controller_name"]

        if self.fallback_controller_name is not None:
            # dynamically import and instantiate the local controller
            # e.g. import_controller_by_name("controller-neural-imitator")
            Controller = import_controller_by_name(
                f"controller-{self.fallback_controller_name}".replace("-", "_")
            )
            self._fallback_controller = Controller(
                self.environment_name, self.control_limits, self.initial_environment_attributes
            )
            self._fallback_controller.configure()

    # ------------------------------------------------------------------ STEP
    def step(
        self,
        s: np.ndarray,
        time=None,
        updated_attributes: "dict[str, np.ndarray]" = {},
    ):
        """
        Serialises the data, ships it to the server, waits up to 50 ms for Q,
        and returns it—or falls back on timeout to the fallback controller or zero control.
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
            self._purge_stale()            # clear the queue
            if self.fallback_controller_name is not None:
                # use local controller on timeout
                return self._fallback_controller.step(
                    s, time=time, updated_attributes=updated_attributes
                )
            return np.array(0.0, dtype=np.float32)

        # —— discard stale packets ——————————
        while resp.get("rid") != rid:
            try:
                resp = self._sock.recv_json()
            except zmq.error.Again:
                # genuine timeout – treat as lost reply
                if self.fallback_controller_name is not None:
                    return self._fallback_controller.step(
                        s, time=time, updated_attributes=updated_attributes
                    )
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
