#!/usr/bin/env python3
"""
remote_nn_controller_server.py

ZeroMQ ROUTER server that uses gui_selection to pick controller and optimizer,
then serves step requests.
"""

import sys
import numpy as np
import zmq
import json

from Control_Toolkit.controller_server.gui import choose_controller_and_optimizer
from Control_Toolkit.others.globals_and_utils import import_controller_by_name

# Hardcoded ZeroMQ endpoint
ENDPOINT = "tcp://*:5555"


initial_environment_attributes = {
    "target_position": 0.0,
    "target_equilibrium": 0.0,
    "m_pole": 0.0,
    "L": 0.0,
    "Q_ccrc": 0.0,
    "Q_applied_-1": 0.0,
}

def main():
    # Launch the GUI to get controller/optimizer
    ctrl_name, opt_name = choose_controller_and_optimizer()
    print(f"[server] ▶️  Controller: {ctrl_name}   Optimizer: {opt_name}")

    # Dynamically import & instantiate
    ControllerClass = import_controller_by_name(ctrl_name)
    ctrl = ControllerClass(
        environment_name="CartPole",
        control_limits=(-1.0, 1.0),
        initial_environment_attributes=initial_environment_attributes,  # populate as needed
    )

    # Configure with or without optimizer
    if ctrl.has_optimizer:
        ctrl.configure(optimizer_name=opt_name)
    else:
        ctrl.configure()

    # ─── ZeroMQ ROUTER socket ────────────────────────────────────────
    ctx  = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    sock.bind(ENDPOINT)
    print(f"[server] 🚀 listening on {ENDPOINT}")

    while True:
        # Receive either [identity, payload] or [identity, b"", payload]
        parts = sock.recv_multipart()
        if len(parts) == 2:
            client_identity, payload = parts
        elif len(parts) == 3 and parts[1] == b"":
            client_identity, _empty, payload = parts
        else:
            # Unexpected framing; skip it
            continue

        try:
            req   = json.loads(payload.decode("utf-8"))
            s   = np.asarray(req["state"], dtype=np.float32)
            t   = req.get("time", None)
            upd = req.get("updated_attributes", {})

            Q = ctrl.step(s, t, upd)
            reply = json.dumps({"Q": Q.tolist()}).encode("utf-8")

        except Exception as e:
            reply = json.dumps({"error": str(e)}).encode("utf-8")

        # ─── send back just two frames: [identity, reply] ───────────────
        # dropping the empty-delimiter ensures the DEALER's recv_json()
        # sees exactly one JSON frame.
        sock.send_multipart([client_identity, reply])


if __name__ == "__main__":
    main()
