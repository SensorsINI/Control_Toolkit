#!/usr/bin/env python3
"""
remote_nn_controller_server.py

ZeroMQ REP server that uses gui_selection to pick controller and optimizer,
then serves step requests.
"""

import sys
import numpy as np
import zmq

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

    # ZeroMQ REP socket
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(ENDPOINT)
    print(f"[server] 🚀 listening on {ENDPOINT}")

    while True:
        req = sock.recv_json()
        try:
            s   = np.asarray(req["state"], dtype=np.float32)
            t   = req.get("time", None)
            upd = req.get("updated_attributes", {})

            Q = ctrl.step(s, t, upd)
            sock.send_json({"Q": Q.tolist()})

        except Exception as e:
            sock.send_json({"error": str(e)})


if __name__ == "__main__":
    main()
