# Control Toolkit

> **Note**: AI-generated on 14.11.2024, not human verified.

A modular Python toolkit for implementing advanced control algorithms with a focus on **Model Predictive Control (MPC)**. Supports multiple computation backends (TensorFlow, PyTorch, NumPy) with a unified interface based on the [OpenAI Gym Interface](https://arxiv.org/pdf/1606.01540).

## Features

- 🎯 **Multiple Control Strategies**: MPC, neural network imitators, remote/embedded controllers
- 🔧 **Pluggable Optimizers**: CEM, RPGD, MPPI, gradient-based methods, and more
- 📊 **Flexible Cost Functions**: Define custom objectives with consistent interface
- 🖥️ **Multi-Backend Support**: TensorFlow, PyTorch, NumPy
- 📡 **Remote Control**: ZeroMQ-based controller server
- 📈 **Built-in Logging**: Comprehensive trajectory and optimization metrics
- 🔌 **Hardware Integration**: Serial interface helpers for embedded systems

## Installation

Add as submodules to your repository:

```bash
git submodule add https://github.com/SensorsINI/SI_Toolkit
git submodule add <control-toolkit-repo-url> Control_Toolkit
git submodule update --init --recursive
pip install -r Control_Toolkit/requirements.txt
```

## Quick Start

```python
from Control_Toolkit.others.globals_and_utils import import_controller_by_name
import numpy as np

# Instantiate MPC controller
ControllerClass = import_controller_by_name("mpc")
controller = ControllerClass(
    environment_name="YourEnvironment",
    control_limits=(-1.0, 1.0),
    initial_environment_attributes={"target_position": 0.0}
)

# Configure with optimizer
controller.configure(optimizer_name="rpgd-tf")

# Run control loop
state = np.array([0.1, 0.2, 0.3, 0.4])
control_input = controller.step(state, time=0.0)
```

## Architecture

### Design Philosophy

The toolkit separates **general-purpose controllers** (environment-agnostic) from **application-specific controllers** (domain-tailored), promoting code reuse while maintaining flexibility.

### Folder Structure

```
your_project/
├── Control_Toolkit/              # This repository (submodule)
│   ├── Controllers/              # General-purpose controllers
│   ├── Optimizers/               # Optimization algorithms
│   ├── Cost_Functions/           # Cost function base classes
│   ├── controller_server/        # Remote controller server
│   └── others/                   # Utilities and helpers
├── Control_Toolkit_ASF/          # Your application-specific files
│   ├── Controllers/              # Custom controllers
│   ├── Cost_Functions/           # Custom cost functions
│   ├── config_controllers.yml    # Controller configurations
│   ├── config_optimizers.yml     # Optimizer configurations
│   └── config_cost_function.yml  # Cost function configurations
└── SI_Toolkit/                   # Predictors (submodule)
```

**Naming Convention**: Files and classes use `controller_<name>.py` or `optimizer_<name>.py` format and must inherit from their respective template classes.

## Available Controllers

| Controller | Description | File |
|------------|-------------|------|
| **MPC** | Model Predictive Control with pluggable optimizers | `controller_mpc.py` |
| **Neural Imitator** | Neural network-based controller | `controller_neural_imitator.py` |
| **Remote** | Client for remote controller server | `controller_remote.py` |
| **Embedded** | Interface for embedded hardware | `controller_embedded.py` |
| **C Controller** | Wrapper for C-based controllers | `controller_C.py` |

Define custom controllers in `Control_Toolkit_ASF/Controllers/`. Template available in `Control_Toolkit_ASF_Template/`.

## Available Optimizers

### Sampling-Based

| Optimizer | Description | Backend |
|-----------|-------------|---------|
| **cem-tf** | Cross-Entropy Method: samples random sequences, selects elites, refits distribution | TensorFlow |
| **cem-naive-grad-tf** | CEM + gradient refinement of elite samples [[Bharadhwaj et al., 2020]](https://arxiv.org/abs/2003.10768) | TensorFlow |
| **cem-gmm-tf** | CEM with Gaussian Mixture Model | TensorFlow |

### Gradient-Based

| Optimizer | Description | Backend |
|-----------|-------------|---------|
| **rpgd** | Resampling Parallel Gradient Descent: maintains population of trajectories, optimizes with Adam, periodic resampling [[Heetmeyer et al., 2023]](https://ieeexplore.ieee.org/document/10161233) | TensorFlow |
| **gradient-tf** | Pure gradient descent optimization | TensorFlow |

### Hybrid

| Optimizer | Description | Backend |
|-----------|-------------|---------|
| **mppi** | Model Predictive Path Integral + Adam refinement | TensorFlow |

### Other

| Optimizer | Description | Backend |
|-----------|-------------|---------|
| **random-action-tf** | Random action baseline | TensorFlow |
| **nlp-forces** | Nonlinear programming via FORCES Pro | NumPy |

## Controller Server

Run controllers as a service via ZeroMQ:

```bash
python -m Control_Toolkit.controller_server.controller_server
```

**Protocol** (endpoint: `tcp://*:5555`):

Request:
```json
{"rid": "request_id", "state": [0.1, 0.2], "time": 0.5, "updated_attributes": {}}
```

Response:
```json
{"rid": "request_id", "Q": 0.25}
```

Client example:
```python
import zmq, json
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")
socket.send_json({"rid": "1", "state": [0.1, 0.2], "time": 0.0})
control = socket.recv_json()["Q"]
```

## Logging

Enable logging in controller config: `controller_logging: true`

**Logged variables**: `Q_logged`, `J_logged`, `s_logged`, `u_logged`, `realized_cost_logged`, `trajectory_ages_logged`, `rollout_trajectories_logged`

**Access logs**:
```python
outputs = controller.get_outputs()
control_history = outputs['Q_logged']
```

## Configuration

Configuration files in `Control_Toolkit_ASF/`:

**config_controllers.yml**:
```yaml
mpc:
  optimizer: rpgd-tf
  computation_library: tensorflow
  device: cpu
  predictor_specification: "my_predictor"
  cost_function_specification: "tracking"
  controller_logging: true
```

**config_optimizers.yml**:
```yaml
rpgd-tf:
  num_rollouts: 100
  mpc_horizon: 20
  mpc_timestep: 0.05
  learning_rate: 0.01
  seed: 42
```

## Hardware Integration

**Serial Interface Helper** (for STM, ZYNQ boards):

```python
from Control_Toolkit.serial_interface_helper import get_serial_port, set_ftdi_latency_timer

port = get_serial_port(chip_type="STM")  # or "ZYNQ"
set_ftdi_latency_timer(port)  # Low-latency configuration
```

## Projects Using This Toolkit

- [CartPole Simulator](https://github.com/SensorsINI/CartPoleSimulation/tree/reproduction_of_results_sep22)
- [ControlGym](https://github.com/frehe/ControlGym/tree/reproduction_of_results_sep22)
- [Physical CartPole](https://github.com/neuromorphs/physical-cartpole/tree/reproduction_of_results_sep2022_physical_cartpole)
- [F1TENTH INI](https://github.com/F1Tenth-INI/f1tenth_development_gym)

See [CartPoleSimulation Control_Toolkit_ASF](https://github.com/SensorsINI/CartPoleSimulation/tree/master/Control_Toolkit_ASF/Controllers) for examples of application-specific controllers (do-mpc, LQR, etc.).

## Requirements

```
tensorflow, tensorflow_probability, numpy, torch, torchvision, gymnasium, watchdog
```

**Note**: Install only what you need (e.g., NumPy-only setups don't require TensorFlow/PyTorch).

## Citation

If using RPGD optimizer, please cite:

```bibtex
@inproceedings{heetmeyer2023rpgd,
  title={RPGD: A Small-Batch Parallel Gradient Descent Optimizer with Explorative 
         Resampling for Nonlinear Model Predictive Control},
  author={Heetmeyer, Frederik and Paluch, Marcin and Bolliger, Diego},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={3218--3224},
  year={2023},
  organization={IEEE},
  doi={10.1109/icra48891.2023.10161233}
}
```

---

**Template**: Use `Control_Toolkit_ASF_Template/` to create your application-specific folder structure.
