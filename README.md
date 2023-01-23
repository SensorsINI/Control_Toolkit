# Control_Toolkit

This folder contains general controller classes conforming to an interface loosely based on the [OpenAI Gym Interface](https://arxiv.org/pdf/1606.01540).

The `Control_Toolkit_ASF/models_for_nn_as_mpc` folder contains exemplary neural networks for those controllers which need one, which they can import directly.

To use the toolkit, add this and the [SI_Toolkit](https://github.com/SensorsINI/SI_Toolkit) as Git submodules at the top level of your repository:

```
git submodule add https://github.com/SensorsINI/SI_Toolkit
git submodule update –init
```


## Repositories using the Toolkit

- <a href="https://github.com/SensorsINI/CartPoleSimulation/tree/reproduction_of_results_sep22" target="_blank">CartPole Simulator</a>
- <a href="https://github.com/frehe/ControlGym/tree/reproduction_of_results_sep22" target="_blank">ControlGym</a>
- <a href="https://github.com/neuromorphs/physical-cartpole/tree/reproduction_of_results_sep2022_physical_cartpole" target="_blank">Physical CartPole</a>
- <a href="https://github.com/F1Tenth-INI/f1tenth_development_gym" target="_blank">F1TENTH INI</a>


## Software Design and Motivation

### Folders

The motivation behind this toolkit is universality: A systems control algorithm should be implemented as much agnostic to the environment it is deployed on as possible. However, not all controllers can be formulated in such a general manner. For this reason, one may also add a folder `Control_Toolkit_ASF` for application-specific control as follows:

```
main_control_repository
L Control_Toolkit (submodule)
L Control_Toolkit_ASF (regular folder)
```

Find a template for the ASF folder within the toolkit. The template contains sample configuration files, whose structure should be kept consistent.

### Controller Design

Each controller is defined in a separate module. File name and class name should match and have the "controller_" prefix.

A controller can possess any of the following optional subcomponents:

- `Cost_Functions`: This folder contains a general base class and wrapper class for defining cost functions. You can define cost functions for your application in the `Control_Toolkit_ASF`.
- `Optimizers`: Interchangeable optimizers which return the cost-minimizing input given dynamics imposed by state predictions.
- `Predictors`: Defined in the `SI_Toolkit`.

This toolkit focuses on model-predictive control. Currently, only a `controller_mpc` is provided. You can however define other controllers in the application-specific files.


## List of available MPC optimizers with description
    
- `cem-tf`:
    A standard implementation of the cem algorithm. Samples a number of random input sequences from a normal distribution,
    then simulates them and selectes the 'elite' set of random inputs with lowest costs. The sampling distribution
    is fitted to the elite set and the procedure repeated a fixed number of times. 
    In the end the mean of the elite set is used as input.

- `cem-naive-grad-tf`:
    Same as cem, but between selecting the elite set and fitting the distribution, all input sequences in the elite
    set are refined with vanilla gradient descent. Re-Implementation of Bharadhwaj, Xie, Shkurti 2020.

- `rpgd-tf` (`formerly dist-adam-resamp2-tf`):
    Initially samples a set of control sequences, then optimizes them with the adam optimizer projecting control inputs,
    clipping inputs which violate the constraints. For the next time step, the optimizations are warm started with
    the solution from the last one. In regular intervals the only a subset of cheap control sequences are 
    warm started, while the other ones are resampled.

- `mppi-optimze-tf`:
    First find an initial guess of control sequence with the standard mppi approach. Then optimze it using the adam
    optimizer.


## Logging

The toolkit provides a uniform interface to log values in the controller. These values could for example be rollout trajectories or intermediate optimization results.

The `controller_mpc.step` method takes the `optimizer.logging_values` dictionary and copies it to its `controller_mpc.logs` dictionary in each step. The `template_controller` has two related attributes: `controller_logging` and `save_vars`. If the former is `true`, then the controller populates the fields of `save_vars` in the `template_controller.logs` dictionary with values if your controller calls `update_logs` within the `step` method.


## Examples of Application-Specific Controllers

We refer to the [Control_Toolkit_ASF of our CartPoleSimulation Project](https://github.com/SensorsINI/CartPoleSimulation/tree/master/Control_Toolkit_ASF/Controllers).
