from abc import ABC, abstractmethod
from Control_Toolkit.others.environment import EnvironmentBatched

import numpy as np

"""
For a controller to be found and imported by CartPoleGUI/DataGenerator it must:
1. Be in Controller folder
2. Have a name starting with "controller_"
3. The name of the controller class must be the same as the name of the file.
4. It must have __init__ and step methods

We recommend you derive it from the provided template.
See the provided examples of controllers to gain more insight.
"""


class template_controller(ABC):
    _controller_name: str = ""

    def __init__(self, environment: EnvironmentBatched, **kwargs):
        self.env_mock: EnvironmentBatched = environment
    
    @abstractmethod
    def step(self, s: np.ndarray, time=None):
        Q = None  # This line is not obligatory. ;-) Just to indicate that Q must me defined and returned
        pass
        return Q  # normed control input in the range [-1,1]

    # Optionally: A method called after an experiment.
    # May be used to print some statistics about controller performance (e.g. number of iter. to converge)
    def controller_report(self):
        raise NotImplementedError

    # Optionally: reset the controller after an experiment
    # May be useful for stateful controllers, like these containing RNN,
    # To reload the hidden states e.g. if the controller went unstable in the previous run.
    # It is called after an experiment,
    # but only if the controller is supposed to be reused without reloading (e.g. in GUI)
    def controller_reset(self):
        raise NotImplementedError
    
    @property
    def controller_name(self):
        name = self.__class__.__name__
        if name != "template_controller":
            return name.replace("controller_", "").replace("_", "-").lower()
        else:
            raise AttributeError()
