"""
Use this controller as a template for your own application-specific controllers.
If you want to design a new optimizer for MPC, then do that in Control_Toolkit.Optimizers instead of here.
"""

from SI_Toolkit.computation_library import NumpyLibrary, TensorFlowLibrary, PyTorchLibrary, TensorType
import numpy as np
import yaml
import os

from Control_Toolkit.Controllers import template_controller
from others.globals_and_utils import create_rng

# TODO: You can load and access config files here, like this:
# config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)
# The controller-specific config is loaded automatically in template_controller.__init__
# and you can use it as self.config_controller.


class controller_barebone(template_controller):
    _computation_library = ...  # TODO: One of NumpyLibrary, TensorflowLibrary, PyTorchLibrary.
    # This is required if the controller only supports one computation library.
    # In that case, you do not need to specify the computation_library in this controller's configuration.
    
    def configure(self):
        # TODO: Do things like defining a random number generator, loading models, computing constants, etc.
        # Examples:
        # seed = self.config_controller["seed"]
        # self.rng = create_rng(self.__class__.__name__, seed if seed==None else seed*2)
        # s = np.zeros((4,), dtype=np.float32)
        # u = 0.0
        pass

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        # The controller has to adapt when environment-related attributes such as target positions change
        # Updated targets etc. are passed as a dictionary updated_attributes
        self.update_attributes(updated_attributes)  # After this call, updated attributes are available as self.<<attribute_name>>
        
        # TODO: Implement your controller here
        # Examples:
        # Q = np.dot(-self.K, state).item()
        # Q *= (1 + self.p_Q * float(self.rng.uniform(self.action_low, self.action_high)))
        # Q = np.clip(Q, -1.0, 1.0, dtype=np.float32)
        # return Q
        return 0.0
