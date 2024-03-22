import glob
import logging
import os
from datetime import datetime
from importlib import import_module

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # all TF messages

import tensorflow as tf
import torch
from numpy.random import SFC64, Generator
from SI_Toolkit.Functions.TF.Compile import CompileTF, CompileAdaptive
from SI_Toolkit.computation_library import ComputationLibrary, NumpyLibrary, TensorFlowLibrary, PyTorchLibrary

LOGGING_LEVEL = logging.INFO


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name):
    """ Get instance of standard logger, giving it a name
    :param name: your name, e.g. __name__
    :returns: the logger

    """
    # logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(LOGGING_LEVEL)
    # create console handler
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    return logger


log = get_logger(__name__)


class torch_gen_like_TF:

    def __init__(self, seed):
        self.rng = torch.Generator().manual_seed(seed)

    def normal(self, shape, dtype):
        return torch.normal(mean=0.0, std=1.0, size=shape, generator=self.rng, dtype=dtype)


def create_rng(id: str, seed: int, computation_library: ComputationLibrary = NumpyLibrary):
    if seed == None:
        log.info(f"{id}: No random seed specified. Seeding with datetime.")
        seed = int(
            (datetime.now() - datetime(1970, 1, 1)).total_seconds() * 1000.0
        )  # Fully random

    if computation_library == NumpyLibrary:
        return Generator(SFC64(seed=seed))
    elif computation_library == TensorFlowLibrary:
        return tf.random.Generator.from_seed(seed=seed)
    elif computation_library == PyTorchLibrary:
        return torch_gen_like_TF(seed)



def find_optimizer_if_it_exists(optimizer_name: str) -> str:
    """Look up optimizer by name in the Control_Toolkit and Control_Toolkit_ASF folders.
    If the optimizer exists, returns its full name and path to it. Otherwise, returns False.
    """
    optimizer_name = optimizer_name.replace("-", "_")
    if not optimizer_name.startswith("optimizer"):
        optimizer_full_name = "optimizer_" + optimizer_name
    else:
        optimizer_full_name = optimizer_name
    
    # Find optimizer if it exists.
    optimizer_relative_paths = (
        glob.glob(f"{os.path.join('Control_Toolkit_ASF', 'Optimizers', optimizer_full_name)}.py")
        + glob.glob(f"{os.path.join('**', 'Control_Toolkit', 'Optimizers', optimizer_full_name)}.py", recursive=True)
    )
    if len(optimizer_relative_paths) > 1:
        raise ValueError(f"Optimizer {optimizer_full_name} must be in a unique location. {len(optimizer_relative_paths)} found.")
    elif len(optimizer_relative_paths) == 1:
        # If the optimizer exists, the controller is MPC
        return optimizer_full_name, optimizer_relative_paths[0]
    return False, None


def import_optimizer_by_name(optimizer_name: str) -> type:
    optimizer_full_name, optimizer_relative_path = find_optimizer_if_it_exists(optimizer_name)
    
    if optimizer_full_name:
        Optimizer = getattr(import_module(optimizer_relative_path.replace(".py", "").replace(os.sep, ".")), optimizer_full_name)
        return Optimizer
    else:
        raise ValueError(f"Optimizer {optimizer_full_name} not found.")


def import_controller_by_name(controller_name: str) -> type:
    """Search for the specified controller name in the following order:
    1) Control_Toolkit_ASF/Controllers/
    2) Control_Toolkit/Controllers/
    
    Important to note: The controller name can refer to one of the optimizers in Control_Toolkit/Optimizers/.
    In that case, the controller name is controller_mpc and the optimizer name is optimizer_<controller_name>.
    """
    controller_name = controller_name.replace("-", "_")
    if not controller_name.startswith("controller"):
        controller_full_name = "controller_" + controller_name
    else:
        controller_full_name = controller_name
    
    # Find optimizer if it exists.
    optimizer_full_name, _ = find_optimizer_if_it_exists(controller_name)
    if optimizer_full_name:
        # If the optimizer exists, the controller is MPC
        controller_full_name = "controller_mpc"
    
    # Search for the controller in the Controllers folders
    controller_relative_paths = (
        glob.glob(f"{os.path.join('Control_Toolkit_ASF', 'Controllers', controller_full_name)}.py")
        + glob.glob(f"{os.path.join('**', 'Control_Toolkit_ASF', 'Controllers', controller_full_name)}.py")
        + glob.glob(f"{os.path.join('**', 'Control_Toolkit', 'Controllers', controller_full_name)}.py", recursive=True)
    )
    
    assert len(controller_relative_paths) == 1, f"Controller {controller_full_name} must be in a unique location. {len(controller_relative_paths)} found."
    controller_relative_path = controller_relative_paths[0]

    log.info(f"Importing controller from {controller_relative_path}")
    
    Controller: type = getattr(import_module(controller_relative_path.replace(".py", "").replace(os.sep, ".")), controller_full_name)
    return Controller


def get_available_optimizer_names() -> "list[str]":
    """
    Method returns the list of optimizers available in the Control Toolkit or Application Specific Files
    """
    optimizer_files = (
        glob.glob(f"Control_Toolkit_ASF/Optimizers/optimizer_*.py")
        + glob.glob(f"**/Control_Toolkit/Optimizers/optimizer_*.py", recursive=True)
    )
    optimizer_names = [os.path.basename(item)[len('optimizer_'):-len('.py')].replace('_', '-') for item in optimizer_files]
    optimizer_names.sort()
    return optimizer_names
    
    
def get_available_controller_names() -> "list[str]":
    """
    Method returns the list of controllers available in the Control Toolkit or Application Specific Files
    """
    controller_files = (
        glob.glob(f"Control_Toolkit_ASF/Controllers/controller_*.py")
        + glob.glob(f"**/Control_Toolkit_ASF/Controllers/controller_*.py")
        + glob.glob(f"**/Control_Toolkit/Controllers/controller_*.py", recursive=True)
    )
    controller_names = ['manual-stabilization']
    controller_names.extend([os.path.basename(item)[len('controller_'):-len('.py')].replace('_', '-') for item in controller_files])
    
    controller_names.sort()
    return controller_names


def get_controller_name(controller_names=None, controller_name=None, controller_idx=None) -> type:
    """
    The method sets a new controller as the current controller.
    The controller may be indicated either by its name
    or by the index on the controller list (see get_available_controller_names method).
    """

    # Check if the proper information was provided: either controller_name or controller_idx
    if (controller_name is None) and (controller_idx is None):
        raise ValueError('You have to specify either controller_name or controller_idx to set a new controller.'
                            'You have specified none of the two.')
    elif (controller_name is not None) and (controller_idx is not None):
        raise ValueError('You have to specify either controller_name or controller_idx to set a new controller.'
                            'You have specified both.')
    else:
        pass
        
    if controller_names is None:
        controller_names = get_available_controller_names()

    # If controller name provided get controller index and vice versa
    if (controller_name is not None):
        try:
            controller_idx = controller_names.index(controller_name)
        except ValueError:
            print('{} is not in list. \n In list are: {}'.format(controller_name, controller_names))
            return None
    else:
        controller_name = controller_names[controller_idx]

    return controller_name, controller_idx


def get_optimizer_name(optimizer_names=None, optimizer_name=None, optimizer_idx=None) -> type:
    """
    The method sets a new optimizer as the current optimizer.
    The optimizer may be indicated either by its name
    or by the index on the optimizer list (see get_available_optimizer_names method).
    """

    # Check if the proper information was provided: either optimizer_name or optimizer_idx
    if (optimizer_name is None) and (optimizer_idx is None):
        raise ValueError('You have to specify either optimizer_name or optimizer_idx to set a new optimizer.'
                            'You have specified none of the two.')
    elif (optimizer_name is not None) and (optimizer_idx is not None):
        raise ValueError('You have to specify either optimizer_name or optimizer_idx to set a new optimizer.'
                            'You have specified both.')
    else:
        pass
        
    if optimizer_names is None:
        optimizer_names = get_available_optimizer_names()

    # If optimizer name provided get optimizer index and vice versa
    if (optimizer_name is not None):
        try:
            optimizer_idx = optimizer_names.index(optimizer_name)
        except ValueError:
            print('{} is not in list. \n In list are: {}'.format(optimizer_name, optimizer_names))
            return None
    else:
        optimizer_name = optimizer_names[optimizer_idx]

    return optimizer_name, optimizer_idx